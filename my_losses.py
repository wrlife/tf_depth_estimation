from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np

from Demon_Data_loader import *
from tensorflow.contrib.slim.python.slim.learning import train_step
import os
from utils_lr import *
from tfutils import *


def get_reference_explain_mask( downscaling, FLAGS):
    opt = FLAGS
    tmp = np.array([0,1])
    ref_exp_mask = np.tile(tmp, 
                           (opt.batch_size, 
                            int(opt.resizedheight/(2**downscaling)), 
                            int(opt.resizedwidth/(2**downscaling)), 
                            1))
    ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
    return ref_exp_mask



def compute_smooth_loss(pred_disp,):
    def gradient(pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy
    dx, dy = gradient(pred_disp)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    smoothout = (tf.reduce_mean(tf.abs(dx2)) + tf.reduce_mean(tf.abs(dxdy)) + tf.reduce_mean(tf.abs(dydx)) + tf.reduce_mean(tf.abs(dy2)))
    return smoothout


def compute_exp_reg_loss( pred, ref):
    l = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(ref, [-1, 2]),
        logits=tf.reshape(pred, [-1, 2]))
    return tf.reduce_mean(l)


def compute_loss_single_depth(pred_depth,label,global_step,FLAGS):

    #=======
    #Depth loss
    #=======
    depth_loss = 0
    smooth_loss = 0
    loss_depth_sig = 0
    epsilon = 0.000001

    global_stepf = tf.to_float(global_step)
    depth_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_sig_weight, float(FLAGS.max_steps//3))
    #sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}
    #import pdb;pdb.set_trace()
    #pre_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth[0], perm=[0,3,1,2]), **sig_params)
    #gt_depth_sig = scale_invariant_gradient(tf.transpose(label, perm=[0,3,1,2]), **sig_params)
    #loss_depth_sig = depth_sig_weight* pointwise_l2_loss(pre_depth_sig, gt_depth_sig, epsilon=epsilon)


    for s in range(FLAGS.num_scales):

        #=======
        #Smooth loss
        #=======
        # smooth_loss += FLAGS.smooth_weight/(2**s) * \
        #     compute_smooth_loss(1.0/pred_depth[s])


        curr_label = tf.image.resize_area(label, 
            [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])

        
        sig_params = {'deltas':[2], 'weights':[1], 'epsilon': 0.001}
        #import pdb;pdb.set_trace()
        pre_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth[s], perm=[0,3,1,2]), **sig_params)
        gt_depth_sig = scale_invariant_gradient(tf.transpose(curr_label, perm=[0,3,1,2]), **sig_params)
        loss_depth_sig += depth_sig_weight* pointwise_l2_loss(pre_depth_sig, gt_depth_sig, epsilon=epsilon)
        
        
        

        diff = sops.replace_nonfinite(curr_label - pred_depth[s])
        curr_depth_error = tf.abs(diff)
        depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**s)
        #depth_loss += pointwise_l2_loss(curr_label, pred_depth[s], epsilon=epsilon)*FLAGS.depth_weight/(2**s)





    return depth_loss,smooth_loss,loss_depth_sig




def compute_loss_pairwise_depth(image_left, image_right,
                                pred_depth_left, pred_poses_right, pred_exp_logits_left,
                                pred_depth_right, pred_poses_left, pred_exp_logits_right,
                                gt_right_cam,
                                intrinsics,
                                label, FLAGS,
                                global_step):

    #============================================   
    #Specify the loss function:
    #============================================
    with tf.name_scope("compute_loss"):
        depth_loss = 0
        pixel_loss = 0
        smooth_loss = 0
        exp_loss = 0
        consist_loss = 0
        cam_loss = 0
        loss_depth_sig = 0

        epsilon = 0.000001

        left_image_all = []
        right_image_all = []

        proj_image_left_all = []
        proj_image_right_all = []

        proj_error_stack_all = []
        optflow_x_all = []
        optflow_y_all = []
        exp_mask_all = []


        #Adaptively changing weights
        #import pdb;pdb.set_trace()
        GT_proj_l2r = pose_vec2mat(gt_right_cam,'angleaxis')
        global_stepf = tf.to_float(global_step)
        depth_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_sig_weight, float(FLAGS.max_steps//3))
        #data_weight = ease_out_quad(global_stepf, 0, FLAGS.data_weight, float(FLAGS.max_steps//3))
        # depth_weight_consist = ease_out_quad_zero(global_stepf, 0, FLAGS.depth_weight_consist, float(FLAGS.max_steps//3))                      


        # proj_l2r = tf.cond(depth_weight_consist > 0, lambda: pose_vec2mat(pred_poses_right[:,0,:],'angleaxis'), lambda: GT_proj_l2r)
        # proj_r2l = tf.cond(depth_weight_consist > 0, lambda: pose_vec2mat(pred_poses_left[:,0,:],'angleaxis'), lambda: tf.matrix_inverse(GT_proj_l2r))
        
        #import pdb;pdb.set_trace()
        proj_l2r = pose_vec2mat(pred_poses_right[:,0,:],'angleaxis')
        proj_r2l = pose_vec2mat(pred_poses_left[:,0,:],'angleaxis')
 
        # proj_l2r_loss = pose_vec2mat(pred_poses_right[:,0,:],'angleaxis')
        # proj_r2l_loss = pose_vec2mat(pred_poses_left[:,0,:],'angleaxis')






        #=============
        #Compute camera loss
        #=============
        # cam_loss += tf.reduce_mean((gt_right_cam[:,0:3]-pred_poses_right[:,0,:][:,0:3])**2)*FLAGS.cam_weight_tran
        # cam_loss += tf.reduce_mean((gt_right_cam[:,3:]-pred_poses_right[:,0,:][:,3:])**2)*FLAGS.cam_weight_rot
        #import pdb;pdb.set_trace()
        cam_loss  += tf.reduce_mean((GT_proj_l2r[:,0:3,0:3]-proj_l2r[:,0:3,0:3])**2)*FLAGS.cam_weight_rot
        cam_loss  += tf.reduce_mean((tf.matrix_inverse(GT_proj_l2r)[:,0:3,0:3]-proj_r2l[:,0:3,0:3])**2)*FLAGS.cam_weight_rot
        cam_loss  += tf.reduce_mean((GT_proj_l2r[:,0:3,3]-proj_l2r[:,0:3,3])**2)*FLAGS.cam_weight_tran
        cam_loss  += tf.reduce_mean((tf.matrix_inverse(GT_proj_l2r)[:,0:3,3]-proj_r2l[:,0:3,3])**2)*FLAGS.cam_weight_tran



        for s in range(2,FLAGS.num_scales):
        

            
            
            #=======
            #Smooth loss
            #=======
            # smooth_loss += FLAGS.smooth_weight/(2**s) * \
            #     compute_smooth_loss(1.0/pred_depth_left[s-2])

            # smooth_loss += FLAGS.smooth_weight/(2**s) * \
            #     compute_smooth_loss(1.0/pred_depth_right[s-2])


            curr_label = tf.image.resize_area(label, 
                [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])
            curr_image_left = tf.image.resize_area(image_left, 
                [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))]) 
            curr_image_right = tf.image.resize_area(image_right, 
                [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))]) 


            
            #=======
            #sig depth loss
            #=======

            sig_params = {'deltas':[2], 'weights':[1], 'epsilon': 0.001}
            #import pdb;pdb.set_trace()
            pre_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth_left[s-2], perm=[0,3,1,2]), **sig_params)
            gt_depth_sig = scale_invariant_gradient(tf.transpose(curr_label, perm=[0,3,1,2]), **sig_params)
            loss_depth_sig += depth_sig_weight* pointwise_l2_loss(pre_depth_sig, gt_depth_sig, epsilon=epsilon)
            
            
            #import pdb;pdb.set_trace()
            #=======
            #Depth loss
            #=======
            diff = sops.replace_nonfinite(curr_label - pred_depth_left[s-2])
            curr_depth_error = tf.abs(diff)
            depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**(s))
            #depth_loss += pointwise_l2_loss(curr_label, pred_depth_left[s-2], epsilon=epsilon)*FLAGS.depth_weight/(2**s)




            #=======
            #Pixel loss
            #=======

                        
            curr_proj_image_left, src_pixel_coords_right,wmask_left, warp_depth_right,_= projective_inverse_warp(
                curr_image_right, 
                tf.squeeze(1.0/(curr_label), axis=3),
                GT_proj_l2r,
                intrinsics[:,s,:,:],
                format='matrix'
                )


            curr_proj_image_right, src_pixel_coords_left,wmask_right, warp_depth_left, _ = projective_inverse_warp(
                curr_image_left, 
                tf.squeeze(1.0/(pred_depth_right[s-2]), axis=3),
                tf.matrix_inverse(GT_proj_l2r),
                intrinsics[:,s,:,:],
                format='matrix'
                )

            curr_proj_error_left = tf.abs(curr_proj_image_left - curr_image_left)
            curr_proj_error_right = tf.abs(curr_proj_image_right - curr_image_right)


                                       

            # #===============
            # #exp mask
            # #===============

            # ref_exp_mask = get_reference_explain_mask(s,FLAGS)
            
            # if FLAGS.explain_reg_weight > 0:
            #     curr_exp_logits_left = tf.slice(pred_exp_logits_left[s-2], 
            #                                [0, 0, 0, 0], 
            #                                [-1, -1, -1, 2])
            #     exp_loss += FLAGS.explain_reg_weight * \
            #         compute_exp_reg_loss(curr_exp_logits_left,
            #                                   ref_exp_mask)
            #     curr_exp_left = tf.nn.softmax(curr_exp_logits_left)
            # # Photo-consistency loss weighted by explainability
            # if FLAGS.explain_reg_weight > 0:
            #     pixel_loss += tf.reduce_mean(curr_proj_error_left * \

            #         tf.expand_dims(curr_exp_left[:,:,:,1], -1))*FLAGS.data_weight/(2**(s))

            # exp_mask = tf.expand_dims(curr_exp_left[:,:,:,1], -1)                    
            # exp_mask_all.append(exp_mask)


            
            # if FLAGS.explain_reg_weight > 0:
            #     curr_exp_logits_right = tf.slice(pred_exp_logits_right[s-2], 
            #                                [0, 0, 0, 0], 
            #                                [-1, -1, -1, 2])
            #     exp_loss += FLAGS.explain_reg_weight * \
            #         compute_exp_reg_loss(curr_exp_logits_right,
            #                                   ref_exp_mask)
            #     curr_exp_right = tf.nn.softmax(curr_exp_logits_right)
            # # Photo-consistency loss weighted by explainability
            # if FLAGS.explain_reg_weight > 0:
            #     pixel_loss += tf.reduce_mean(curr_proj_error_right * \
            #         tf.expand_dims(curr_exp_right[:,:,:,1], -1))*FLAGS.data_weight/(2**(s))


            # if not depth_weight_consist is None:
            #     #=======
            #     #left right depth Consistent loss
            #     #=======
            #     right_depth_proj_error=consistent_depth_loss(1.0/(pred_depth_right[s-2]),warp_depth_right, src_pixel_coords_right)
            #     left_depth_proj_error=consistent_depth_loss(1.0/(pred_depth_left[s-2]),warp_depth_left, src_pixel_coords_left)

            #     consist_loss += tf.reduce_mean(right_depth_proj_error*tf.expand_dims(curr_exp_left[:,:,:,1], -1))*depth_weight_consist
            #     consist_loss += tf.reduce_mean(left_depth_proj_error*tf.expand_dims(curr_exp_right[:,:,:,1], -1))*depth_weight_consist



            #========
            #For tensorboard visualize
            #========    
            left_image_all.append(curr_image_left)
            right_image_all.append(curr_image_right)


            proj_image_left_all.append(curr_proj_image_left)
            proj_image_right_all.append(curr_proj_image_right)

            proj_error_stack_all.append(curr_proj_error_right)




    return depth_loss, cam_loss, pixel_loss, consist_loss, loss_depth_sig, exp_loss, left_image_all, right_image_all, proj_image_left_all,proj_image_right_all,proj_error_stack_all
