from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np

import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.python.slim.learning import train_step

#from imageselect_Dataloader_optflow_dim11 import DataLoader
from Demon_Data_loader import *

import os

from nets_optflow_depth import *

from utils_lr import *
import lmbspecialops as sops
from depthmotionnet.v2.losses import *

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("validate_dir", "./validation", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_integer("image_height", 192, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 256, "The size of of a sample batch")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_integer("batch_size", 10, "The size of of a sample batch")
flags.DEFINE_string("pretrain_weight_dir", "./pretrained", "Directory name to pretrained weights")
flags.DEFINE_integer("validation_check", 100, "Directory name to pretrained weights")
flags.DEFINE_integer("num_sources", 2, "number of sources")


flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_integer("max_steps", 600000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")



FLAGS = flags.FLAGS

FLAGS.num_scales = 4
FLAGS.smooth_weight = 5
FLAGS.data_weight = 1000
FLAGS.optflow_weight = 0
FLAGS.depth_weight = 500
FLAGS.sig_depth_weight = 1500

FLAGS.explain_reg_weight = 30
FLAGS.cam_consist_weight = 10

FLAGS.consist_weight = 10
FLAGS.cam_weight_rot = 100
FLAGS.cam_weight_tran = 10

FLAGS.resizedheight = 192
FLAGS.resizedwidth = 256

slim = tf.contrib.slim
resnet_v2 = tf.contrib.slim.nets.resnet_v2

def compute_smooth_loss(pred_disp):
    def gradient(pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy
    dx, dy = gradient(pred_disp)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    smoothout = (tf.reduce_mean(tf.abs(dx2)) + tf.reduce_mean(tf.abs(dxdy)) + tf.reduce_mean(tf.abs(dydx)) + tf.reduce_mean(tf.abs(dy2)))
    return smoothout


def deprocess_image(image,FLAGS):

    image = image*255.0
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)

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

def compute_exp_reg_loss( pred, ref):
    l = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(ref, [-1, 2]),
        logits=tf.reshape(pred, [-1, 2]))
    return tf.reduce_mean(l)

def main(_):

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
      tf.gfile.MakeDirs(FLAGS.checkpoint_dir)



    with tf.Graph().as_default():
        

        #============================================
        #Load image and labels
        #============================================
        with tf.name_scope("data_loading"):
            # imageloader = DataLoader(FLAGS.dataset_dir,
            #                          FLAGS.batch_size,
            #                          FLAGS.image_height, 
            #                          FLAGS.image_width,
            #                          FLAGS.num_sources,
            #                          FLAGS.num_scales,
            #                          'train')

            # image_left, image_right, label, intrinsics = imageloader.load_train_batch()

            data_dict,ground_truth,intrinsics = Demon_Dataloader()
            image_left, image_right = tf.split(value=data_dict['IMAGE_PAIR'], num_or_size_splits=2, axis=3)
            label = ground_truth['depth0']
            gt_right_cam = tf.concat([ground_truth['translation'],ground_truth['rotation']],axis=1)

        #============================================
        #Define the model
        #============================================
        with tf.variable_scope("model") as scope:

            with tf.name_scope("depth_prediction"):


                #Using left right to predict
                inputdata = tf.concat([image_left, image_right], axis=3)

                pred_depth_left, pred_poses_right, pred_exp_logits_left, depth_net_endpoints = depth_net(inputdata,                                                     
                                                                                                    is_training=True)                 
                #Using right left to predict
                scope.reuse_variables()
                inputdata = tf.concat([image_right, image_left], axis=3)
                pred_depth_right, pred_poses_left, pred_exp_logits_right, depth_net_endpoints_right = depth_net(inputdata,                                                     
                                                                                                    is_training=True)

                #import pdb;pdb.set_trace()
                # pred_depth_left = [tf.expand_dims(d[:,:,:,0],-1) for d in pred_depth]
                # pred_depth_right =  [tf.expand_dims(d[:,:,:,1],-1) for d in pred_depth]
                # pred_poses_left = pred_poses[:,1,:]
                # pred_poses_right = pred_poses[:,0,:]


        #============================================   
        #Specify the loss function:
        #============================================

        with tf.name_scope("compute_loss"):
            depth_loss = 0
            optflow_loss = 0
            pixel_loss = 0
            smooth_loss = 0
            exp_loss = 0
            consist_loss = 0
            cam_consist_loss = 0 
            cam_loss = 0
            sig_depth_loss = 0
            epsilon = 0.00001

            left_image_all = []
            right_image_all = []

            proj_image_left_all = []
            proj_image_right_all = []

            proj_error_stack_all = []
            optflow_x_all = []
            optflow_y_all = []
            exp_mask_all = []




            # =========
            # left right camera Consistent loss
            # =========
            # cam_consist_loss = tf.reduce_mean((pred_poses_right+pred_poses_left)**2)*FLAGS.cam_consist_weight
            
            #=========
            #Cam pose loss
            #=========

            gt_proj_l2r = pose_vec2mat(gt_right_cam,'angleaxis')
            pose_left2right = pose_vec2mat(pred_poses_right[:,0,:],'angleaxis')
            pose_righ2left = pose_vec2mat(pred_poses_left[:,0,:],'angleaxis')

            cam_loss  += tf.reduce_mean((gt_proj_l2r[:,0:3,0:3]-pose_left2right[:,0:3,0:3])**2)*FLAGS.cam_weight_rot
            cam_loss  += tf.reduce_mean((tf.matrix_inverse(gt_proj_l2r)[:,0:3,3]-pose_righ2left[:,0:3,3])**2)*FLAGS.cam_weight_tran



            
            #=========
            #Gradient loss
            #=========
            
            sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}

            pr_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth_left[0], perm=[0,3,1,2]), **sig_params)
            gt_depth_sig = scale_invariant_gradient(tf.transpose(label, perm=[0,3,1,2]), **sig_params)

            sig_depth_loss += FLAGS.sig_depth_weight* pointwise_l2_loss(pr_depth_sig, gt_depth_sig, epsilon=epsilon)


            for s in range(FLAGS.num_scales):

                #=======
                #Smooth loss
                #=======
                # smooth_loss += FLAGS.smooth_weight/(2**s) * \
                #     compute_smooth_loss(1.0/pred_depth_left[s])

                # smooth_loss += FLAGS.smooth_weight/(2**s) * \
                #     compute_smooth_loss(1.0/pred_depth_right[s])

                curr_label = tf.image.resize_area(label, 
                    [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])
                curr_image_left = tf.image.resize_area(image_left, 
                    [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))]) 
                curr_image_right = tf.image.resize_area(image_right, 
                    [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))]) 


                


                #=======
                #Depth loss
                #=======

                diff = sops.replace_nonfinite(curr_label - pred_depth_left[s])
                curr_depth_error = tf.abs(diff)
                depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**s)

                



                #=======
                #Pixel loss
                #=======
                # wmask = tf.concat([wmask,wmask,wmask],axis=3)

                #import pdb;pdb.set_trace()

                curr_proj_image_left, src_pixel_coords_right,wmask_left, warp_depth_right,_= projective_inverse_warp(
                    curr_image_right, 
                    tf.squeeze(1.0/pred_depth_left[s], axis=3),
                    #pred_poses_right[:,0,:],
                    pose_left2right,
                    intrinsics[:,s,:,:],
                    format='matrix')

                #wmask_left = tf.concat([wmask_left,wmask_left,wmask_left],axis=3)
                curr_proj_error_left = tf.abs(curr_proj_image_left - curr_image_left)


                curr_proj_image_right, src_pixel_coords_left,wmask_right, warp_depth_left,_= projective_inverse_warp(
                    curr_image_left, 
                    tf.squeeze(1.0/pred_depth_right[s], axis=3),
                    #pred_poses_left[:,0,:],
                    pose_righ2left,
                    intrinsics[:,s,:,:],
                    format='matrix')

                #wmask_right = tf.concat([wmask_right,wmask_right,wmask_right],axis=3)
                curr_proj_error_right = tf.abs(curr_proj_image_right - curr_image_right)



                #import pdb;pdb.set_trace()

                


                # =========
                # left right camera Consistent loss
                # =========

                    #cam_consist_loss = tf.reduce_mean((tf.matrix_inverse(pose_left2right)-pose_righ2left)**2)*FLAGS.cam_consist_weight



                #===============
                #exp mask
                #===============
                ref_exp_mask = get_reference_explain_mask(s,FLAGS)
                
                if FLAGS.explain_reg_weight > 0:
                    curr_exp_logits_left = tf.slice(pred_exp_logits_left[s], 
                                               [0, 0, 0, 0], 
                                               [-1, -1, -1, 2])
                    exp_loss += FLAGS.explain_reg_weight * \
                        compute_exp_reg_loss(curr_exp_logits_left,
                                                  ref_exp_mask)
                    curr_exp_left = tf.nn.softmax(curr_exp_logits_left)
                # Photo-consistency loss weighted by explainability
                if FLAGS.explain_reg_weight > 0:
                    pixel_loss += tf.reduce_mean(curr_proj_error_left * \
                        tf.expand_dims(curr_exp_left[:,:,:,1], -1))*FLAGS.data_weight/(2**s)

                exp_mask = tf.expand_dims(curr_exp_left[:,:,:,1], -1)                    
                exp_mask_all.append(exp_mask)


                
                if FLAGS.explain_reg_weight > 0:
                    curr_exp_logits_right = tf.slice(pred_exp_logits_right[s], 
                                               [0, 0, 0, 0], 
                                               [-1, -1, -1, 2])
                    exp_loss += FLAGS.explain_reg_weight * \
                        compute_exp_reg_loss(curr_exp_logits_right,
                                                  ref_exp_mask)
                    curr_exp_right = tf.nn.softmax(curr_exp_logits_right)
                # Photo-consistency loss weighted by explainability
                if FLAGS.explain_reg_weight > 0:
                    pixel_loss += tf.reduce_mean(curr_proj_error_right * \
                        tf.expand_dims(curr_exp_right[:,:,:,1], -1))*FLAGS.data_weight/(2**s)



                #=======
                #left right depth Consistent loss
                #=======

                right_depth_proj_error=consistent_depth_loss(1.0/pred_depth_right[s],warp_depth_right, src_pixel_coords_right)
                left_depth_proj_error=consistent_depth_loss(1.0/pred_depth_left[s],warp_depth_left, src_pixel_coords_left)

                consist_loss += tf.reduce_mean(right_depth_proj_error*tf.expand_dims(curr_exp_left[:,:,:,1], -1))*FLAGS.consist_weight/(2**s)
                consist_loss += tf.reduce_mean(left_depth_proj_error*tf.expand_dims(curr_exp_right[:,:,:,1], -1))*FLAGS.consist_weight/(2**s)


                #import pdb;pdb.set_trace()
                #========
                #For tensorboard visualize
                #========    
                left_image_all.append(curr_image_left)
                right_image_all.append(curr_image_right)


                proj_image_left_all.append(curr_proj_image_left)
                proj_image_right_all.append(curr_proj_image_right)

                proj_error_stack_all.append(curr_proj_error_right)


            total_loss =  pixel_loss + smooth_loss  +exp_loss +cam_loss+  consist_loss + cam_consist_loss + depth_loss + sig_depth_loss



        #============================================
        #Start training
        #============================================


        with tf.name_scope("train_op"):
            tf.summary.scalar('losses/total_loss', total_loss)
            tf.summary.scalar('losses/smooth_loss', smooth_loss)
            tf.summary.scalar('losses/depth_loss', depth_loss)
            tf.summary.scalar('losses/pixel_loss', pixel_loss)
            tf.summary.scalar('losses/cam_loss', cam_loss)
            tf.summary.scalar('losses/exp_loss', exp_loss)
            tf.summary.scalar('losses/consist_loss', consist_loss)
            tf.summary.scalar('losses/cam_consist_loss', cam_consist_loss)
            tf.summary.scalar('losses/sig_depth_loss', sig_depth_loss)

            tf.summary.histogram('scale%d_pred_depth_left' % s,
                pred_depth_left[0])

            tf.summary.histogram('scale%d_pred_depth_right' % s,
                pred_depth_right[0])
            
            tf.summary.histogram('GT_left_depth', \
                             sops.replace_nonfinite(label))            

            for s in range(FLAGS.num_scales):
                
                tf.summary.image('scale%d_left_image' % s, \
                                 left_image_all[s])
                tf.summary.image('scale%d_right_image' % s, \
                                 right_image_all[s])
                tf.summary.image('scale%d_projected_image_left' % s, \
                                 proj_image_left_all[s])
                tf.summary.image('scale%d_projected_image_right' % s, \
                                 proj_image_right_all[s])

                tf.summary.image('scale%d_projected_error_left' % s, \
                                 proj_error_stack_all[s])                

                tf.summary.image('scale%d_pred_depth_left' % s,
                    1.0/pred_depth_left[s])

                tf.summary.image('scale%d_pred_depth_right' % s,
                    1.0/pred_depth_right[s])

                tf.summary.image('scale%d_exp_mask' % s,
                    exp_mask_all[s])

            #tf.get_variable_scope().reuse_variables()
            # Specify the optimization scheme:
            # with tf.variable_scope("scope_global_step") as scope_global_step:
            #     global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            #with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate,FLAGS.beta1)

            # create_train_op that ensures that when we evaluate it to get the loss,
            # the update_ops are done and the gradient updates are computed.
            train_op = slim.learning.create_train_op(total_loss, optimizer)



            global_step = tf.Variable(0, 
                                           name='global_step', 
                                           trainable=False)
            incr_global_step = tf.assign(global_step, 
                                              global_step+1)           


            saver = tf.train.Saver([var for var in tf.model_variables()])
            #import pdb;pdb.set_trace()
            with tf.Session() as sess:


                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir + '/sum',
                                                      sess.graph)


                tf.initialize_all_variables().run()
                tf.initialize_local_variables().run()

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess,coord=coord)
                
                if FLAGS.continue_train:
                    if FLAGS.init_checkpoint_file is None:
                        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                    else:
                        checkpoint = FLAGS.init_checkpoint_file
                    print("Resume training from previous checkpoint: %s" % checkpoint)
                    saver.restore(sess, checkpoint)


                for step in range(1, FLAGS.max_steps):
                    #print("steps %d" % (step))
                    fetches = {
                        "train": train_op,
                        "global_step": global_step,
                        "incr_global_step": incr_global_step
                    }

                    if step % FLAGS.summary_freq == 0:
                        fetches["loss"] = total_loss
                        fetches["summary"] = merged
                        fetches["GT_cam"] = gt_right_cam
                        fetches["est_cam"] = pred_poses_right
                        fetches["est_cam_left"] = pred_poses_left


                    results = sess.run(fetches)
                    gs = results["global_step"]

                    if step % FLAGS.summary_freq == 0:
                        train_writer.add_summary(results["summary"], gs)

                        print("steps: %d === loss: %.3f" \
                                % (gs,
                                    results["loss"]))

                        translation_rotation = results["GT_cam"]
                        print(translation_rotation[0])
                        print(results["est_cam"][0])
                        print(results["est_cam_left"][0])

                    if step % FLAGS.save_latest_freq == 0:
                        saver.save(sess, FLAGS.checkpoint_dir+'/model', global_step=step)


                coord.request_stop()
                coord.join(threads)                

            # def train_step_fn(session, *args, **kwargs):

            #     total_loss, should_stop = train_step(session, *args, **kwargs)


            #     if train_step_fn.step % FLAGS.validation_check == 0:
            #         #accuracy = session.run(train_step_fn.accuracy_validation)

            #         #total_loss_val = session.run(train_step_fn.total_loss_val)
            #         print('Step %s - Loss: %f ' % (str(train_step_fn.step).rjust(6, '0'), total_loss))

            #     train_step_fn.step += 1

            #     return [total_loss, should_stop]

            # train_step_fn.step = 0
            # # train_step_fn.total_loss_val = total_loss_val

            # slim.learning.train(train_op, 
            #                     FLAGS.checkpoint_dir,
            #                     save_summaries_secs=20,
            #                     save_interval_secs = 60,
            #                     #init_fn=InitAssignFn,
            #                     train_step_fn=train_step_fn
            #                     )       







if __name__ == '__main__':
   tf.app.run()