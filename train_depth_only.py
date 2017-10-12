from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np

import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.python.slim.learning import train_step

from imageselect_Dataloader_optflow import DataLoader
import os

from nets_optflow_depth import *

from utils import *

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("validate_dir", "./validation", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_integer("image_height", 240, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 720, "The size of of a sample batch")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_integer("batch_size", 10, "The size of of a sample batch")
flags.DEFINE_integer("max_steps", 20000, "Maximum number of training iterations")
flags.DEFINE_string("pretrain_weight_dir", "./pretrained", "Directory name to pretrained weights")
flags.DEFINE_integer("validation_check", 100, "Directory name to pretrained weights")
flags.DEFINE_integer("num_sources", 2, "number of sources")

FLAGS = flags.FLAGS

FLAGS.num_scales = 4
FLAGS.smooth_weight = 1
FLAGS.data_weight = 0.01
FLAGS.optflow_weight = 0
FLAGS.depth_weight = 1

FLAGS.resizedheight = 240
FLAGS.resizedwidth = 720

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


def deprocess_image(image):

    image = image*255.0
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)


def main(_):

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
      tf.gfile.MakeDirs(FLAGS.checkpoint_dir)



    with tf.Graph().as_default():
        

        #============================================
        #Load image and label
        #============================================
        with tf.name_scope("data_loading"):
            imageloader = DataLoader(FLAGS.dataset_dir,
                                     FLAGS.batch_size,
                                     FLAGS.image_height, 
                                     FLAGS.image_width,
                                     FLAGS.num_sources,
                                     FLAGS.num_scales,
                                     'train')
            image_left, image_right, label, intrinsics, tgt2src_projs,optflow,m_scale = imageloader.load_train_batch()


        #============================================
        #Define the model
        #============================================
        with tf.variable_scope("model") as scope:

            with tf.name_scope("depth_prediction"):
                #estimate depth and optical flow from both left and right image
                batch, height, width, _ = image_left.get_shape().as_list()

                #import pdb;pdb.set_trace()
                #Maybe also add optflow warped image
                proj_image_optflow= optflow_warp(
                    image_right, 
                    tf.expand_dims(optflow[:,:,:,0],-1),
                    tf.expand_dims(optflow[:,:,:,1],-1)
                    )


                #concatenate left and right image
                #img_pair = tf.concat([image_left, image_right, optflow, proj_image_optflow], axis=3)
                #estimate both depth and optical flow of the left image
                pred_disp, depth_net_endpoints_left = depth_net(image_left, 
                                                      is_training=True)

                pred_depth = pred_disp

            with tf.name_scope("validate_loading"):
                imageloader_val = DataLoader(FLAGS.dataset_dir,
                                        1,
                                        FLAGS.image_height, 
                                        FLAGS.image_width,
                                        FLAGS.num_sources,
                                        FLAGS.num_scales,
                                        'val')

                image_left_val, image_right_val, label_val, intrinsics_val, tgt2src_projs_val,optflow_val,m_scale_val = imageloader_val.load_train_batch()

                proj_image_optflow_val= optflow_warp(
                                                image_right_val, 
                                                tf.expand_dims(optflow_val[:,:,:,0],-1),
                                                tf.expand_dims(optflow_val[:,:,:,1],-1)
                                                )

                #concatenate left and right image
                #img_pair_val = tf.concat([image_left_val, image_right_val, optflow_val, proj_image_optflow_val], axis=3)
                #estimate both depth and optical flow of the left image

                scope.reuse_variables()
                pred_disp_val, depth_net_endpoints_left = depth_net(image_left_val, 
                                                  is_training=False)

                pred_depth_val = pred_disp_val

                
                
        #============================================   
        #Specify the loss function:
        #============================================

        with tf.name_scope("compute_loss"):
            depth_loss = 0
            optflow_loss = 0
            pixel_loss = 0
            smooth_loss = 0
            smooth_loss_optx=0
            smooth_loss_opty=0

            left_image_all = []
            right_image_all = []
            proj_image_all = []
            proj_image_right_all = []
            proj_error_stack_all = []
            optflow_x_all = []
            optflow_y_all = []

            for s in range(FLAGS.num_scales):

                #=======
                #Smooth loss
                #=======
                smooth_loss += FLAGS.smooth_weight/(2**s) * \
                    compute_smooth_loss(pred_depth[s])

                curr_label = tf.image.resize_area(label, 
                    [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])
                curr_image_left = tf.image.resize_area(image_left, 
                    [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])              
                # curr_image_right = tf.image.resize_area(image_right, 
                #     [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))]) 

                #=======
                #Depth loss
                #=======
                # di = tf.log(curr_label)-tf.log(pred_depth[s])
                # depth_loss += tf.sqrt(tf.reduce_mean(tf.multiply(di,di))+tf.reduce_mean(di)*tf.reduce_mean(di))*FLAGS.depth_weight/(2**s)

                curr_depth_error = tf.abs(curr_label - pred_depth[s])
                depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**s)


                #=======
                #Pixel loss
                #=======
                # curr_gt_proj_image, src_pixel_coords_gt,wmask= projective_inverse_warp(
                #     curr_image_right, 
                #     #tf.squeeze(1.0/pred_depth[s], axis=3),
                #     tf.squeeze(1.0/curr_label,axis=3), 
                #     tgt2src_projs[:,0,:,:],
                #     intrinsics[:,s,:,:]
                #     )
                # wmask = tf.concat([wmask,wmask,wmask],axis=3)


                # curr_proj_image, src_pixel_coords,_= projective_inverse_warp(
                #     curr_image_right, 
                #     tf.squeeze(1.0/pred_depth[s], axis=3),
                #     #tf.squeeze(1.0/curr_label,axis=3), 
                #     tgt2src_projs[:,0,:,:],
                #     intrinsics[:,s,:,:]
                #     )


                # curr_proj_error = tf.abs(curr_proj_image - curr_image_left)
                # pixel_loss += tf.reduce_mean(curr_proj_error)*FLAGS.data_weight/(2**s)
                                
                
                #========
                #For tensorboard visualize
                #========    
                left_image_all.append(curr_image_left)
                #proj_image_all.append(curr_proj_image)

            total_loss =  depth_loss + smooth_loss#+ optflow_loss + pixel_loss# + depth_loss + smooth_loss  + optflow_loss



       #=================================
       #Validation
       #=================================



        with tf.name_scope("compute_val_error"):
            depth_loss_val = 0
            smooth_loss_val = 0
            for s in range(FLAGS.num_scales):

                #=======
                #Smooth loss
                #=======
                smooth_loss_val += FLAGS.smooth_weight/(2**s) * \
                    compute_smooth_loss(pred_depth_val[s])

                curr_label_val = tf.image.resize_area(label_val, 
                    [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])


                #=======
                #Depth loss
                #=======
               
                di = tf.log(curr_label_val)-tf.log(pred_depth_val[s])
                depth_loss_val += tf.sqrt(tf.reduce_mean(tf.multiply(di,di))+tf.reduce_mean(di)*tf.reduce_mean(di))*FLAGS.depth_weight/(2**s)
                #curr_depth_error = tf.abs(curr_label_val - pred_depth_val[s])
                #depth_loss_val+= tf.reduce_mean(curr_depth_error_val)*FLAGS.depth_weight/(2**s)

            total_loss_val = smooth_loss_val+depth_loss_val
#            img_pair_val = tf.concat([image_left_val, image_right_val], axis=3)
#            pred_disp_val, depth_net_endpoints_left_val = disp_net(img_pair_val, 
#                                      is_training=True)
#            pred_depth_val = [tf.expand_dims(d[:,:,:,0],-1) for d in pred_disp_val]
#            pred_optflow_x_val = [tf.expand_dims(d[:,:,:,1],-1) for d in pred_disp_val]
#            pred_optflow_y_val = [tf.expand_dims(d[:,:,:,2],-1) for d in pred_disp_val]
#            for s in range(FLAGS.num_scales):
#                #=======
#                #Smooth loss
#                #=======
#                smooth_loss_val += FLAGS.smooth_weight/(2**s) * \
#                    compute_smooth_loss(1.0/pred_depth_val[s])
#
#                curr_label_val = tf.image.resize_area(label_val, 
#                    [int(224/(2**s)), int(224/(2**s))])
#                curr_image_left_val = tf.image.resize_area(image_left_val, 
#                    [int(224/(2**s)), int(224/(2**s))])              
#                curr_image_right_val = tf.image.resize_area(image_right_val, 
#                    [int(224/(2**s)), int(224/(2**s))]) 
#                #=======
#                #Depth loss
#                #=======
#                curr_depth_error = tf.abs(curr_label - pred_depth[s])
#                depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**s)
#                #=======
#                #Pixel loss
#                #=======
#                curr_proj_image, src_pixel_coords= projective_inverse_warp(
#                    curr_image_right, 
#                    tf.squeeze(1.0/pred_depth[s], axis=3),
#                    #tf.squeeze(1.0/curr_label,axis=3), 
#                    tgt2src_projs[:,0,:,:],
#                    intrinsics[:,s,:,:]
#                    )
#                curr_proj_error = tf.abs(curr_proj_image - curr_image_left)
#                pixel_loss += tf.reduce_mean(curr_proj_error)*FLAGS.data_weight/(2**s)
#                #=======
#                #Optflow loss
#                #=======
#                #Depth estimate optical flow
#                curr__gt_proj_image, src_pixel_coords_gt= projective_inverse_warp(
#                    curr_image_right, 
#                    #tf.squeeze(1.0/pred_depth[s], axis=3),
#                    tf.squeeze(1.0/curr_label,axis=3), 
#                    tgt2src_projs[:,0,:,:],
#                    intrinsics[:,s,:,:]
#                    )                
#                depth_optflow_x,depth_optflow_y = depth_optflow(src_pixel_coords_gt)
#                curr_optflow_error = tf.abs(pred_optflow_x[s] - depth_optflow_x)+tf.abs(pred_optflow_y[s] - depth_optflow_y)
#                optflow_loss += tf.reduce_mean(curr_optflow_error)*FLAGS.optflow_weight/(2**s)





        #============================================
        #Start training
        #============================================

        #import pdb;pdb.set_trace()
        with tf.name_scope("eval_op"):
            tf.summary.scalar('eval_losses/total_loss', total_loss_val)
            tf.summary.scalar('eval_losses/smooth_loss', smooth_loss_val)
            tf.summary.scalar('eval_losses/depth_loss', depth_loss_val)


        with tf.name_scope("train_op"):
            tf.summary.scalar('losses/total_loss', total_loss)
            tf.summary.scalar('losses/smooth_loss', smooth_loss)
            tf.summary.scalar('losses/depth_loss', depth_loss)
            # tf.summary.scalar('losses/pixel_loss', pixel_loss)

            tf.summary.image('optflow_project_image', \
                             proj_image_optflow)
            #import pdb;pdb.set_trace()
            for s in range(FLAGS.num_scales):
                
                tf.summary.image('scale%d_left_image' % s, \
                                 left_image_all[s])


                tf.summary.image('scale%d_pred_depth' % s,
                    pred_depth[s])

                # tf.summary.image('scale%d_projected_image_left_depth' % s, \
                #     proj_image_all[s])
            #tf.get_variable_scope().reuse_variables()
            # Specify the optimization scheme:
            # with tf.variable_scope("scope_global_step") as scope_global_step:
            #     global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            #with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate,FLAGS.beta1)

            # create_train_op that ensures that when we evaluate it to get the loss,
            # the update_ops are done and the gradient updates are computed.
            train_op = slim.learning.create_train_op(total_loss, optimizer)



            def train_step_fn(session, *args, **kwargs):

                total_loss, should_stop = train_step(session, *args, **kwargs)


                if train_step_fn.step % FLAGS.validation_check == 0:
                    #accuracy = session.run(train_step_fn.accuracy_validation)

                    total_loss_val = session.run(train_step_fn.total_loss_val)
                    print('Step %s - Loss: %f ' % (str(train_step_fn.step).rjust(6, '0'), total_loss_val))

                train_step_fn.step += 1

                return [total_loss, should_stop]

            train_step_fn.step = 0
            train_step_fn.total_loss_val = total_loss_val

            slim.learning.train(train_op, 
                                FLAGS.checkpoint_dir,
                                save_summaries_secs=20,
                                save_interval_secs = 60,
                                #init_fn=InitAssignFn,
                                train_step_fn=train_step_fn
                                )       







if __name__ == '__main__':
   tf.app.run()