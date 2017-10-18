from __future__ import division

from utils import *

import tensorflow as tf
import pprint
import random
import numpy as np

import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.python.slim.learning import train_step

#from imageselect_Dataloader_optflow import DataLoader
import os

from nets_optflow_depth import *



from Demon_Data_loader import *

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("validate_dir", "./validation", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_integer("image_height", 240, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 720, "The size of of a sample batch")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_integer("batch_size", 10, "The size of of a sample batch")
flags.DEFINE_string("pretrain_weight_dir", "./pretrained", "Directory name to pretrained weights")
flags.DEFINE_integer("validation_check", 100, "Directory name to pretrained weights")
flags.DEFINE_integer("num_sources", 2, "number of sources")

flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_integer("max_steps", 200000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")

FLAGS = flags.FLAGS

FLAGS.num_scales = 4
FLAGS.smooth_weight = 1
FLAGS.data_weight = 0.01
FLAGS.optflow_weight = 0
FLAGS.depth_weight = 1

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
            data_dict,ground_truth = Demon_Dataloader()


        #============================================
        #Define the model
        #============================================
        with tf.variable_scope("model") as scope:

            with tf.name_scope("depth_prediction"):
                #estimate depth and optical flow from both left and right image
                image_left, image_right = tf.split(value=data_dict['IMAGE_PAIR'], num_or_size_splits=2, axis=3)
                
                #import pdb;pdb.set_trace()


                
                #Maybe also add optflow warped image
                proj_image_optflow= optflow_warp(
                    image_right, 
                    tf.expand_dims(ground_truth['flow0'][:,:,:,0],-1),
                    tf.expand_dims(ground_truth['flow0'][:,:,:,1],-1)
                    )

                inputdata = tf.concat([data_dict['IMAGE_PAIR'],ground_truth['flow0'],proj_image_optflow],axis = 3)
                label = ground_truth['depth0']
                #concatenate left and right image
                #img_pair = tf.concat([image_left, image_right, optflow, proj_image_optflow], axis=3)
                #estimate both depth and optical flow of the left image
                pred_disp, depth_net_endpoints_left = depth_net(inputdata, 
                                                      is_training=True)

                pred_depth = pred_disp

 
                
                
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

                
                #========
                #For tensorboard visualize
                #========    
                left_image_all.append(curr_image_left)
                #proj_image_all.append(curr_proj_image)

            total_loss =  depth_loss + smooth_loss#+ optflow_loss + pixel_loss# + depth_loss + smooth_loss  + optflow_loss



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

            global_step = tf.Variable(0, 
                                           name='global_step', 
                                           trainable=False)
            incr_global_step = tf.assign(global_step, 
                                              global_step+1)           

        #import pdb;pdb.set_trace()
        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            if FLAGS.continue_train:
                if FLAGS.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                else:
                    checkpoint = FLAGS.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                saver.restore(sess, checkpoint)


            for step in range(1, FLAGS.max_steps):
                fetches = {
                    "train": train_op,
                    "global_step": global_step,
                    "incr_global_step": incr_global_step
                }

                if step % FLAGS.summary_freq == 0:
                    fetches["loss"] = total_loss


                results = sess.run(fetches)
                gs = results["global_step"]

                if step % FLAGS.summary_freq == 0:
                    

                    print("steps: %d === loss: %.3f" \
                            % (gs,
                                results["loss"]))
                if step % FLAGS.save_latest_freq == 0:
                    self.save(sess, FLAGS.checkpoint_dir, 'latest')


            # import pdb;pdb.set_trace()
            # image_tensor,pred_depth,ground_truth = sess.run([data_dict,pred_depth, ground_truth])

            # def train_step_fn(session, *args, **kwargs):

            #     total_loss, should_stop = train_step(session, *args, **kwargs)


            #     if train_step_fn.step % FLAGS.validation_check == 0:
            #         #accuracy = session.run(train_step_fn.accuracy_validation)

            #         #total_loss_val = session.run(train_step_fn.total_loss_val)
            #         print('Step %s - Loss: %f ' % (str(train_step_fn.step).rjust(6, '0'), total_loss))

            #     train_step_fn.step += 1

            #     return [total_loss, should_stop]

            # train_step_fn.step = 0
            # #train_step_fn.total_loss_val = total_loss_val
        # import pdb;pdb.set_trace()
        # slim.learning.train(train_op, 
        #                         FLAGS.checkpoint_dir,
        #                         save_summaries_secs=20,
        #                         save_interval_secs = 60,
        #                         #init_fn=InitAssignFn,
        #                         number_of_steps = 10
        #                         #train_step_fn=train_step_fn
        #                         )       







if __name__ == '__main__':
   tf.app.run()