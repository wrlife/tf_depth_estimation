from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np

import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.python.slim.learning import train_step

from imageselect_Dataloader_tgt_src import DataLoader
import os

from nets import *

from utils import *

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("validate_dir", "./validation", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_integer("image_height", 240, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 720, "The size of of a sample batch")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_integer("batch_size", 30, "The size of of a sample batch")
flags.DEFINE_integer("max_steps", 20000, "Maximum number of training iterations")
flags.DEFINE_string("pretrain_weight_dir", "./pretrained", "Directory name to pretrained weights")
flags.DEFINE_integer("validation_check", 100, "Directory name to pretrained weights")
flags.DEFINE_integer("num_sources", 2, "number of sources")

FLAGS = flags.FLAGS

FLAGS.num_scales = 4
FLAGS.smooth_weight = 0.5
FLAGS.data_weight = 100


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
            image, src_image_stack, label, intrinsics, tgt2src_projs = imageloader.load_train_batch()


        #import pdb;pdb.set_trace()
        #============================================
        #Define the model
        #============================================
        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net(image, 
                                                  is_training=True)
            pred_depth = [1./d for d in pred_disp]


        #============================================   
        #Specify the loss function:
        #============================================

        with tf.name_scope("compute_loss"):
            depth_loss = 0
            pixel_loss = 0
            smooth_loss = 0

            tgt_image_all = []
            src_image_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []

            for s in range(FLAGS.num_scales):
                smooth_loss += FLAGS.smooth_weight/(2**s) * \
                    compute_smooth_loss(pred_disp[s])

                curr_label = tf.image.resize_area(label, 
                    [int(224/(2**s)), int(224/(2**s))])

                curr_tgt_image = tf.image.resize_area(image, 
                    [int(224/(2**s)), int(224/(2**s))])

                curr_src_image_stack = tf.image.resize_area(src_image_stack, 
                    [int(224/(2**s)), int(224/(2**s))])              


                curr_depth_error = tf.abs(curr_label - pred_disp[s])
                depth_loss += tf.reduce_mean(curr_depth_error)

                #inverse warping
                for i in range(FLAGS.num_sources):

                    curr_proj_image= projective_inverse_warp(
                        curr_src_image_stack[:,:,:,3*i:3*(i+1)], 
                        tf.squeeze(1.0/pred_disp[s], axis=3), 
                        tgt2src_projs[:,i,:,:],
                        intrinsics[:,s,:,:]
                        )

                    curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
                    pixel_loss += tf.reduce_mean(curr_proj_error)*FLAGS.data_weight/(2**s)

                    # Prepare images for tensorboard summaries
                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                    else:
                        proj_image_stack = tf.concat([proj_image_stack, 
                                                      curr_proj_image], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack, 
                                                      curr_proj_error], axis=3)                                 
                    


                tgt_image_all.append(curr_tgt_image)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)

            total_loss =  smooth_loss  +  depth_loss




        #============================================
        #Start training
        #============================================

        with tf.name_scope("train_op"):
            tf.summary.scalar('losses/total_loss', total_loss)
            tf.summary.scalar('losses/smooth_loss', smooth_loss)
            tf.summary.scalar('losses/depth_loss', depth_loss)
            tf.summary.scalar('losses/pixel_loss', pixel_loss)

            #import pdb;pdb.set_trace()
            for s in range(FLAGS.num_scales):
                tf.summary.histogram("scale%d_depth" % s, pred_depth[s])
                #tf.summary.image('scale%d_disparity_image' % s, 1./self.pred_depth[s])
                tf.summary.image('scale%d_target_image' % s, \
                                 tgt_image_all[s])
                for i in range(FLAGS.num_sources):
                    tf.summary.image(
                        'scale%d_source_image_%d' % (s, i), 
                        src_image_stack_all[s][:, :, :, i*3:(i+1)*3])
                    tf.summary.image('scale%d_projected_image_%d' % (s, i), 
                        proj_image_stack_all[s][:, :, :, i*3:(i+1)*3])
                    tf.summary.image('scale%d_proj_error_%d' % (s, i),
                        tf.expand_dims(proj_error_stack_all[s][:,:,:,i], -1))



            # Specify the optimization scheme:
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate,FLAGS.beta1)

            # create_train_op that ensures that when we evaluate it to get the loss,
            # the update_ops are done and the gradient updates are computed.
            train_op = slim.learning.create_train_op(total_loss, optimizer)


            def train_step_fn(session, *args, **kwargs):

                total_loss, should_stop = train_step(session, *args, **kwargs)


                if train_step_fn.step % FLAGS.validation_check == 0:
                    #accuracy = session.run(train_step_fn.accuracy_validation)
                    print('Step %s - Loss: %.2f ' % (str(train_step_fn.step).rjust(6, '0'), total_loss))

                train_step_fn.step += 1

                return [total_loss, should_stop]

            train_step_fn.step = 0

            slim.learning.train(train_op, 
                                FLAGS.checkpoint_dir,
                                save_summaries_secs=20,
                                save_interval_secs = 60,
                                #init_fn=InitAssignFn,
                                train_step_fn=train_step_fn
                                )       







if __name__ == '__main__':
   tf.app.run()