from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
import PIL.Image as pil
from glob import glob
import cv2


import tensorflow.contrib.slim.nets

from imageselect_Dataloader_optflow import DataLoader
import os

from nets_optflow_depth import *
import util

slim = tf.contrib.slim
resnet_v2 = tf.contrib.slim.nets.resnet_v2

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("output_dir", "", "Dataset directory")
flags.DEFINE_integer("image_height", 240, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 720, "The size of of a sample batch")

FLAGS = flags.FLAGS
FLAGS.resizedheight = 224
FLAGS.resizedwidth = 224
FLAGS.checkpoint_dir="./checkpoints"




def main(_):




    with tf.Graph().as_default():
        #Load image and label
        x = tf.placeholder(shape=[None, FLAGS.resizedheight, FLAGS.resizedwidth, 6], dtype=tf.float32)

        img_list = sorted(glob(FLAGS.dataset_dir + '/*.jpg'))

        # # Define the model:
        with tf.variable_scope("model") as scope:
            with tf.name_scope("depth_prediction"):
            #with tf.variable_scope("depth_prediction") as scope:

                pred_disp, pred_poses, pred_exp_logits, depth_net_endpoints_left = depth_net(x, 
                                                                                            is_training=False)

                # predictions, end_points = resnet_v2.resnet_v2_50(x,
                #                                           global_pool=False,
                #                                           is_training=False
                #                                           )

                # multilayer = [end_points['model/resnet_v2_50/block4'], 
                #               end_points['model/resnet_v2_50/block2'],
                #               end_points['model/resnet_v2_50/block1'],
                #               end_points['model/depth_prediction/resnet_v2_50/block1/unit_3/bottleneck_v2/conv1'],
                #               end_points['model/depth_prediction/resnet_v2_50/conv1']]

                # pred_disp = upconvolution_net(multilayer,is_training=False)


                saver = tf.train.Saver([var for var in tf.model_variables()])
                #import pdb;pdb.set_trace()
                checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

            with tf.Session() as sess:

                saver.restore(sess, checkpoint)
                #import pdb;pdb.set_trace()
                for i in range(len(img_list)-1):

                    fh = open(img_list[i],'r')
                    I = pil.open(fh)
                    I = np.array(I)
                    I = cv2.resize(I,(FLAGS.resizedwidth,FLAGS.resizedheight),interpolation = cv2.INTER_AREA)

                    
                    fh = open(img_list[i+1],'r')
                    I1 = pil.open(fh)
                    I1 = np.array(I1)
                    I1 = cv2.resize(I1,(FLAGS.resizedwidth,FLAGS.resizedheight),interpolation = cv2.INTER_AREA)

                    inputdata = np.concatenate([I,I1],axis=2)


                    #import pdb;pdb.set_trace()


                    pred,pose = sess.run([pred_disp,pred_poses],feed_dict={x:inputdata[None,:,:,:]})

                    np.savetxt(img_list[i]+'.txt', pose[0,0,:], fmt='%f')
                    #import pdb;pdb.set_trace()
                    # with open(img_list[i]+'.txt', 'w') as f:
                    #     f.write(pose[0,0,:])


                    #import pdb;pdb.set_trace()
                    z=cv2.resize(pred[0][0,:,:,0],(FLAGS.image_width,FLAGS.image_height),interpolation = cv2.INTER_CUBIC)
                    z = cv2.bilateralFilter(z,9,75,75)
                    #z=1.0/z#[0][0,:,:,0]
                    z.astype(np.float32).tofile(FLAGS.output_dir+img_list[i].split('/')[-1]+'_z.bin')
                    
                    print("The %dth frame is processed"%(i))



if __name__ == '__main__':
   tf.app.run()
