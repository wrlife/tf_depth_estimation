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
FLAGS.resizedheight = 240
FLAGS.resizedwidth = 720
FLAGS.checkpoint_dir="./checkpoints"




def main(_):




    with tf.Graph().as_default():
        #Load image and label
        x = tf.placeholder(shape=[None, FLAGS.resizedheight, FLAGS.resizedwidth, 11], dtype=tf.float32)

        img_list = sorted(glob(FLAGS.dataset_dir + '/*.jpg'))

        # # Define the model:
        with tf.variable_scope("model") as scope:
            with tf.name_scope("depth_prediction"):
            #with tf.variable_scope("depth_prediction") as scope:

                pred_disp, depth_net_endpoints = depth_net(x, 
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
                    #I = cv2.resize(I,(FLAGS.resizedwidth,FLAGS.resizedheight),interpolation = cv2.INTER_AREA)

                    
                    fh = open(img_list[i+1],'r')
                    I1 = pil.open(fh)
                    I1 = np.array(I1)


                    # fh = open(img_list[i],'r')
                    # I = pil.open(fh)
                    
                    # I = np.array(I)

                    # I1 = I[:,720:,:]
                    # I = I[:,:720,:]

                    #I1 = cv2.resize(I1,(FLAGS.resizedwidth,FLAGS.resizedheight),interpolation = cv2.INTER_AREA)                 

                    #I = I.resize((224,224),pil.ANTIALIAS)
                    
                    #I = I/255.0

                    #import pdb;pdb.set_trace()
                    #Optical flow
                    #flow=np.fromfile(FLAGS.dataset_dir+'/2342_2373.flo.flo', dtype=np.float32).reshape( I1.shape[0],I1.shape[1],2)
                    flow = util.readFlow(FLAGS.dataset_dir+'/z.flo')

                    #flow = np.zeros_like(flow)

                    x_coord = np.repeat(np.reshape(np.linspace(0, I1.shape[1]-1, I1.shape[1]),[1,I1.shape[1]]),I1.shape[0],0)
                    y_coord = np.repeat(np.reshape(np.linspace(0, I1.shape[0]-1, I1.shape[0]),[I1.shape[0],1]),I1.shape[1],1)


                    x_coord = x_coord+flow[:,:,0]
                    y_coord = y_coord+flow[:,:,1]
                    


                    #Bilinear interpolation
                    I_warp,_ = util.bilinear_interpolate(I1,np.reshape(x_coord,-1), np.reshape(y_coord,-1)) 
                    I_warp = I_warp.reshape(I1.shape[0],I1.shape[1],3)
                    I_warp = I_warp.astype(np.float32)
                    I = I.astype(np.float32)
                    I1 = I1.astype(np.float32)
                    
                    #I_warp = np.zeros_like(I1)
                    #I1 = np.zeros_like(I1)
                    #I = np.zeros_like(I1)
                    inputdata = np.concatenate([I,I1,flow,I_warp],axis=2)





                    pred = sess.run(pred_disp,feed_dict={x:inputdata[None,:,:,:]})
                    #test = np.asarray(pred[4])
                    import pdb;pdb.set_trace()
                    


                    #import pdb;pdb.set_trace()
                    #z = pred[0][0][0][0,:,:,:]
                    z = pred[0][0,:,:,0]
                    #z=cv2.resize(pred[0][0,:,:,:],(FLAGS.image_width,FLAGS.image_height),interpolation = cv2.INTER_CUBIC)
                    #z = cv2.bilateralFilter(z,9,75,75)
                    #z=1.0/z#[0][0,:,:,0]
                    z.astype(np.float32).tofile(FLAGS.output_dir+img_list[i].split('/')[-1]+'_z.bin')
                    
                    print("The %dth frame is processed"%(i))



if __name__ == '__main__':
   tf.app.run()
