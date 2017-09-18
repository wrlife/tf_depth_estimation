from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
import PIL.Image as pil
from glob import glob
import cv2


import tensorflow.contrib.slim.nets

from imageselect_Dataloader import DataLoader
import os

from nets import *
from utils import *
import util
from numpy.linalg import inv

import matplotlib.pyplot as plt


flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("output_dir", "", "Dataset directory")
flags.DEFINE_integer("image_height", 240, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 720, "The size of of a sample batch")
flags.DEFINE_float("learning_rate", 0.000000000002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")

FLAGS = flags.FLAGS
FLAGS.num_scales = 4
FLAGS.smooth_weight = 0.05

FLAGS.checkpoint_dir="./checkpoints"


def make_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    #import pdb;pdb.set_trace()
    batch_size = fx.get_shape().as_list()[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0.,0.,1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)
    return intrinsics


def get_multi_scale_intrinsics(intrinsics, num_scales,x_resize_ratio,y_resize_ratio):

    intrinsics_mscale = []


    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = intrinsics[:,0,0]/(2 ** s)*x_resize_ratio
        fy = intrinsics[:,1,1]/(2 ** s)*y_resize_ratio
        cx = intrinsics[:,0,2]/(2 ** s)*x_resize_ratio
        cy = intrinsics[:,1,2]/(2 ** s)*y_resize_ratio
        intrinsics_mscale.append(
            make_intrinsics_matrix(fx, fy, cx, cy))
    intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
    return intrinsics_mscale

def compute_smooth_loss(pred_disp):
	def gradient(pred):
		D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
		D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
		return D_dx, D_dy
	dx, dy = gradient(pred_disp)
	dx2, dxdy = gradient(dx)
	dydx, dy2 = gradient(dy)
	return tf.reduce_mean(tf.abs(dx2)) + \
	   tf.reduce_mean(tf.abs(dxdy)) + \
	   tf.reduce_mean(tf.abs(dydx)) + \
	   tf.reduce_mean(tf.abs(dy2))

def main(_):




	with tf.Graph().as_default():
		#Load image and label
		x1 = tf.placeholder(shape=[1, 224, 224, 3], dtype=tf.float32)
		x2 = tf.placeholder(shape=[1, 224, 224, 3], dtype=tf.float32)
		gt_x1_depth = tf.placeholder(shape=[1, 224, 224, 1], dtype=tf.float32)

		img_list = sorted(glob(FLAGS.dataset_dir + '/*.jpg'))

		pred_poses = tf.placeholder(shape=[1,4,4],dtype=tf.float32)
		intrinsics = tf.placeholder(shape=[1,3,3],dtype=tf.float32)

		#import pdb;pdb.set_trace()
		m_stack_intrinsics = get_multi_scale_intrinsics(intrinsics,FLAGS.num_scales,224.0/720.0,224.0/240.0)

		# # Define the model:
		with tf.name_scope("Prediction"):

			pred_disp, depth_net_endpoints = disp_net(x1, 
			                                      is_training=False)

		with tf.name_scope("compute_loss"):

			pixel_loss = 0

			smooth_loss = 0

			curr_proj_image = []

			for s in range(FLAGS.num_scales):
				smooth_loss += FLAGS.smooth_weight/(2**s) * \
				    compute_smooth_loss(pred_disp[s])

				curr_src_image = tf.image.resize_area(x1, 
				    [int(224/(2**s)), int(224/(2**s))]) 
				curr_tgt_image = tf.image.resize_area(x2, 
					[int(224/(2**s)), int(224/(2**s))])  

				curr_gt_x1_depth = tf.image.resize_area(gt_x1_depth, 
					[int(224/(2**s)), int(224/(2**s))])

				curr_proj_image.append(projective_inverse_warp(
				    curr_tgt_image, 
				    #tf.squeeze(1.0/curr_gt_x1_depth,axis=3),
				    tf.squeeze(1.0/pred_disp[s], axis=3), 
				    pred_poses, 
				    m_stack_intrinsics[:,s,:,:]))           


				curr_proj_error = tf.abs(curr_src_image - curr_proj_image[s])
				curr_depth_error = tf.abs(curr_gt_x1_depth - pred_disp[s])

				pixel_loss += tf.reduce_mean(curr_proj_error)
				pixel_loss += tf.reduce_mean(curr_depth_error)*FLAGS.smooth_weight

			total_loss = pixel_loss + smooth_loss






			saver = tf.train.Saver([var for var in tf.model_variables()])
			#import pdb;pdb.set_trace()
			checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)


			optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate,FLAGS.beta1)
			global_step = tf.Variable(0, name='global_step', trainable=False)
			train_op = optimizer.minimize(total_loss, global_step=global_step)			

			with tf.Session() as sess:

				init = tf.global_variables_initializer()
				sess.run(init)
				saver.restore(sess, checkpoint)



				I1 = pil.open(img_list[0])
				I1 = np.array(I1)
				I1 = cv2.resize(I1,(224,224),interpolation = cv2.INTER_AREA)

				I2 = pil.open(img_list[1])
				I2 = np.array(I2)
				I2 = cv2.resize(I2,(224,224),interpolation = cv2.INTER_AREA)

				m_intric = np.array([[567.239,0.0,376.04],[0.0,261.757,100.961],[0.0,0.0,1.0]]);


				z=np.fromfile('/home/wrlife/project/deeplearning/depth_prediction/case002/frame5946.jpg_z.bin',dtype=np.float32).reshape(FLAGS.image_height,FLAGS.image_width,1)

				z = 1.0/z
				
				z = cv2.resize(z,(224,224),interpolation = cv2.INTER_AREA)
				z = z[:,:,np.newaxis]
				

				#================================
				#
				#================================
				#Get camera translation matrix

				import pdb;pdb.set_trace()
				r1,t1,_,_=util.get_camera_pose(FLAGS.dataset_dir+"/images.txt",'frame5946.jpg')
				r2,t2,_,_=util.get_camera_pose(FLAGS.dataset_dir+"/images.txt",'frame5951.jpg')

				pad = np.array([[0, 0, 0, 1]])

				homo1 = np.append(r1,t1.reshape(3,1),1)
				homo1 = np.append(homo1,pad,0)

				homo2 = np.append(r2,t2.reshape(3,1),1)
				homo2 = np.append(homo2,pad,0)

				src2tgt_proj = inv(homo2)*homo1




				#pred_poses = np.array([src2tgt_proj[0,3],src2tgt_proj[1,3],src2tgt_proj[2,3],])
				I1 = I1/255.0
				I2 = I2/255.0				

				for i in range(1,100001):

					

					#I = I.resize((224,224),pil.ANTIALIAS)
					#import pdb;pdb.set_trace()2
					


					#import pdb;pdb.set_trace()

					_,pred,reproject_img,tmp_total_loss,tmp_proj = sess.run([train_op,pred_disp,curr_proj_image,total_loss,pred_poses],
																	feed_dict={x1:I1[None,:,:,:],
																			   x2:I2[None,:,:,:],
																			   pred_poses:src2tgt_proj[None,:,:],
																			   intrinsics:m_intric[None,:,:],
																			   gt_x1_depth:z[None,:,:,:]}
																			   )

					# z=cv2.resize(pred[0][0,:,:,0],(FLAGS.image_width,FLAGS.image_height))
					# z.astype(np.float32).tofile(FLAGS.output_dir+img_list[i].split('/')[-1]+'_z.bin')


					if i>=0 and i%100==0:
						print('Step %s - Loss: %.2f ' % (i, tmp_total_loss))

					if i == 1:

						test=reproject_img[0]
						test = test*255
						test = test.astype(np.uint8)

						plt.imshow(test[0])
						plt.show()
						import pdb;pdb.set_trace()
					

					#print("The %dth frame is processed"%(i))
					if i == 50000:
						z_out=cv2.resize(1.0/pred[0][0,:,:,0],(FLAGS.image_width,FLAGS.image_height))
						z_out.astype(np.float32).tofile(FLAGS.output_dir+img_list[0].split('/')[-1]+'_z.bin')

					if i == 1000:

						test=reproject_img[0]
						test = test*255
						test = test.astype(np.uint8)

						plt.imshow(test[0])
						plt.show()
						import pdb;pdb.set_trace()



if __name__ == '__main__':
   tf.app.run()