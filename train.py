from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np

import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.python.slim.learning import train_step

from imageselect_Dataloader import DataLoader
import os

from nets import *

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("validate_dir", "./validation", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_integer("image_height", 240, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 720, "The size of of a sample batch")
flags.DEFINE_float("learning_rate", 0.00002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_integer("batch_size", 20, "The size of of a sample batch")
flags.DEFINE_integer("max_steps", 20000, "Maximum number of training iterations")
flags.DEFINE_string("pretrain_weight_dir", "./pretrained", "Directory name to pretrained weights")
flags.DEFINE_integer("validation_check", 100, "Directory name to pretrained weights")

FLAGS = flags.FLAGS

FLAGS.num_scales = 4
FLAGS.smooth_weight = 1.0


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
	return tf.reduce_mean(tf.abs(dx2)) + \
	   tf.reduce_mean(tf.abs(dxdy)) + \
	   tf.reduce_mean(tf.abs(dydx)) + \
	   tf.reduce_mean(tf.abs(dy2))


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
									 'train')
			image,label = imageloader.load_train_batch()


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
			pixel_loss = 0

			smooth_loss = 0

			for s in range(FLAGS.num_scales):
				smooth_loss += FLAGS.smooth_weight/(2**s) * \
				    compute_smooth_loss(pred_disp[s])

				curr_label = tf.image.resize_area(label, 
					[int(224/(2**s)), int(224/(2**s))])                


				curr_depth_error = tf.abs(curr_label - pred_disp[s])
				pixel_loss += tf.reduce_mean(curr_depth_error)

			total_loss = pixel_loss + smooth_loss




		#============================================
		#Start training
		#============================================

		with tf.name_scope("train_op"):
			tf.summary.scalar('losses/total_loss', total_loss)

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