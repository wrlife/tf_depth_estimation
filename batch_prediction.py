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


flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("output_dir", "", "Dataset directory")
flags.DEFINE_integer("image_height", 240, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 720, "The size of of a sample batch")

FLAGS = flags.FLAGS

FLAGS.checkpoint_dir="./checkpoints"

def main(_):




	with tf.Graph().as_default():
		#Load image and label
		x = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)

		img_list = sorted(glob(FLAGS.dataset_dir + '/*.jpg'))

		# # Define the model:
		with tf.name_scope("Prediction"):

			pred_disp, depth_net_endpoints = disp_net(x, 
			                                      is_training=False)


			saver = tf.train.Saver([var for var in tf.model_variables()])
			#import pdb;pdb.set_trace()
			checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

			with tf.Session() as sess:

				saver.restore(sess, checkpoint)

				for i in range(len(img_list)):

					fh = open(img_list[i],'r')
					I = pil.open(fh)
					I = np.array(I)
					I = cv2.resize(I,(224,224),interpolation = cv2.INTER_AREA)
					#I = I.resize((224,224),pil.ANTIALIAS)
					
					I = I/255.0

					#import pdb;pdb.set_trace()

					pred = sess.run(pred_disp,feed_dict={x:I[None,:,:,:]})

					#import pdb;pdb.set_trace()
					z=cv2.resize(1.0/pred[0][0,:,:,0],(FLAGS.image_width,FLAGS.image_height),interpolation = cv2.INTER_AREA)
					#z=1.0/z#[0][0,:,:,0]
					z.astype(np.float32).tofile(FLAGS.output_dir+img_list[i].split('/')[-1]+'_z.bin')
					
					print("The %dth frame is processed"%(i))



if __name__ == '__main__':
   tf.app.run()
