    
import tensorflow as tf
import numpy as np
from tfutils import *
import os
import sys
import glob
import json

import depthmotionnet.datareader as datareader
from depthmotionnet.v2.blocks import *
from depthmotionnet.v2.losses import *

def make_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    batch_size = fx.get_shape().as_list()[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0.,0.,1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)
    return intrinsics

def get_multi_scale_intrinsics(intrinsics, num_scales):

    intrinsics_mscale = []


    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = intrinsics[:,0,0]/(2 ** s)
        fy = intrinsics[:,1,1]/(2 ** s)
        cx = intrinsics[:,0,2]/(2 ** s)
        cy = intrinsics[:,1,2]/(2 ** s)
        intrinsics_mscale.append(
            make_intrinsics_matrix(fx, fy, cx, cy))
    intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
    return intrinsics_mscale



def Demon_Dataloader():

    # set the path to the training h5 files here
    _data_dir = './data/benchmark/'

    top_output = ('IMAGE_PAIR', 'MOTION', 'DEPTH', 'INTRINSICS')

    batch_size = 10

    reader_params = {
        'batch_size': batch_size,
        'test_phase': False,
        'motion_format': 'ANGLEAXIS6',
        'inverse_depth': True,
        'builder_threads': 1,
        'scaled_width': 256,
        'scaled_height': 192,
        'norm_trans_scale_depth': False,        
        'top_output': top_output,
        'scene_pool_size': 650,
        'augment_rot180': 0.5,
        'augment_mirror_x': 0.5,        
        'builder_threads': 8,
    }

    # add data sources
    reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'sun3d_train*.h5')), 0.8)
    reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'rgbd_*_train.h5')), 0.2)
    reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'mvs_breisach.h5')), 0.3)
    reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'mvs_citywall.h5')), 0.3)
    #reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'mvs_achteck_turm.h5')), 0.003)
    reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'scenes11_train.h5')), 0.2)

    

    #import pdb;pdb.set_trace()
    reader_tensors = datareader.multi_vi_h5_data_reader(len(top_output), json.dumps(reader_params))
    data_tensors = reader_tensors[2]
    data_dict_all = dict(zip(top_output, data_tensors))
    num_test_iterations, current_batch_buffer, max_batch_buffer, current_read_buffer, max_read_buffer = tf.unstack(reader_tensors[0])

    # split the data for the individual towers
    data_dict = {}
    for k,v in data_dict_all.items():
        if k == 'INFO':
            continue # skip info vector
        data_dict[k] = v

    # dict of the losses of the current tower
    loss_dict = {}

    # data preprocessing
    with tf.name_scope("data_preprocess"):

        #import pdb;pdb.set_trace()
        rotation, translation = tf.split(value=data_dict['MOTION'], num_or_size_splits=2, axis=1)


        ground_truth = prepare_ground_truth_tensors(
            data_dict['DEPTH'],
            rotation,
            translation,
            data_dict['INTRINSICS'],
        )

        ground_truth['rotation'] = rotation
        ground_truth['translation'] = translation

        ground_truth['flow0'] = tf.transpose(ground_truth['flow0'], perm=[0,2,3,1])
        ground_truth['depth0'] = tf.transpose(ground_truth['depth0'], perm=[0,2,3,1])

        data_dict['IMAGE_PAIR'] = tf.transpose(data_dict['IMAGE_PAIR'], perm=[0,2,3,1])

        # zeros = tf.zeros([batch_size,1, 1])
        # ones = tf.ones([batch_size,1, 1])
        # fx = tf.expand_dims(tf.expand_dims(data_dict['INTRINSICS'][:,0], -1), -1)
        # fy = tf.expand_dims(tf.expand_dims(data_dict['INTRINSICS'][:,1], -1), -1)
        # cx = tf.expand_dims(tf.expand_dims(data_dict['INTRINSICS'][:,2], -1), -1)
        # cy = tf.expand_dims(tf.expand_dims(data_dict['INTRINSICS'][:,3], -1), -1)

        # intic1 = tf.concat([fx, zeros, cx], axis=2)
        # intic2 = tf.concat([zeros, fy, cy], axis=2)
        # intic3 = tf.concat([zeros, zeros, ones], axis=2)
        # intrinsics = tf.concat([intic1, intic2, intic3], axis=1)
        batch, height, width, _ = ground_truth['depth0'].get_shape().as_list()

        intrinsics=make_intrinsics_matrix(data_dict['INTRINSICS'][:,0]*width,data_dict['INTRINSICS'][:,1]*height,data_dict['INTRINSICS'][:,2]*width,data_dict['INTRINSICS'][:,3]*height)

        intrinsics = get_multi_scale_intrinsics(
            intrinsics, 4)
        


    return data_dict,ground_truth,intrinsics
