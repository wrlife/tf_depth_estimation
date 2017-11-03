    
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


def Demon_Dataloader():

    # set the path to the training h5 files here
    _data_dir = '/playpen/research/Data/benchmark/traindata'

    top_output = ('IMAGE_PAIR', 'MOTION', 'DEPTH', 'INTRINSICS')

    batch_size = 8

    reader_params = {
        'batch_size': batch_size,
        'test_phase': False,
        'motion_format': 'ANGLEAXIS6',
        'inverse_depth': True,
        'builder_threads': 1,
        'scaled_width': 256,
        'scaled_height': 192,
        'norm_trans_scale_depth': True,        
        'top_output': top_output,
        'scene_pool_size': 650,
        'builder_threads': 8,
    }

    # add data sources
    reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'sun3d_train_0.1m_to_0.2m.h5')), 0.8)
    # reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'rgbd_*_train.h5')), 0.2)
    # reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'mvs_breisach.h5')), 0.3)
    # reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'mvs_citywall.h5')), 0.3)
    # #reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'mvs_achteck_turm.h5')), 0.003)
    # reader_params = datareader.add_sources(reader_params, glob.glob(os.path.join(_data_dir,'scenes11_train.h5')), 0.2)

    

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

    return data_dict,ground_truth
