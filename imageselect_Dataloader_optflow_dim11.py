from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
import os

class DataLoader(object):
    def __init__(self,
                 dataset_dir,
                 batch_size,
                 image_height,
                 image_width,
                 num_source,
                 num_scales,
                 split):
        self.dataset_dir=dataset_dir
        self.batch_size=batch_size
        self.image_height=image_height
        self.image_width=image_width
        self.split=split
        self.num_source = num_source
        self.num_scales = num_scales
        self.resizedheight = 224
        self.resizedwidth = 224
        self.depth_dir = '/home/wrlife/project/Unsupervised_Depth_Estimation/scripts/data/goodimages/'


    def load_train_batch(self):

        seed = random.randint(0, 2**31 - 1)

        # Reads pfathes of images together with their labels
        file_list = self.read_labeled_image_list()

        image_paths_queue = tf.convert_to_tensor(file_list['image_file_list'], dtype=tf.string)
        depth_paths_queue = tf.convert_to_tensor(file_list['gt_depth_file_list'], dtype=tf.string)
        cam_paths_queue = tf.convert_to_tensor(file_list['cam_file_list'], dtype=tf.string)
        tgt2src_paths_queue = tf.convert_to_tensor(file_list['tgt2src_proj_list'], dtype=tf.string)

        # Makes an input queue
        input_queue = tf.train.slice_input_producer([image_paths_queue,
                                                     depth_paths_queue,
                                                     cam_paths_queue,
                                                     tgt2src_paths_queue],
                                                     num_epochs = 1500,
                                                     shuffle=True)

        #,tgt2scr_projs,m_scale
        image_all, label, intrinsics,tgt2scr_projs = self.read_images_from_disk(input_queue)



        # Form training batches
        src_image_stack, tgt_image ,label_batch,  intrinsics, tgt2scr_projs  = \
                tf.train.batch([image_all[ :, :, 3:], image_all[ :, :, :3], label, intrinsics,tgt2scr_projs], 
                               batch_size=self.batch_size)


        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales,tf.cast(self.resizedwidth,tf.float32)/self.image_width,tf.cast(self.resizedheight,tf.float32)/self.image_height)
        
        #import pdb;pdb.set_trace()
        return tgt_image, src_image_stack, label_batch, intrinsics, tgt2scr_projs
    

    def read_labeled_image_list(self):
        """Reads a .txt file containing pathes and labeles
        Args:
           image_list_file: a .txt file with one /path/to/image per line
           label: optionally, if set label will be pasted after each line
        Returns:
           List with all filenames in file image_list_file
        """


        with open(self.dataset_dir + '/%s.txt' % self.split, 'r') as f:
            frames = f.readlines()      

        #import pdb;pdb.set_trace()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1] + '_'+ x.split(' ')[2][:-1] for x in frames]
        image_file_list = [os.path.join(self.dataset_dir, subfolders[i], 
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(self.dataset_dir, subfolders[i], 
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        gt_depth_file_list = [os.path.join(self.depth_dir, subfolders[i],'normalized_sfs_results', 
            'frame'+frame_ids[i].split('_')[0] + '.jpg'+'_z.bin') for i in range(len(frames))]
        tgt2src_proj_list = [os.path.join(self.dataset_dir, subfolders[i], 
            frame_ids[i] + '_tgt2src_proj.txt') for i in range(len(frames))]
        

        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        all_list['gt_depth_file_list'] = gt_depth_file_list
        all_list['tgt2src_proj_list'] = tgt2src_proj_list


        return all_list


    def read_images_from_disk(self, input_queue):
        """Consumes a single filename and label as a ' '-delimited string.
        Args:
          filename_and_label_tensor: A scalar string tensor.
        Returns:
          Two tensors: the decoded image, and the string label.
        """

        #import pdb;pdb.set_trace()

        image_file = tf.read_file(input_queue[0])
        label_file = tf.read_file(input_queue[1])
        cam_file = tf.read_file(input_queue[2])
        tgt2src_proj_file = tf.read_file(input_queue[3])
     

        #=====================
        #Image
        #=====================
        if(self.split=="validate"):
            image_seq = tf.to_float(tf.image.resize_images(tf.image.decode_jpeg(image_file),[self.resizedheight,self.resizedwidth*2]))
        else:
            image_seq = tf.to_float(tf.image.resize_images(tf.image.decode_jpeg(image_file),[self.resizedheight,self.resizedwidth*2]))
            #image_seq = tf.to_float(tf.image.decode_jpeg(image_file))

        image_seq = image_seq/255.0-0.5
        tgt_image, src_image_stack = \
            self.unpack_image_sequence(
                image_seq, self.resizedheight, self.resizedwidth, self.num_source)
        image_all = tf.concat([tgt_image, src_image_stack], axis=2)

        #=====================
        #Depth
        #=====================
        label = tf.reshape(tf.decode_raw(label_file, tf.float32),[self.image_height,self.image_width,1])

        #import pdb;pdb.set_trace()
        label = tf.image.resize_images(label,[self.resizedheight,self.resizedwidth],method = tf.image.ResizeMethod.AREA)
        label = label

        label.set_shape([self.resizedheight,self.resizedwidth, 1])


        # #=====================
        # #Optflow
        # #=====================

        # optflow = tf.reshape(tf.decode_raw(optflow_file, tf.float32),[self.image_height,self.image_width,2])
        # optflow = tf.image.resize_images(optflow,[self.resizedheight,self.resizedwidth],method = tf.image.ResizeMethod.AREA)
        # optflow.set_shape([self.resizedheight,self.resizedwidth, 2])


        #=====================
        #Intrinsic camera
        #=====================
        rec_def = []
        for i in range(9):
            rec_def.append([1.])
        raw_cam_vec = tf.decode_csv(cam_file, 
                                    record_defaults=rec_def)

        intrinsics = tf.reshape(raw_cam_vec, [3, 3])


        #=====================
        #tgt2scr_project
        #=====================
        
        rec_def = []
        for i in range(6):
            rec_def.append([1.])
        raw_tgt2src_vec = tf.decode_csv(tgt2src_proj_file, 
                                    record_defaults=rec_def,field_delim = ',')

        #import pdb;pdb.set_trace()
        # raw_tgt2src_vec = raw_tgt2src_vec[:-1]
        # m_scale = raw_tgt2src_vec[-1]
        tgt2scr_projs = raw_tgt2src_vec
        #tgt2scr_projs = tf.reshape(raw_tgt2src_vec, [2, 4, 4])

        return image_all, label, intrinsics,tgt2scr_projs


    def data_augmentation(self, im, out_h, out_w):
            # Random scaling
           


            # Random cropping
            def random_cropping(im, out_h, out_w):
                # batch_size, in_h, in_w, _ = im.get_shape().as_list()
                in_h, in_w, _ = tf.unstack(tf.shape(im))
                offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
                offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
                im = tf.image.crop_to_bounding_box(
                    im, offset_y, offset_x, out_h, out_w)

                return im


            #im = random_cropping(im, out_h, out_w)

            im = tf.image.random_flip_left_right(im)

            im = tf.image.random_flip_up_down(im)

            #im = tf.per_image_standardization(im)

            #im = tf.cast(im, dtype=tf.uint8)

            return im


    def unpack_image_sequence(self, image_seq, image_height, image_width, num_source):

        # Assuming the center image is the target frame

        tgt_image = tf.slice(image_seq, 
                             [0, 0, 0], 
                             [-1, image_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, image_width, 0], 
                               [-1, image_width, -1])

        src_image_1.set_shape([image_height, 
                                   image_width, 
                                    3])
        tgt_image.set_shape([image_height, image_width, 3])
        return tgt_image, src_image_1



    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics


    def get_multi_scale_intrinsics(self,intrinsics, num_scales,x_resize_ratio,y_resize_ratio):

        intrinsics_mscale = []


        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)*x_resize_ratio
            fy = intrinsics[:,1,1]/(2 ** s)*y_resize_ratio
            cx = intrinsics[:,0,2]/(2 ** s)*x_resize_ratio
            cy = intrinsics[:,1,2]/(2 ** s)*y_resize_ratio
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale
