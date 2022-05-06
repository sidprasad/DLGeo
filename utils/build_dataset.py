import os
import numpy as np
import keras
import keras.utils.all_utils
import tensorflow as tf

import uuid

from PIL import Image
import matplotlib as plt

import gc


class BuildDataSet():
    """
    Generates data for Keras to consume.
    
    
    batch_size : Size of each batch
    shuffle: Whether we should shuffle data for each epoch.
    
    
    base_path : Path where we will find our data
    """
    
    def __init__(self, base_path, target_path, input_image_dims = (5000, 5000, 3), target_dims = (256, 256, 1)):


        assert(os.path.exists(base_path))
        assert(os.path.exists(target_path))

        self.base_path = base_path
        self.target_path = target_path

        h = input_image_dims[0]
        w = input_image_dims[1]

        t_h = target_dims[0]
        t_w = target_dims[1]

        self.ts =  ( h - (h % t_h), w - (w % t_w))


        self.input_img_dims = input_image_dims
        self.target_dims = target_dims
        self.image_path = os.path.join(self.base_path, 'images')
        self.labels_path = os.path.join(self.base_path, 'gt')


        self.image_target = os.path.join(self.target_path, 'images')
        self.labels_target = os.path.join(self.target_path, 'gt')

        self.labels = os.listdir(self.labels_path)
        self.list_IDs = os.listdir(self.image_path)



    


    def convert_all(self):

        for id in self.list_IDs:
            self.split_image_and_data(id)




    def split_image_and_data(self, fname):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        print('Splitting  ' + fname, flush=True)

        images = self.image_to_array(os.path.join(self.image_path, fname))
        labels = self.image_to_array(os.path.join(self.labels_path, fname))

        m = 361

        name = os.path.splitext(fname)[0] 

        i_n = os.path.join(self.image_target, name)
        l_n = os.path.join(self.labels_target, name)

        for j in range(m):



            i = images[j] * 255
            l = labels[j] * 255

            id = str(j) + '.tif'

            ip = i_n + id
            lp = l_n + id



            tf.keras.utils.save_img(ip, i)
            tf.keras.utils.save_img(lp, l)





 
    def image_to_array(self, path_to_image):

        tile_size = (1,) + self.target_dims

        reshaped_size = (-1,) + self.target_dims

        try:
            im = tf.keras.preprocessing.image.load_img(
                path_to_image, 
                color_mode="grayscale",
                target_size=self.ts)

            patches = tf.image.extract_patches(
                images = tf.expand_dims(tf.keras.utils.img_to_array(im), axis=0), 
                sizes = tile_size,
                strides = tile_size,
                rates = [1, 1, 1, 1], padding='SAME') # Have to think about the padding
            
            im = None

            gc.collect()

            return (tf.reshape(patches, reshaped_size) * 1.0) / 255.0
        finally:

            gc.collect()


d = BuildDataSet(base_path='/Users/siddharthaprasad/Desktop/DLGeo/AerialImagesDataset/train', target_path='/Users/siddharthaprasad/Desktop/DLGeo/utils/target')
d.convert_all()