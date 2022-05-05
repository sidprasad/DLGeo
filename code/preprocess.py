import os
import numpy as np
import keras
import keras.utils.all_utils
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

#disable_eager_execution()


import gc

from memory_profiler import profile


class DataGenerator(keras.utils.all_utils.Sequence):
    """
    Generates data for Keras to consume.
    
    
    batch_size : Size of each batch
    shuffle: Whether we should shuffle data for each epoch.
    
    
    base_path : Path where we will find our data
    """
    
    def __init__(self, base_path, input_image_dims = (5000, 5000, 3), target_dims = (256, 256, 1), shuffle=True):
        'Initialization'


        assert(os.path.exists(base_path))

        h = input_image_dims[0]
        w = input_image_dims[1]

        t_h = target_dims[0]
        t_w = target_dims[1]

        self.ts =  ( h - (h % t_h), w - (w % t_w))




        self.base_path = base_path


        self.tiles_per_image = int((self.ts[0] * self.ts[1]) / (t_h * t_w))
        self.num_images = 1 ## JUST SET TO 1 FOR NOW
        self.batch_size = int(self.num_images * self.tiles_per_image)

        self.input_img_dims = input_image_dims
        self.target_dims = target_dims
        self.image_path = os.path.join(self.base_path, 'images')
        self.labels_path = os.path.join(self.base_path, 'gt')

        self.labels = os.listdir(self.labels_path)
        self.list_IDs = os.listdir(self.image_path)

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs * self.tiles_per_image) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        """returns data in the shape (batch_size, target_dims) and (batch_size, target_dims) """

        ## TODO: NEED TO CHANGE THIS #################

        try:
            indexes = self.indexes[index*self.num_images:(index+1)*self.num_images]

            list_IDs_temp = [self.list_IDs[k] for k in indexes]

            X, y = self.__data_generation(list_IDs_temp)

            return X, y
        finally:
            gc.collect()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    # TODO: Leak is here!!
    """
    
    Input: List of IDs of the target images.
    
    """

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        print('Generating data for ' + list_IDs_temp[0], flush=True)
        #Hack
        fname = list_IDs_temp[0]
        X = self.image_to_array(os.path.join(self.image_path, fname))
        y = self.image_to_array(os.path.join(self.labels_path, fname)) 

        print('DONE Generating data for ' + list_IDs_temp[0], flush=True)
        return X, y

    @profile
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


