import os
import numpy as np
import tensorflow.keras # import keras
from tensorflow import keras
#from tensorflow import keras.utils.all_utils
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import PIL

#disable_eager_execution()


import gc

#from memory_profiler import profile


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras to consume.
    
    
    batch_size : Size of each batch
    shuffle: Whether we should shuffle data for each epoch.
    
    
    base_path : Path where we will find our data
    """
    
    def __init__(self, base_path, dims = (256, 256, 1), batch_size = 16, shuffle=True):
        'Initialization'


        assert(os.path.exists(base_path))
        self.base_path = base_path
        self.image_path = os.path.join(self.base_path, 'images')
        self.labels_path = os.path.join(self.base_path, 'gt')

        self.batch_size = batch_size
        self.dims = dims

        self.labels = os.listdir(self.labels_path)
        self.list_IDs = os.listdir(self.image_path)

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        """returns data in the shape (batch_size, target_dims) and (batch_size, target_dims) """

        ## TODO: NEED TO CHANGE THIS #################


        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    """
    
    Input: List of IDs of the target images.
    
    """

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        d = (self.batch_size,) + self.dims
        X = np.array([ 
                self.image_to_array(os.path.join(self.image_path, fname))
                    for fname in list_IDs_temp])
        y = np.array([ 
                self.image_to_array(os.path.join(self.labels_path, fname))
                    for fname in list_IDs_temp])     

        return X, y

    # TODO: Is there a mem leak here?

    #@profile
    def image_to_array(self, path_to_image):

        with PIL.Image.open(path_to_image) as im:
            return ( 1. * np.asarray(im) / 255.0)


        # return np.array(tf.keras.preprocessing.image.load_img(
        #     path_to_image, 
        #     color_mode="grayscale"))

