import os
import numpy as np
import keras
import keras.utils.all_utils
from PIL import Image
import tensorflow as tf


class DataGenerator(keras.utils.all_utils.Sequence):
    """
    Generates data for Keras to consume.
    
    
    batch_size : Size of each batch
    shuffle: Whether we should shuffle data for each epoch.
    
    
    base_path : Path where we will find our data
    """
    
    def __init__(self, base_path, batch_size=32, img_dims = (5000, 5000, 3), label_dims = (5000,5000), shuffle=True):
        'Initialization'


        assert(os.path.exists(base_path))

        self.base_path = base_path
        self.batch_size = batch_size

        self.img_dims = img_dims
        self.label_dims = label_dims
        self.image_path = os.path.join(self.base_path, 'images')
        self.labels_path = os.path.join(self.base_path, 'gt')

        self.labels = os.listdir(self.labels_path)
        self.list_IDs = os.listdir(self.image_path)

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        """returns data in the shape (batch_size, img_dims) and (batch_size, label_dims) """

        ## TODO: NEED TO CHANGE THIS #################


        # Generate indexes of the batch 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        ##########################################

        # Generate data
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

        img_batch_dim = (self.batch_size,) + self.img_dims
        label_batch_dim = (self.batch_size,) + self.label_dims

        X = np.empty(img_batch_dim )
        y = np.empty(label_batch_dim) 

        # Read a batch from memory
        for (i, fname) in enumerate(list_IDs_temp):

            train_image_name = os.path.join(self.image_path, fname)
            test_image_name = os.path.join(self.labels_path, fname)
            
            X[i] = (self.image_to_array(train_image_name))
            y[i] = (self.image_to_array(test_image_name))

        # TODO:  # UNET DOES NOT WORK WITH ARBITRARY INPUT SIZES. We need to split our images up to (256,256, 3) --> Currently (5000,5000,3)

        return X, y


    def image_to_array(self, path_to_image):


        # TODO: Should I be resizing images or breaking them up into tiles. Seems like information loss. 
        # For now, just resize for ease.


        im = Image.open(path_to_image)

        im = im.resize((256,256))

        imarray = np.array(im)
        imarray = imarray * 1.0

        if (len(imarray.shape) == 3):
        # GREYSCALE? TODO: REMOVE THIS!!!!
            imarray = tf.image.rgb_to_grayscale(imarray)
            imarray = imarray.numpy() * 1.0

        

        #Normalize
        return imarray / 255.0

