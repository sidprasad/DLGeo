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
    
    def __init__(self, base_path, num_images=1, input_image_dims = (5000, 5000, 3), target_dims = (256, 256, 1), shuffle=True):
        'Initialization'


        assert(os.path.exists(base_path))

        h = input_image_dims[0]
        w = input_image_dims[1]

        t_h = target_dims[0]
        t_w = target_dims[1]

        self.ts =  ( h - (h % t_h), w - (w % t_w))




        self.base_path = base_path


        self.tiles_per_image = int((self.ts[0] * self.ts[1]) / (t_h * t_w))
        self.num_images = num_images
        self.batch_size = int(num_images * self.tiles_per_image)

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


        # Generate indexes of the batch 
        indexes = self.indexes[index*self.num_images:(index+1)*self.num_images]

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

        img_batch_dim = (self.batch_size,) + self.target_dims
        label_batch_dim = (self.batch_size,) + self.target_dims

        X = np.empty(img_batch_dim )
        y = np.empty(label_batch_dim) 

        # Read a batch from memory
        for (i, fname) in enumerate(list_IDs_temp):

            train_image_name = os.path.join(self.image_path, fname)
            test_image_name = os.path.join(self.labels_path, fname)
            
            start = i * self.tiles_per_image
            end = start + self.tiles_per_image
            # Now we get multiple images here. How do we expand?
            X[start: end] = (self.image_to_array(train_image_name))
            y[start: end]= (self.image_to_array(test_image_name))

        
        return X, y


    def image_to_array(self, path_to_image):

        im = tf.keras.preprocessing.image.load_img(
            path_to_image, 
            color_mode="grayscale",
            target_size=self.ts)


        tile_size = self.target_dims

        imarray = np.array(im)
        imarray = np.reshape(imarray, (self.ts[0], self.ts[1], 1))

        imarray = (imarray * 1.0) / 255.0
        
        # Now tile:
        s = split_image(imarray, tile_size)
        return s 



###where image3 is a 3-dimensional tensor (e.g. an image), and tile_size is a pair of values [H, W]
#  specifying the size of a tile. 
# The output is a tensor with shape [B, H, W, C]
def split_image(image3, tile_size):
    image_shape = tf.shape(image3)
    tile_rows = tf.reshape(image3, [image_shape[0], -1, tile_size[1], image_shape[2]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    return tf.reshape(serial_tiles, [-1, tile_size[1], tile_size[0], image_shape[2]])