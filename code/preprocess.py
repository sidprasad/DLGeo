import os
import numpy as np
import keras
import keras.utils.all_utils
from PIL import Image


# TODO: Keep working on this. Refer to https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator(keras.utils.all_utils.Sequence):
    """
    Generates data for Keras to consume.
    
    
    batch_size : Size of each batch
    shuffle: Whether we should shuffle data for each epoch.
    dim: Dimensions of data we want.
    
    base_path : Path where we will find our data
    """
    
    def __init__(self, base_path, batch_size=32, dims = (256, 256, 1), shuffle=True):
        'Initialization'


        assert(os.path.exists(base_path))
        ## TODO: Change this to just load names as all the files in the images path / labels path.
        self.base_path = base_path
        self.batch_size = batch_size

        self.dims = dims
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
        X = np.empty(self.batch_size)
        y = np.empty(self.batch_size) # Need to add the additional dimensions here.




        # Generate data
        for (i, fname) in enumerate(list_IDs_temp):

            train_image_name = os.path.join(self.image_path, fname)
            test_image_name = os.path.join(self.labels_path, fname)
            
            X.append(self.image_to_array(train_image_name))
            y.append(self.image_to_array(test_image_name))


            

        return X, y


    def image_to_array(self, path_to_image):
        im = Image.open(path_to_image)
        imarray = np.array(im)

        #Normalize
        return imarray / 255.0


### This is just for validation. Will remove.
if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    base_path = os.path.join(dirname, '../AerialImagesDataset/train') # Just set to train for now
    d = DataGenerator( base_path )

    a, b = d.__getitem__(0)
    pass