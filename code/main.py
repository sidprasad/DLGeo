from preprocess import DataGenerator
from unetmodel import UNetModel
import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def main():
    '''
    Stiches our modules together
    
    :return: None
    '''

    # May have to play with this a little to work with the unet model
    # UNET DOES NOT WORK WITH ARBITRARY INPUT SIZES. We need to split our images up to (256,256, 3)


    dirname = os.path.dirname(__file__)
    train_path = os.path.join(dirname, '../AerialImagesDataset/train') 
    training_generator = DataGenerator( train_path, img_dims = (256, 256, 3), label_dims = (256,256) ) ## Eventually these dims must change

    model = UNetModel()


    # Train model on dataset
    model.fit_generator(generator=training_generator,
                    use_multiprocessing=True,
                    workers=6)

    # I think the issue is that the output of the model is in 3 channels, 
    # instead of 1. Not sure why -- but a potential problem as flagged below.
    
    print('Saving model!')
    model.save(os.path.join(dirname, 'model'))


    image = training_generator.image_to_array(train_path + '/images/austin1.tif')

    # TEST HERE
    #image = tf.keras.preprocessing.image.load_img(train_path + '/images/austin1.tif', target_size = (256, 256))
    input_arr = np.array([image])  # Convert single image to a batch.
    predictions = model.predict(input_arr)

    plt.imshow((predictions[0] * 255).astype(np.uint8))
    plt.show()


if __name__ == '__main__':
    main()