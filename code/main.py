from preprocess import DataGenerator
from unetmodel import UNetModel
import os
import tensorflow as tf
import numpy as np
import PIL
from unetsegmentationmodel import U_Net


def main():
    '''
    Stiches our modules together
    
    :return: None
    '''

    num_epochs = 2
    new_gen = True

    # May have to play with this a little to work with the unet model
    # UNET DOES NOT WORK WITH ARBITRARY INPUT SIZES. We need to split our images up to (256,256, 3)


    dirname = os.path.dirname(__file__)
    train_path = os.path.join(dirname, '../AerialImagesDataset/train') 
    training_generator = DataGenerator( train_path ) ## Eventually these dims must change



    if new_gen:
        model = U_Net((256, 256, 1))
    else:
        model = UNetModel()
    
    model.fit(training_generator, epochs=num_epochs)









    


    # model.fit(training_generator,
    #             use_multiprocessing=True, # Set to true later
    #             epochs = num_epochs,
    #             steps_per_epoch = 50,

    #             workers=6
    #             )

    # callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
    # ]

    # # Train the model, doing validation at the end of each epoch.
    # epochs = 15


    # I think the issue is that the output of the model is in 3 channels, 
    # instead of 1. Not sure why -- but a potential problem as flagged below.
    
    print('Saving model!')
    model.save(os.path.join(dirname, 'model'))


    input_arr = training_generator.image_to_array(train_path + '/images/austin90.tif')

    # # TEST HERE
    # #image = tf.keras.preprocessing.image.load_img(train_path + '/images/austin1.tif', target_size = (256, 256))
    # input_arr = np.array([image])  # Convert single image to a batch.
    predictions = model.predict(input_arr)

    p = predictions[0]
    p = np.uint8(p * 255)
    
    mat = np.reshape(p, (256,256))

    im = PIL.Image.fromarray(mat)
    im.show()




if __name__ == '__main__':
    main()