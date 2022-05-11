from preprocess import DataGenerator
from unetmodel import UNetModel
import os
import tensorflow as tf




def main():
    '''
    Stiches our modules together
    
    :return: None
    '''

    num_epochs = 1

    # May have to play with this a little to work with the unet model
    # UNET DOES NOT WORK WITH ARBITRARY INPUT SIZES. We need to split our images up to (256,256, 3)


    dirname = os.path.dirname(__file__)
    train_path = os.path.join(dirname, '../AerialImagesDataset/train') 
    training_generator = DataGenerator( train_path ) ## Eventually these dims must change

    model = UNetModel()


    # Train model on dataset
    # model.fit_generator(generator=training_generator,
    #                 use_multiprocessing=True, # Set to true later
    #                 epochs = num_epochs,
    #                 steps_per_epoch = 100,

    #                 workers=6,
    #                 )

    model.fit(training_generator,
                use_multiprocessing=True, # Set to true later
                epochs = num_epochs,
                steps_per_epoch = 100,

                workers=6
                )

    # I think the issue is that the output of the model is in 3 channels, 
    # instead of 1. Not sure why -- but a potential problem as flagged below.
    
    print('Saving model!')
    model.save(os.path.join(dirname, 'model'))






if __name__ == '__main__':
    main()