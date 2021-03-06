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


    dirname = os.path.dirname(__file__)
    train_path = os.path.join(dirname, '../AerialImagesDataset/train') 
    training_generator = DataGenerator( train_path ) ## Eventually these dims must change

    model = UNetModel()



    model.fit(training_generator,
                use_multiprocessing=True, # Set to true later
                epochs = num_epochs,
                steps_per_epoch = 100,

                workers=6
                )

    
    print('Saving model!')
    model.save(os.path.join(dirname, 'model'))






if __name__ == '__main__':
    main()