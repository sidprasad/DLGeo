from preprocess import DataGenerator
from unetmodel import UNetModel
import os


def main():
    '''
    Stiches our modules together
    
    :return: None
    '''

    # May have to play with this a little to work with the unet model
    current_input_size = (5000,5000,3)

    dirname = os.path.dirname(__file__)
    train_path = os.path.join(dirname, '../AerialImagesDataset/train') 
    training_generator = DataGenerator( train_path )

    model = UNetModel(input_size = current_input_size)


    # Train model on dataset
    model.fit_generator(generator=training_generator,
                    use_multiprocessing=True,
                    workers=6)


if __name__ == '__main__':
    main()