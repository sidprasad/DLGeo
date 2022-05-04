from preprocess import DataGenerator
from unetmodel import UNetModel
import os


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


if __name__ == '__main__':
    main()