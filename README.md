# DLGeo


# Notes from Huang et Al


## Modules

Paper implements supervised learning with 5 modules:

1. Dataset Loader: Implements the loading of the database for training. 
    - It seems like multiple datasets were used, but all from the INRIA Aerial Image Labeling Dataset. This provides aerial images and their separate respective markers. 
    - We have 180 high-resolution images from different cities around the world (including drawings of buildings and non-buildings). 
    - As far as I can tell, all images are 5000x5000 pixels, with a DPI of 72 pixels per inch.
    
2. Neural Network Module: Implements U-Net architecture.   

3. Loss Function Module : Implements a special self-correcting loss function.

4. Trainer Module: Stiches everything together for training (sort of like what we've been doing in `main.py` in class assignments). This probably needs to be parallelizable, and leverage GPUs.

5. Tester Module: Same, but for testing. Can be a little simpler -- doesn't have to test in parallel.


# Hardware

From the paper: 

>"For the purposes of this paper, the
>NVIDIA graphics processer based on the general computing engine
>CUDA 9.0 is made use of to achieve massive parallel computing acceleration training."

# Some useful links:

- Understanding U-NET : https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406
- Loading large Datasets in Keras : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly