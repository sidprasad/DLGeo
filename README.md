# DLGeo

This project aims to calculate the Photovoltaic potential of Providence, RI using Image Segmentation via a U-Net based model. This is an attempt to re-implement the paper [Urban solar utilization potential mapping via deep learning technology: A case study of Wuhan, China](https://www.sciencedirect.com/science/article/pii/S0306261919307780?via%3Dihub) by Z. Huang et al, with a specific focus on Providence RI.




# References

- [Urban solar utilization potential mapping via deep learning technology: A case study of Wuhan, China](https://www.sciencedirect.com/science/article/pii/S0306261919307780?via%3Dihub) by Z. Huang et al.

- [Semantic Segmentation of GEE High Resolution Imagery](https://notebooks.githubusercontent.com/view/ipynb?color_mode=auto&commit=d0e308f624349ca7c68d1e4c42986de74bd0698d&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f676973742f6d6f727463616e74792f61633463343865336431306538393637366237666539623361366631626133612f7261772f643065333038663632343334396361376336386431653463343239383664653734626430363938642f73656d616e7469635f7365676d656e746174696f6e2e6970796e62&logged_in=false&nwo=mortcanty%2Fac4c48e3d10e89676b7fe9b3a6f1ba3a&path=semantic_segmentation.ipynb&repository_id=115050046&repository_type=Gist) by Mort Canty


- [Understanding U-NETs](https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406)


- [Loading large Datasets in Keras](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)


# Folder Structure

- `code-try-1` : Contains some U-Net code from our first attempt at implementing this paper.
- `utilites` : Some utility code we built to pre-process and tile data
- `segmentation_notebook.ipynb` : Our final logic/model in the form of a Colab notebook. This can also be found in its latest format [here](https://colab.research.google.com/drive/1Cy3RaU4K3rAgQRUv8U-JfsIHp3m7xYLq?usp=sharing). This borrows heavily from Mort Canty's segmentation algorithm, as listed above.