# FlexSampling
This is an official Pytorch implementation of the paper published in MICCAI 2022 *[Flexible Sampling for Long-tailed Skin Lesion Classification](https://arxiv.org/abs/2204.03161)*.

The codes will be released by the end of the year. For someone who would like to follow the benchmark on the ISIC dataset right away, please follow the instructions below:


For the 8-class dataset, please download [ISIC 2019 dataset](https://www.kaggle.com/datasets/cdeotte/jpeg-isic2019-384x384). Then you can use our provided split training, validation, and test data, all stored in a NumPy format. Please use 'np.load()' to extract the information.

For the 14-class dataset, please use [official API](https://github.com/ImageMarkup/isic-cli#isic-cli=) to download the extra images from the ISIC dataset gallery using our provided .csv file. Also, the split data is provided. Please use 'np.load()' to extract the information.
