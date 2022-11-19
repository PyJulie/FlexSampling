# FlexSampling
This is an official Pytorch implementation of the paper publicated in MICCAI 2022 *[Flexible Sampling for Long-tailed Skin Lesion Classification](https://arxiv.org/abs/2204.03161)*.

The codes will be released by the end of the year. For someone who would like to follow the benchmark on ISIC dataset right away, plese following the instructions below:


For 8-class dataset, please download [ISIC 2019 dataset](https://www.kaggle.com/datasets/cdeotte/jpeg-isic2019-384x384). Then you can use our provided split trainning, validation and test data which are all stored in a numpy format. Please use 'np.load()' to extract the information.

For 14-class dataset, please use our provided downloader to download the images from the ISIC dataset gallery. Also, the split data is provided. Please use 'np.load()' to extract the information.
