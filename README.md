# Flexible Sampling for Long-tailed Skin Lesion Classification
by Lie Ju, Yicheng Wu, Lin Wang, Zhen Yu, Xin Zhao, Xin Wang, Paul Bonnington and Zongyuan Ge*

## Intro.
This is an official Pytorch implementation of our paper published in MICCAI 2022.

*[Flexible Sampling for Long-tailed Skin Lesion Classification](https://arxiv.org/abs/2204.03161)*

The code is avaliable now!

We have also provided other widely-used tricks for long-tailed learning. Feel free to use them!

## Usage
Please follow:

1. On the 8-class dataset, please download [ISIC 2019 dataset](https://www.kaggle.com/datasets/cdeotte/jpeg-isic2019-384x384).

3. Then you can use our provided split training, validation, and test data, all stored in a NumPy format. Please use 'np.load()' to extract the information.

3. For the 14-class dataset, please use [official API](https://github.com/ImageMarkup/isic-cli#isic-cli=) to download the extra images from the ISIC dataset gallery using our provided .csv file.

4. Also, the split data is provided. Please use 'np.load()' to extract the information.


## Citation
If you find this repository is helpful for your work, please cite our work:

```
@inproceedings{ju2022flexible,
  title={Flexible Sampling for Long-Tailed Skin Lesion Classification},
  author={Ju, Lie and Wu, Yicheng and Wang, Lin and Yu, Zhen and Zhao, Xin and Wang, Xin and Bonnington, Paul and Ge, Zongyuan},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2022: 25th International Conference, Singapore, September 18--22, 2022, Proceedings, Part III},
  pages={462--471},
  year={2022},
  organization={Springer}
}
```

## Reference
[1] Cui, Yin, et al. "Class-balanced loss based on effective number of samples." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

[2] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017.

[3] Zhang, Songyang, et al. "Distribution alignment: A unified framework for long-tail visual recognition." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

[4] Cao, Kaidi, et al. "Learning imbalanced datasets with label-distribution-aware margin loss." Advances in neural information processing systems 32 (2019).

[5] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).
