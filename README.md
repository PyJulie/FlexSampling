# FlexSampling

Official PyTorch implementation of **"Flexible Sampling for Long-tailed Skin Lesion Classification"** (MICCAI 2022).

[[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-16437-8_46)

## Overview

This repository provides a modular toolkit for handling class imbalance in medical image classification, including:

- **Loss functions**: Focal Loss, Class-Balanced Loss, LDAM Loss, GRW Loss
- **Samplers**: Weighted Resampling, Class-Aware Sampler
- **Augmentation**: Mixup
- **Dataset**: ISIC skin lesion loader (NumPy format)

## Installation

```bash
pip install -e .

# With demo dependencies (timm, pyyaml, pandas, scikit-learn)
pip install -e ".[demo]"
```

## Quick Start

### As a library

```python
from flexsampling import build_loss, build_sampler, mixup_data, mixup_criterion

# Class-Balanced Focal Loss
cls_num_list = [500, 200, 100, 50, 20, 10, 5, 2]
criterion = build_loss("cb_focal", cls_num_list=cls_num_list, gamma=1.0, beta=0.9999)

# Class-Aware Sampler
sampler = build_sampler("class_aware", labels=train_dataset.labels)
loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

# Mixup augmentation in training loop
for images, targets in loader:
    images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=0.2)
    logits = model(images)
    loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
```

### Available components

| Type | Name | Description |
|------|------|-------------|
| Loss | `focal` | Focal Loss (Lin et al., ICCV 2017) |
| Loss | `cb_focal` | Class-Balanced Focal Loss (Cui et al., CVPR 2019) |
| Loss | `cb_softmax` | Class-Balanced Softmax CE |
| Loss | `cb_sigmoid` | Class-Balanced Sigmoid CE |
| Loss | `ldam` | Label-Distribution-Aware Margin (Cao et al., NeurIPS 2019) |
| Loss | `grw` | Generalized Reweight Loss |
| Sampler | `weighted` | Inverse-frequency weighted random sampling |
| Sampler | `class_aware` | Uniform class sampling with per-class cycling |
| Aug | `mixup_data` | Mixup input interpolation (Zhang et al., ICLR 2018) |

### Training demo

```bash
python examples/train.py --config examples/configs/isic_8class.yaml

# Override config from command line
python examples/train.py --config examples/configs/isic_8class.yaml \
    --loss.name ldam --sampler.name weighted --training.epochs 100
```

## Dataset

Download ISIC 2019 images and place them in a directory. The NumPy split files (`dic.npy`, `train.npy`, `val.npy`, `test.npy`) define which images belong to each split and their labels.

```
dataset/
  8-class/
    dic.npy        # {filename: label_index}
    train.npy      # array of filenames for training
    val.npy
    test.npy
  14-class/
    ...
```

## Project Structure

```
FlexSampling/
  flexsampling/
    losses/          # FocalLoss, CBLoss, LDAMLoss, GRWLoss
    samplers/        # WeightedResampler, ClassAwareSampler
    augmentations/   # mixup
    data/            # ISICDataset loader
  examples/
    train.py         # Full training script
    configs/         # YAML configs
```

## Citation

```bibtex
@inproceedings{ju2022flexible,
  title={Flexible Sampling for Long-tailed Skin Lesion Classification},
  author={Ju, Lie and Wu, Yicheng and Wang, Lin and Yu, Zhen and Zhao, Xin and Wang, Xin and Bonnington, Paul and Ge, Zongyuan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={462--471},
  year={2022},
  organization={Springer}
}
```

## License

MIT
