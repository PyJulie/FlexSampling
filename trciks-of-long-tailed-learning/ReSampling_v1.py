import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

train_image_datasets = '' #define train_image_datasets here
target = train_image_datasets.img_label
class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in target])
samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))