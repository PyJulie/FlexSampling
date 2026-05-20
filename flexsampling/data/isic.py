"""ISIC skin lesion dataset loader.

Supports the NumPy-format splits shipped with the original FlexSampling repo
(8-class and 14-class variants) as well as image-folder layouts.
"""
import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def default_val_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ISICDataset(Dataset):
    """ISIC dataset from NumPy arrays.

    Expected file layout (as in the original repo)::

        dataset_root/
            dic.npy              # {filename: label_index}
            train.npy            # array of filenames (14-class)
            train_100.npy        # or with sample-size suffix (8-class)
            val.npy / val_100.npy
            test.npy / test_100.npy

    Args:
        root: Path to the split directory containing .npy files.
        image_dir: Path or list of paths to directories containing image files.
        split: Split name, used as the .npy filename stem (e.g. 'train', 'train_100').
        transform: Torchvision transform. If None, uses default.
        img_size: Image size for default transforms.
    """

    def __init__(
        self,
        root: str,
        image_dir,
        split: str = "train",
        transform: Optional[Callable] = None,
        img_size: int = 224,
    ):
        if isinstance(image_dir, (list, tuple)):
            self.image_dirs = list(image_dir)
        else:
            self.image_dirs = [image_dir]
        label_dict = np.load(os.path.join(root, "dic.npy"), allow_pickle=True).item()
        filenames = np.load(os.path.join(root, f"{split}.npy"), allow_pickle=True)

        self.samples: List[Tuple[str, int]] = []
        for fname in filenames:
            fname = str(fname)
            if fname in label_dict:
                self.samples.append((fname, int(label_dict[fname])))

        if transform is not None:
            self.transform = transform
        elif split.startswith("train"):
            self.transform = default_train_transform(img_size)
        else:
            self.transform = default_val_transform(img_size)

    @property
    def labels(self) -> List[int]:
        return [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _find_image(self, fname: str) -> str:
        for d in self.image_dirs:
            for ext in ("", ".jpg", ".jpeg", ".png"):
                path = os.path.join(d, fname + ext)
                if os.path.exists(path):
                    return path
        return os.path.join(self.image_dirs[0], fname)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        fname, label = self.samples[index]
        path = self._find_image(fname)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def get_cls_num_list(labels: List[int]) -> List[int]:
    """Count samples per class, ordered by class index."""
    from collections import Counter
    counter = Counter(labels)
    num_classes = max(counter.keys()) + 1
    return [counter.get(i, 0) for i in range(num_classes)]
