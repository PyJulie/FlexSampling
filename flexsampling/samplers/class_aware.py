"""Class-Aware Sampler: balanced sampling across classes.

Each batch samples classes uniformly, then picks random instances within each class.
Adapted from OLTR (Liu et al., CVPR 2019).
"""
import itertools
import random
from collections import defaultdict
from typing import Iterator, List

from torch.utils.data import Sampler


class _CyclicShuffler:
    """Yields indices for a single class in shuffled order, cycling forever."""

    def __init__(self, indices: List[int]):
        self._indices = list(indices)
        self._iter: Iterator[int] = iter([])

    def __next__(self) -> int:
        try:
            return next(self._iter)
        except StopIteration:
            random.shuffle(self._indices)
            self._iter = iter(self._indices)
            return next(self._iter)


class ClassAwareSampler(Sampler[int]):
    """Samples classes uniformly, then picks a random instance per class.

    Args:
        labels: Per-sample class labels for the entire training set.
        num_samples_per_cls: How many instances to draw per class per cycle.
        num_samples: Total samples per epoch. Defaults to len(labels).
    """

    def __init__(
        self,
        labels: List[int],
        num_samples_per_cls: int = 1,
        num_samples: int = 0,
    ):
        self.num_samples = num_samples if num_samples > 0 else len(labels)
        self.num_samples_per_cls = num_samples_per_cls

        cls_to_indices: defaultdict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            cls_to_indices[label].append(idx)

        self._classes = sorted(cls_to_indices.keys())
        self._cyclers = {c: _CyclicShuffler(cls_to_indices[c]) for c in self._classes}

    def __iter__(self) -> Iterator[int]:
        yielded = 0
        while yielded < self.num_samples:
            random.shuffle(self._classes)
            for c in self._classes:
                cycler = self._cyclers[c]
                for _ in range(self.num_samples_per_cls):
                    if yielded >= self.num_samples:
                        return
                    yield next(cycler)
                    yielded += 1

    def __len__(self) -> int:
        return self.num_samples
