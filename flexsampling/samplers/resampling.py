"""Weighted random resampling based on inverse class frequency."""
from collections import Counter
from typing import List

from torch.utils.data import WeightedRandomSampler


class WeightedResampler(WeightedRandomSampler):
    """Oversamples minority classes via inverse-frequency weights.

    Each sample's draw probability is proportional to 1/n_c where n_c is the
    count of its class. Effectively balances the training distribution.

    Args:
        labels: Per-sample class labels for the entire training set.
        num_samples: Total samples per epoch. Defaults to len(labels).
        replacement: Sample with replacement. Default True.
    """

    def __init__(
        self,
        labels: List[int],
        num_samples: int = 0,
        replacement: bool = True,
    ):
        counter = Counter(labels)
        weights = [1.0 / counter[l] for l in labels]
        if num_samples <= 0:
            num_samples = len(labels)
        super().__init__(weights, num_samples=num_samples, replacement=replacement)
