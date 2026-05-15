"""Generalized Reweight Loss.

Assigns per-class weights proportional to 1 / (ratio ^ exp_scale),
where ratio = n_i / max(n).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


class GRWLoss(nn.Module):
    """Generalized Reweight cross-entropy loss.

    Args:
        cls_num_list: Number of samples per class.
        exp_scale: Exponent controlling reweight strength. Default 1.2.
    """

    def __init__(self, cls_num_list: List[int], exp_scale: float = 1.2):
        super().__init__()
        counts = np.array(cls_num_list, dtype=np.float64)
        ratios = counts / counts.max()
        weights = 1.0 / np.power(ratios, exp_scale)
        weights = weights / weights.sum() * len(cls_num_list)
        self.register_buffer("weight", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weight)
