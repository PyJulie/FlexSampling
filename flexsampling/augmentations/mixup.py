"""Mixup data augmentation (Zhang et al., ICLR 2018).

Creates virtual training examples by linearly interpolating pairs of inputs
and their labels.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mix inputs and return mixed data with both label sets.

    Args:
        x: Input batch (N, ...).
        y: Target labels (N,).
        alpha: Beta distribution parameter. 0 disables mixup.

    Returns:
        (mixed_x, y_a, y_b, lam) where mixed_x = lam*x + (1-lam)*x[perm].
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    indices = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[indices]
    return mixed_x, y, y[indices], lam


def mixup_criterion(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixup loss as weighted sum of losses on both label sets."""
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
