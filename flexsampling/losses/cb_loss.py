"""Class-Balanced Loss (Cui et al., CVPR 2019).

Reweights loss by effective number of samples: (1-beta)/(1-beta^n_i).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class CBLoss(nn.Module):
    """Class-Balanced Loss wrapping focal, sigmoid CE, or softmax CE.

    Args:
        cls_num_list: Number of samples per class.
        beta: Effective number hyperparameter in (0, 1). Default 0.9999.
        gamma: Focal loss gamma (only used when loss_type='focal').
        loss_type: 'focal', 'sigmoid', or 'softmax'.
    """

    def __init__(
        self,
        cls_num_list: List[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        loss_type: str = "focal",
    ):
        super().__init__()
        assert loss_type in ("focal", "sigmoid", "softmax")
        self.loss_type = loss_type
        self.gamma = gamma

        effective_num = 1.0 - np.power(beta, cls_num_list)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(cls_num_list)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw scores.
            targets: (N,) class indices.
        """
        per_cls_weights = self.weights[targets]

        if self.loss_type == "focal":
            log_p = F.log_softmax(logits, dim=1)
            ce = F.nll_loss(log_p, targets, reduction="none")
            p_t = torch.exp(-ce)
            focal = (1.0 - p_t) ** self.gamma * ce
            loss = per_cls_weights * focal
        elif self.loss_type == "sigmoid":
            # one-hot encoding for BCE
            one_hot = F.one_hot(targets, logits.size(1)).float()
            bce = F.binary_cross_entropy_with_logits(
                logits, one_hot, reduction="none"
            )
            loss = (per_cls_weights.unsqueeze(1) * bce).sum(dim=1)
        else:  # softmax
            ce = F.cross_entropy(logits, targets, reduction="none")
            loss = per_cls_weights * ce

        return loss.mean()
