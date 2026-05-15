"""Focal Loss for addressing class imbalance (Lin et al., ICCV 2017)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t).

    Args:
        gamma: Focusing parameter. Higher gamma down-weights easy examples more.
        alpha: Per-class weights. Tensor of shape (C,), or None for uniform.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw scores.
            targets: (N,) class indices.
        """
        log_p = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(log_p, targets, reduction="none")
        p_t = torch.exp(-ce)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
