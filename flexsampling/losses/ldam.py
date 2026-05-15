"""LDAM Loss (Cao et al., NeurIPS 2019).

Label-Distribution-Aware Margin loss that enforces larger margins for
minority classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class LDAMLoss(nn.Module):
    """LDAM loss: margin = C / n_j^{1/4}, applied to the correct class logit.

    Args:
        cls_num_list: Number of samples per class.
        max_margin: Maximum margin C. Default 0.5.
        weight: Optional per-class loss weights (e.g. from deferred reweighting).
        scale: Logit scale factor s. Default 30.
    """

    def __init__(
        self,
        cls_num_list: List[int],
        max_margin: float = 0.5,
        weight: Optional[torch.Tensor] = None,
        scale: float = 30.0,
    ):
        super().__init__()
        margins = 1.0 / np.sqrt(np.sqrt(np.array(cls_num_list, dtype=np.float64)))
        margins = margins * (max_margin / margins.max())
        self.register_buffer("margins", torch.tensor(margins, dtype=torch.float32))
        self.scale = scale
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw scores.
            targets: (N,) class indices.
        """
        # Subtract margin only from the target class logit
        margin = self.margins[targets]  # (N,)
        one_hot = F.one_hot(targets, logits.size(1)).float()
        adjusted = logits - one_hot * margin.unsqueeze(1) * self.scale

        return F.cross_entropy(adjusted, targets, weight=self.weight)
