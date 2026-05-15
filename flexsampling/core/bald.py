"""BALD Uncertainty Estimation (Section 2.3.2).

Computes Bayesian Active Learning by Disagreement (BALD) scores using
MC Dropout to estimate mutual information between predictions and model
parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Optional


def _enable_dropout(model: nn.Module):
    """Enable dropout layers during inference for MC sampling."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


class BALDUncertainty:
    """BALD-based uncertainty scorer using MC Dropout.

    Computes: I(y; omega | x) = H(y|x) - E[H(y|x, omega)]
    where H is entropy and the expectation is over MC dropout samples.

    Args:
        n_samples: Number of MC forward passes.
        dropout_rate: If provided, temporarily sets all Dropout layers to this
            rate. If None, uses existing dropout rates.
    """

    def __init__(self, n_samples: int = 10, dropout_rate: Optional[float] = None):
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    @torch.no_grad()
    def score(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: torch.device,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> np.ndarray:
        """Compute BALD uncertainty scores for all samples.

        Args:
            model: Classifier with dropout layers.
            dataset: Dataset to score.
            device: Torch device.

        Returns:
            (N,) array of uncertainty scores (higher = more uncertain).
        """
        model.eval()
        _enable_dropout(model)

        # Optionally override dropout rate
        original_rates = {}
        if self.dropout_rate is not None:
            for name, m in model.named_modules():
                if isinstance(m, nn.Dropout):
                    original_rates[name] = m.p
                    m.p = self.dropout_rate

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        all_scores = []
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            # Collect MC samples: (T, N, C)
            mc_probs = []
            for _ in range(self.n_samples):
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                mc_probs.append(probs.cpu())
            mc_probs = torch.stack(mc_probs, dim=0)  # (T, N, C)

            # H(y|x): entropy of mean prediction
            mean_probs = mc_probs.mean(dim=0)  # (N, C)
            h_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)

            # E[H(y|x, omega)]: mean entropy across MC samples
            h_per_sample = -(mc_probs * torch.log(mc_probs + 1e-10)).sum(dim=2)  # (T, N)
            mean_h = h_per_sample.mean(dim=0)  # (N,)

            # BALD = H(y|x) - E[H(y|x, omega)]
            bald = h_mean - mean_h
            all_scores.append(bald.numpy())

        # Restore dropout rates
        if self.dropout_rate is not None:
            for name, m in model.named_modules():
                if isinstance(m, nn.Dropout) and name in original_rates:
                    m.p = original_rates[name]

        model.eval()
        return np.concatenate(all_scores)
