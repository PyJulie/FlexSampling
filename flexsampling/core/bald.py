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
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def _inject_dropout(model: nn.Module, p: float = 0.1) -> list:
    """Inject Dropout layers after BatchNorm+ReLU in models that lack them.

    This is needed for MC Dropout in architectures like ResNet that don't
    have built-in dropout. Adds dropout after the last ReLU in each
    residual block.

    Returns list of (parent, attr_name, original_module) for cleanup.
    """
    injected = []
    for name, module in model.named_modules():
        # Look for Sequential blocks (residual block internals)
        if isinstance(module, nn.Sequential) and len(list(module.children())) >= 2:
            children = list(module.named_children())
            # Check if last child is ReLU or activation
            last_name, last_child = children[-1]
            if isinstance(last_child, (nn.ReLU, nn.GELU, nn.SiLU)):
                # Insert dropout after activation
                new_seq = nn.Sequential(last_child, nn.Dropout2d(p=p))
                setattr(module, last_name, new_seq)
                injected.append((module, last_name, last_child))
    return injected


def _remove_injected_dropout(injected: list):
    """Remove previously injected dropout layers."""
    for parent, attr_name, original in injected:
        setattr(parent, attr_name, original)


class BALDUncertainty:
    """BALD-based uncertainty scorer using MC Dropout.

    Computes: I(y; omega | x) = H(y|x) - E[H(y|x, omega)]
    where H is entropy and the expectation is over MC dropout samples.

    If the model has no built-in Dropout layers, temporary Dropout2d layers
    are injected to enable stochastic forward passes.

    Args:
        n_samples: Number of MC forward passes.
        dropout_rate: Dropout probability for MC sampling (default 0.1).
    """

    def __init__(self, n_samples: int = 10, dropout_rate: float = 0.1):
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def _has_dropout(self, model: nn.Module) -> bool:
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                return True
        return False

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
            model: Classifier (dropout layers injected automatically if missing).
            dataset: Dataset to score.
            device: Torch device.

        Returns:
            (N,) array of uncertainty scores (higher = more uncertain).
        """
        model.eval()

        # Inject dropout if model doesn't have any
        injected = []
        if not self._has_dropout(model):
            injected = _inject_dropout(model, p=self.dropout_rate)

        # Enable dropout layers for MC sampling
        _enable_dropout(model)

        # Override dropout rate on pre-existing layers if needed
        original_rates = {}
        if not injected:
            for name, m in model.named_modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d)):
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

        # Cleanup: restore original dropout rates and remove injected layers
        if original_rates:
            for name, m in model.named_modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d)) and name in original_rates:
                    m.p = original_rates[name]
        if injected:
            _remove_injected_dropout(injected)

        model.eval()
        return np.concatenate(all_scores)
