"""Anchor Point Selection (Section 2.2).

Selects representative samples closest to each class prototype in feature
space to form a less-imbalanced initial training subset.
"""
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset, Subset


class AnchorSelector:
    """Selects anchor points by distance to class prototypes.

    Given a feature extractor (e.g. SSL-pretrained encoder), computes class
    prototypes (mean feature vectors) and selects the nearest samples per class
    to form an anchor set with a controlled imbalance ratio.

    Args:
        scaling: Imbalance scaling factor s in (0, 1). The anchor set has
            imbalance ratio r_hat = r * s. Smaller s -> more balanced.
        min_per_class: Minimum anchor samples per class.
    """

    def __init__(self, scaling: float = 0.1, min_per_class: int = 2):
        self.scaling = scaling
        self.min_per_class = min_per_class

    @torch.no_grad()
    def extract_features(
        self,
        encoder: nn.Module,
        dataset: Dataset,
        device: torch.device,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for all samples.

        Returns:
            (features, labels) as numpy arrays.
        """
        encoder.eval()
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        all_feats, all_labels = [], []
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            feats = encoder(images)
            if feats.dim() > 2:
                feats = feats.mean(dim=list(range(1, feats.dim() - 1)))
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
        return np.concatenate(all_feats), np.concatenate(all_labels)

    def compute_prototypes(
        self, features: np.ndarray, labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Compute mean feature vector (prototype) per class."""
        prototypes = {}
        for c in np.unique(labels):
            mask = labels == c
            prototypes[int(c)] = features[mask].mean(axis=0)
        return prototypes

    def select(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        prototypes: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[int]:
        """Select anchor point indices.

        For each class, computes distance to prototype and selects the closest
        samples. The number of anchors per class is determined by the scaling
        factor to produce a less-imbalanced subset.

        Args:
            features: (N, D) feature array.
            labels: (N,) label array.
            prototypes: Pre-computed prototypes. If None, computed from data.

        Returns:
            List of selected sample indices.
        """
        if prototypes is None:
            prototypes = self.compute_prototypes(features, labels)

        # Compute per-class counts
        classes = sorted(prototypes.keys())
        cls_counts = {c: int((labels == c).sum()) for c in classes}
        max_count = max(cls_counts.values())

        # Determine anchor counts with scaled imbalance ratio.
        # Original ratio: r = max_count / min_count.
        # Target ratio: r_hat = r * s.
        # For the head class, select N_0 = min_count * r_hat samples.
        # For other classes, interpolate so the anchor distribution has
        # imbalance ratio r_hat instead of r.
        min_count = min(cls_counts.values())
        original_ratio = max_count / max(1, min_count)
        target_ratio = original_ratio * self.scaling

        # Head class gets N_0, tail class keeps its count, others interpolate
        anchor_counts = {}
        for c in classes:
            # Proportion of this class relative to tail in original data
            proportion = cls_counts[c] / max(1, min_count)
            # Compress this proportion under the target ratio
            compressed = min(proportion, target_ratio)
            n_anchor = max(
                self.min_per_class,
                int(min_count * compressed),
            )
            n_anchor = min(n_anchor, cls_counts[c])
            anchor_counts[c] = n_anchor

        # Select closest samples to prototype per class
        selected = []
        for c in classes:
            mask = labels == c
            indices = np.where(mask)[0]
            feats_c = features[indices]
            proto = prototypes[c]
            distances = np.linalg.norm(feats_c - proto, axis=1)
            order = np.argsort(distances)
            n_select = anchor_counts[c]
            selected.extend(indices[order[:n_select]].tolist())

        return sorted(selected)

    def select_from_dataset(
        self,
        encoder: nn.Module,
        dataset: Dataset,
        device: torch.device,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> Tuple[List[int], Dict[int, np.ndarray]]:
        """End-to-end: extract features, compute prototypes, select anchors.

        Returns:
            (anchor_indices, prototypes)
        """
        features, labels = self.extract_features(
            encoder, dataset, device, batch_size, num_workers,
        )
        prototypes = self.compute_prototypes(features, labels)
        indices = self.select(features, labels, prototypes)
        return indices, prototypes
