"""Curriculum Sampling Module (Section 2.3).

Dynamically queries new samples from the unsampled pool based on:
  1. Class-wise sampling probability derived from validation accuracy.
  2. Instance-wise selection using BALD uncertainty scores.
"""
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Set

from flexsampling.core.bald import BALDUncertainty


class CurriculumSampler:
    """Curriculum-based dynamic sampler for FlexSampling.

    At each query step, computes per-class sampling probabilities from
    validation accuracy (classes performing worse get sampled more), then
    selects the most uncertain instances from the unsampled pool.

    Args:
        num_classes: Total number of classes.
        query_ratio: Fraction of remaining unsampled pool to query each step.
    """

    def __init__(self, num_classes: int, query_ratio: float = 0.1):
        self.num_classes = num_classes
        self.query_ratio = query_ratio

    def compute_class_probabilities(
        self, val_accuracy_per_class: Dict[int, float],
    ) -> Dict[int, float]:
        """Compute normalized class-wise sampling probability.

        p_cj = 1 - accuracy(cj), then normalize so probabilities sum to K.

        Args:
            val_accuracy_per_class: {class_idx: accuracy} from validation.

        Returns:
            {class_idx: normalized_probability}
        """
        raw = {}
        for c in range(self.num_classes):
            acc = val_accuracy_per_class.get(c, 0.0)
            raw[c] = 1.0 - acc

        total = sum(raw.values())
        if total < 1e-8:
            # All classes perfect — uniform probability
            return {c: 1.0 for c in range(self.num_classes)}

        delta = self.num_classes / total
        return {c: raw[c] * delta for c in range(self.num_classes)}

    def query(
        self,
        all_labels: List[int],
        current_indices: Set[int],
        uncertainty_scores: np.ndarray,
        val_accuracy_per_class: Dict[int, float],
    ) -> List[int]:
        """Select new samples to add to the training set.

        Args:
            all_labels: Labels for the full training dataset.
            current_indices: Indices already in the active training set.
            uncertainty_scores: BALD scores for ALL samples (indexed by dataset index).
            val_accuracy_per_class: Current per-class validation accuracy.

        Returns:
            List of newly selected indices to add.
        """
        probs = self.compute_class_probabilities(val_accuracy_per_class)

        # Build per-class pools of unsampled indices
        pool_by_class: Dict[int, List[int]] = {c: [] for c in range(self.num_classes)}
        for idx, label in enumerate(all_labels):
            if idx not in current_indices:
                pool_by_class[label].append(idx)

        newly_selected = []
        for c in range(self.num_classes):
            pool = pool_by_class[c]
            if not pool:
                continue

            # Number to query for this class.
            # probs[c] is normalized so sum(probs) = num_classes, i.e. avg = 1.
            # So per-class query = pool_size * query_ratio * probs[c] / num_classes
            # gives the right total: sum over classes ≈ total_pool * query_ratio.
            n_query = max(1, int(len(pool) * self.query_ratio * probs[c]))
            n_query = min(n_query, len(pool))

            # Sort by uncertainty (descending) and pick top
            pool_scores = [(idx, uncertainty_scores[idx]) for idx in pool]
            pool_scores.sort(key=lambda x: x[1], reverse=True)
            newly_selected.extend([idx for idx, _ in pool_scores[:n_query]])

        return newly_selected
