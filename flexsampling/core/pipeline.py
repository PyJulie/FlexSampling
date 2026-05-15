"""FlexSampling Training Pipeline (Full Algorithm).

Implements the complete FlexSampling training procedure:
  Phase 1: Self-supervised pre-training for balanced features.
  Phase 2: Anchor point selection using class prototypes.
  Phase 3: Warm-up training on anchor set.
  Phase 4: Curriculum sampling — dynamically expand training set.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable, Dict, List, Optional, Set, Tuple

from flexsampling.core.anchor import AnchorSelector
from flexsampling.core.bald import BALDUncertainty
from flexsampling.core.curriculum import CurriculumSampler


class FlexSamplingTrainer:
    """Complete FlexSampling training pipeline.

    Args:
        encoder: Feature extractor (backbone). Should output (N, D) features.
        classifier: Full model (backbone + head) for classification.
        train_dataset: Full training dataset.
        val_dataset: Validation dataset.
        num_classes: Number of classes.
        device: Torch device.
        anchor_scaling: Scaling factor for anchor selection (default 0.1).
        query_ratio: Fraction of pool to query each curriculum step (default 0.1).
        warmup_epochs: Epochs to train on anchor set before curriculum (default 30).
        total_epochs: Total training epochs (default 100).
        patience: Epochs without val improvement before triggering query (default 10).
        bald_samples: Number of MC forward passes for BALD (default 10).
        batch_size: Training batch size (default 64).
        lr: Learning rate (default 3e-4).
        num_workers: DataLoader workers (default 4).
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_classes: int,
        device: torch.device,
        anchor_scaling: float = 0.1,
        query_ratio: float = 0.1,
        warmup_epochs: int = 30,
        total_epochs: int = 100,
        patience: int = 10,
        bald_samples: int = 10,
        batch_size: int = 64,
        lr: float = 3e-4,
        num_workers: int = 4,
    ):
        self.encoder = encoder
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.device = device
        self.anchor_scaling = anchor_scaling
        self.query_ratio = query_ratio
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.patience = patience
        self.bald_samples = bald_samples
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers

        self.anchor_selector = AnchorSelector(scaling=anchor_scaling)
        self.curriculum = CurriculumSampler(num_classes, query_ratio=query_ratio)
        self.bald = BALDUncertainty(n_samples=bald_samples)

        self._all_labels = self._get_all_labels(train_dataset)

    @staticmethod
    def _get_all_labels(dataset: Dataset) -> List[int]:
        if hasattr(dataset, "labels"):
            return list(dataset.labels)
        labels = []
        for i in range(len(dataset)):
            _, y = dataset[i]
            labels.append(int(y) if isinstance(y, (int, np.integer)) else y)
        return labels

    def _make_loader(self, indices: List[int], shuffle: bool = True) -> DataLoader:
        subset = Subset(self.train_dataset, indices)
        return DataLoader(
            subset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def _train_one_epoch(
        self, model: nn.Module, loader: DataLoader,
        optimizer: torch.optim.Optimizer, criterion: nn.Module,
    ) -> float:
        model.train()
        total_loss, n = 0.0, 0
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
        return total_loss / max(1, n)

    @torch.no_grad()
    def _evaluate(self, model: nn.Module) -> Tuple[float, Dict[int, float]]:
        """Evaluate on validation set. Returns (overall_acc, per_class_acc)."""
        model.eval()
        loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
        correct_per_class: Dict[int, int] = {c: 0 for c in range(self.num_classes)}
        total_per_class: Dict[int, int] = {c: 0 for c in range(self.num_classes)}

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            for pred, target in zip(preds, targets):
                c = int(target)
                total_per_class[c] += 1
                if int(pred) == c:
                    correct_per_class[c] += 1

        per_class_acc = {}
        for c in range(self.num_classes):
            if total_per_class[c] > 0:
                per_class_acc[c] = correct_per_class[c] / total_per_class[c]
            else:
                per_class_acc[c] = 0.0

        overall = sum(correct_per_class.values()) / max(1, sum(total_per_class.values()))
        return overall, per_class_acc

    def run(
        self,
        criterion: Optional[nn.Module] = None,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """Execute the full FlexSampling pipeline.

        Args:
            criterion: Loss function. Defaults to CrossEntropyLoss.
            callback: Optional callback(epoch, metrics_dict) called each epoch.

        Returns:
            Dictionary with training history and final metrics.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # --- Phase 1 & 2: Anchor selection ---
        print("[FlexSampling] Phase 1-2: Selecting anchor points...")
        anchor_indices, prototypes = self.anchor_selector.select_from_dataset(
            self.encoder, self.train_dataset, self.device,
            self.batch_size, self.num_workers,
        )
        active_indices = set(anchor_indices)
        print(f"  Anchor set: {len(anchor_indices)} samples "
              f"(from {len(self.train_dataset)} total)")

        # --- Phase 3: Warm-up on anchors ---
        print(f"[FlexSampling] Phase 3: Warm-up training ({self.warmup_epochs} epochs)...")
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        best_val_acc = 0.0
        epochs_without_improvement = 0
        history = {"epoch": [], "loss": [], "val_acc": [], "active_size": [], "per_class_acc": []}

        for epoch in range(self.total_epochs):
            loader = self._make_loader(sorted(active_indices))
            loss = self._train_one_epoch(self.classifier, loader, optimizer, criterion)
            val_acc, per_class_acc = self._evaluate(self.classifier)

            history["epoch"].append(epoch)
            history["loss"].append(loss)
            history["val_acc"].append(val_acc)
            history["active_size"].append(len(active_indices))
            history["per_class_acc"].append(per_class_acc)

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            phase = "warmup" if epoch < self.warmup_epochs else "curriculum"
            print(f"  [{phase}] Epoch {epoch+1}/{self.total_epochs}  "
                  f"loss={loss:.4f}  val_acc={val_acc:.4f}  "
                  f"active={len(active_indices)}  best={best_val_acc:.4f}")

            if callback:
                callback(epoch, {
                    "loss": loss, "val_acc": val_acc,
                    "per_class_acc": per_class_acc,
                    "active_size": len(active_indices),
                })

            # --- Phase 4: Curriculum sampling (after warm-up) ---
            if (epoch >= self.warmup_epochs
                    and epochs_without_improvement >= self.patience
                    and len(active_indices) < len(self.train_dataset)):
                print(f"  [curriculum] Querying new samples (patience={self.patience} reached)...")

                # Compute BALD uncertainty on full dataset
                scores = self.bald.score(
                    self.classifier, self.train_dataset,
                    self.device, self.batch_size, self.num_workers,
                )

                # Query new samples
                new_indices = self.curriculum.query(
                    self._all_labels, active_indices, scores, per_class_acc,
                )
                if new_indices:
                    active_indices.update(new_indices)
                    print(f"  [curriculum] Added {len(new_indices)} samples "
                          f"(total active: {len(active_indices)})")

                    # Reset optimizer with fresh learning rate
                    optimizer = torch.optim.Adam(
                        self.classifier.parameters(), lr=self.lr,
                    )
                    epochs_without_improvement = 0
                else:
                    print("  [curriculum] No more samples to query.")

        final_acc, final_per_class = self._evaluate(self.classifier)
        history["final_acc"] = final_acc
        history["final_per_class_acc"] = final_per_class
        history["best_val_acc"] = best_val_acc

        print(f"\n[FlexSampling] Done. Best val acc: {best_val_acc:.4f}, "
              f"Final: {final_acc:.4f}")
        return history
