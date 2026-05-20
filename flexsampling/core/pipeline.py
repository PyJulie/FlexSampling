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
from collections import Counter
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable, Dict, List, Optional, Set, Tuple

from flexsampling.core.anchor import AnchorSelector
from flexsampling.core.bald import BALDUncertainty
from flexsampling.core.curriculum import CurriculumSampler
from flexsampling.core.ssl_pretrain import SSLPretrainer
from flexsampling.samplers.class_aware import ClassAwareSampler


class FlexSamplingTrainer:
    """Complete FlexSampling training pipeline.

    Args:
        encoder: Feature extractor (backbone). Should output (N, D) features.
        classifier: Full model (backbone + head) for classification.
        train_dataset: Full training dataset.
        val_dataset: Validation dataset.
        num_classes: Number of classes.
        device: Torch device.
        embed_dim: Encoder output feature dimension.
        ssl_epochs: Epochs for SSL pre-training (0 to skip).
        ssl_batch_size: Batch size for SSL pre-training.
        ssl_accum_steps: Gradient accumulation steps for SSL.
        ssl_lr: Learning rate for SSL pre-training.
        ssl_temperature: NT-Xent temperature.
        img_size: Image size for SSL augmentations.
        anchor_scaling: Scaling factor for anchor selection (default 0.1).
        query_ratio: Fraction of pool to query each curriculum step (default 0.1).
        query_interval: Force a curriculum query every N epochs after warmup,
            in addition to plateau-triggered queries. 0 to disable (default 5).
        warmup_epochs: Epochs to train on anchor set before curriculum (default 20).
        total_epochs: Total training epochs (default 120).
        patience: Epochs without val improvement before triggering query (default 7).
        bald_samples: Number of MC forward passes for BALD (default 10).
        batch_size: Training batch size (default 32).
        lr: Initial learning rate (default 0.01).
        momentum: SGD momentum (default 0.9).
        weight_decay: Weight decay (default 1e-4).
        use_class_aware: Use ClassAwareSampler within active set (default True).
        mixup_alpha: Mixup alpha, 0 to disable (default 0.0).
        num_workers: DataLoader workers (default 0).
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_classes: int,
        device: torch.device,
        embed_dim: int = 2048,
        ssl_epochs: int = 100,
        ssl_batch_size: int = 64,
        ssl_accum_steps: int = 4,
        ssl_lr: float = 0.001,
        ssl_temperature: float = 0.5,
        img_size: int = 224,
        anchor_scaling: float = 0.1,
        query_ratio: float = 0.1,
        query_interval: int = 5,
        warmup_epochs: int = 20,
        total_epochs: int = 120,
        patience: int = 7,
        bald_samples: int = 10,
        batch_size: int = 32,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        use_class_aware: bool = True,
        mixup_alpha: float = 0.0,
        num_workers: int = 0,
    ):
        self.encoder = encoder
        self.classifier = classifier
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.device = device
        self.embed_dim = int(embed_dim)
        self.ssl_epochs = int(ssl_epochs)
        self.ssl_batch_size = int(ssl_batch_size)
        self.ssl_accum_steps = int(ssl_accum_steps)
        self.ssl_lr = float(ssl_lr)
        self.ssl_temperature = float(ssl_temperature)
        self.img_size = int(img_size)
        self.anchor_scaling = float(anchor_scaling)
        self.query_ratio = float(query_ratio)
        self.query_interval = int(query_interval)
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.use_class_aware = bool(use_class_aware)
        self.mixup_alpha = float(mixup_alpha)
        self.num_workers = int(num_workers)

        self.anchor_selector = AnchorSelector(scaling=anchor_scaling)
        self.curriculum = CurriculumSampler(num_classes, query_ratio=query_ratio)
        self.bald = BALDUncertainty(n_samples=int(bald_samples))

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

    def _make_loader(self, indices: List[int]) -> DataLoader:
        """Build DataLoader with optional class-aware sampling."""
        subset = Subset(self.train_dataset, indices)
        if self.use_class_aware and len(indices) > self.batch_size:
            subset_labels = [self._all_labels[i] for i in indices]
            sampler = ClassAwareSampler(
                labels=subset_labels, num_samples=len(indices),
            )
            return DataLoader(
                subset, batch_size=self.batch_size, sampler=sampler,
                num_workers=self.num_workers, pin_memory=True, drop_last=True,
            )
        return DataLoader(
            subset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def _train_one_epoch(
        self, model: nn.Module, loader: DataLoader,
        optimizer: torch.optim.Optimizer, criterion: nn.Module,
        scheduler=None,
    ) -> float:
        model.train()
        total_loss, n = 0.0, 0
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.mixup_alpha > 0:
                from flexsampling.augmentations.mixup import mixup_data, mixup_criterion
                images, ya, yb, lam = mixup_data(images, targets, self.mixup_alpha)
                logits = model(images)
                loss = mixup_criterion(criterion, logits, ya, yb, lam)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * images.size(0)
            n += images.size(0)
        return total_loss / max(1, n)

    @torch.no_grad()
    def _evaluate(self, model: nn.Module) -> Tuple[float, Dict[int, float]]:
        """Evaluate on validation set. Returns (balanced_acc, per_class_acc)."""
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
        accs = []
        for c in range(self.num_classes):
            if total_per_class[c] > 0:
                a = correct_per_class[c] / total_per_class[c]
            else:
                a = 0.0
            per_class_acc[c] = a
            accs.append(a)

        balanced_acc = float(np.mean(accs))
        return balanced_acc, per_class_acc

    def _build_criterion(self, active_indices: Set[int]) -> nn.Module:
        """Build class-balanced loss based on current active set distribution."""
        from flexsampling.losses.cb_loss import CBLoss
        active_labels = [self._all_labels[i] for i in active_indices]
        counter = Counter(active_labels)
        cls_num_list = [counter.get(c, 1) for c in range(self.num_classes)]
        return CBLoss(cls_num_list=cls_num_list, beta=0.9999, gamma=1.0,
                      loss_type="focal").to(self.device)

    def _do_query(
        self, active_indices: Set[int], per_class_acc: Dict[int, float],
        use_adaptive_loss: bool, criterion: nn.Module,
    ) -> Tuple[Set[int], nn.Module, int]:
        """Run one curriculum query round. Returns (updated_active, criterion, n_added)."""
        if len(active_indices) >= len(self.train_dataset):
            return active_indices, criterion, 0

        scores = self.bald.score(
            self.classifier, self.train_dataset,
            self.device, self.batch_size, self.num_workers,
        )
        new_indices = self.curriculum.query(
            self._all_labels, active_indices, scores, per_class_acc,
        )
        if new_indices:
            active_indices.update(new_indices)
            if use_adaptive_loss:
                criterion = self._build_criterion(active_indices)
            return active_indices, criterion, len(new_indices)
        return active_indices, criterion, 0

    def run(
        self,
        criterion: Optional[nn.Module] = None,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """Execute the full FlexSampling pipeline.

        Args:
            criterion: Loss function. If None, uses adaptive CB Focal loss
                that recomputes weights when active set changes.
            callback: Optional callback(epoch, metrics_dict) called each epoch.

        Returns:
            Dictionary with training history and final metrics.
        """
        use_adaptive_loss = criterion is None

        # --- Phase 1: Self-supervised pre-training ---
        if self.ssl_epochs > 0:
            print(f"[FlexSampling] Phase 1: SSL contrastive pre-training ({self.ssl_epochs} epochs)...")
            ssl_trainer = SSLPretrainer(
                encoder=self.encoder,
                embed_dim=self.embed_dim,
                dataset=self.train_dataset,
                device=self.device,
                img_size=self.img_size,
                epochs=self.ssl_epochs,
                batch_size=self.ssl_batch_size,
                accum_steps=self.ssl_accum_steps,
                lr=self.ssl_lr,
                temperature=self.ssl_temperature,
                num_workers=self.num_workers,
            )
            self.encoder = ssl_trainer.train()
        else:
            print("[FlexSampling] Phase 1: Skipping SSL (ssl_epochs=0), using pre-trained weights.")

        # --- Phase 2: Anchor selection ---
        print("[FlexSampling] Phase 2: Selecting anchor points...")
        anchor_indices, prototypes = self.anchor_selector.select_from_dataset(
            self.encoder, self.train_dataset, self.device,
            self.batch_size, self.num_workers,
        )
        active_indices = set(anchor_indices)
        print(f"  Anchor set: {len(anchor_indices)} samples "
              f"(from {len(self.train_dataset)} total)")

        # --- Phase 2.5: Transfer SSL encoder weights to classifier ---
        if self.ssl_epochs > 0:
            enc_state = self.encoder.state_dict()
            clf_state = self.classifier.state_dict()
            transferred = 0
            for key in enc_state:
                if key in clf_state and enc_state[key].shape == clf_state[key].shape:
                    clf_state[key] = enc_state[key]
                    transferred += 1
            self.classifier.load_state_dict(clf_state)
            print(f"  Transferred {transferred} parameter tensors from SSL encoder to classifier.")

        # --- Phase 3 & 4: Training with curriculum ---
        print(f"[FlexSampling] Phase 3: Warm-up training ({self.warmup_epochs} epochs)...")

        # Build initial loss
        if use_adaptive_loss:
            criterion = self._build_criterion(active_indices)

        # SGD + step-level cosine annealing (matches baseline schedule)
        # T_max based on full dataset so early small-set epochs burn fewer steps.
        optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
        )
        full_steps_per_epoch = max(1, len(self.train_dataset) // self.batch_size)
        total_steps = self.total_epochs * full_steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps,
        )
        print(f"  LR schedule: step-level cosine, T_max={total_steps} steps")

        best_val_acc = 0.0
        best_state = None
        epochs_without_improvement = 0
        epochs_since_last_query = 0
        history = {"epoch": [], "loss": [], "val_acc": [], "active_size": [], "per_class_acc": []}

        for epoch in range(self.total_epochs):
            loader = self._make_loader(sorted(active_indices))
            loss = self._train_one_epoch(
                self.classifier, loader, optimizer, criterion, scheduler,
            )
            val_acc, per_class_acc = self._evaluate(self.classifier)

            history["epoch"].append(epoch)
            history["loss"].append(loss)
            history["val_acc"].append(val_acc)
            history["active_size"].append(len(active_indices))
            history["per_class_acc"].append(per_class_acc)

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.classifier.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            phase = "warmup" if epoch < self.warmup_epochs else "curriculum"
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [{phase}] Epoch {epoch+1}/{self.total_epochs}  "
                  f"loss={loss:.4f}  val_acc={val_acc:.4f}  "
                  f"active={len(active_indices)}  best={best_val_acc:.4f}  "
                  f"lr={lr_now:.6f}")

            if callback:
                callback(epoch, {
                    "loss": loss, "val_acc": val_acc,
                    "per_class_acc": per_class_acc,
                    "active_size": len(active_indices),
                })

            # --- Phase 4: Curriculum sampling (after warm-up) ---
            if epoch >= self.warmup_epochs:
                epochs_since_last_query += 1

                # Trigger query on plateau OR on fixed interval
                plateau_trigger = epochs_without_improvement >= self.patience
                interval_trigger = (
                    self.query_interval > 0
                    and epochs_since_last_query >= self.query_interval
                )

                if (plateau_trigger or interval_trigger) \
                        and len(active_indices) < len(self.train_dataset):
                    trigger = "plateau" if plateau_trigger else "scheduled"
                    print(f"  [curriculum] Querying new samples ({trigger})...")

                    active_indices, criterion, n_added = self._do_query(
                        active_indices, per_class_acc, use_adaptive_loss, criterion,
                    )
                    if n_added > 0:
                        print(f"  [curriculum] Added {n_added} samples "
                              f"(total active: {len(active_indices)}, "
                              f"{100*len(active_indices)/len(self.train_dataset):.0f}%)")
                        epochs_since_last_query = 0
                        if plateau_trigger:
                            epochs_without_improvement = 0
                    else:
                        print("  [curriculum] No more samples to query.")

        # Restore best model
        if best_state is not None:
            self.classifier.load_state_dict(best_state)

        final_acc, final_per_class = self._evaluate(self.classifier)
        history["final_acc"] = final_acc
        history["final_per_class_acc"] = final_per_class
        history["best_val_acc"] = best_val_acc

        print(f"\n[FlexSampling] Done. Best val acc: {best_val_acc:.4f}, "
              f"Final (restored best): {final_acc:.4f}")
        return history
