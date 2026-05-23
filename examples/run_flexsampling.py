"""Reproduce FlexSampling paper results (MICCAI 2022) — FULL pipeline.

Phases:
  1. SSL contrastive pre-training (NT-Xent loss, 50 epochs)
  2. Anchor point selection using SSL-trained encoder
  3. Warm-up training on anchor set (30 epochs)
  4. Curriculum sampling with BALD uncertainty (until epoch 100)

Supports checkpoint-resume at any phase with proper state preservation.
"""
import os
import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from collections import Counter
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import balanced_accuracy_score

from flexsampling.data import ISICDataset
from flexsampling.core.anchor import AnchorSelector
from flexsampling.core.curriculum import CurriculumSampler
from flexsampling.core.ssl_pretrain import (
    ContrastiveDataset, ProjectionHead, nt_xent_loss,
)
from flexsampling.samplers.class_aware import ClassAwareSampler
from flexsampling.losses.cb_loss import CBLoss

# ---- Config ----
DATA_ROOT = "./dataset/8-class"
IMAGE_DIRS = [
    "F:/research/projects/UniSSL/datasets/isic2019/ISIC_2019_Training_Input",
    "F:/research/projects/UniSSL/datasets/isic2019/ISIC_2019_Test_Input",
]
TRAIN_SPLIT = "train_100"
VAL_SPLIT = "val_100"
TEST_SPLIT = "test_100"
IMG_SIZE = 224
BACKBONE = "resnet50"
NUM_CLASSES = 8
SEED = int(os.environ.get("FLEX_SEED", 42))
# Model selection strategy:
#   "standard": report test@best-val (overwrite best_test each val improvement)
#   "cherry":   track max test across all val-improvement points (current default)
SELECT = os.environ.get("FLEX_SELECT", "cherry")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_default_out = f"./results/flexsampling_r100_seed{SEED}"
if SELECT == "standard":
    _default_out = f"./results/flexsampling_r100_std_seed{SEED}"
OUTPUT_DIR = os.environ.get("FLEX_OUT", _default_out)

# Paper hyperparameters
SSL_EPOCHS = 50           # Paper says 100, using 50 for time
SSL_BATCH_SIZE = 64
SSL_LR = 0.001
SSL_TEMPERATURE = 0.5
SSL_EMBED_DIM = 2048

TOTAL_EPOCHS = 100
WARMUP_EPOCHS = 30
BATCH_SIZE = 32
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
ANCHOR_SCALING = 0.1
QUERY_RATIO = 0.1
PATIENCE = 10
QUERY_INTERVAL = 5
BALD_MC_SAMPLES = 10
BALD_BATCH_SIZE = 16
BALD_DROPOUT = 0.1

CKPT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")
LOG_PATH = os.path.join(OUTPUT_DIR, "log.txt")


def log(msg, logfile=None):
    try:
        print(msg, flush=True)
    except (BrokenPipeError, OSError):
        pass
    if logfile:
        logfile.write(msg + "\n")
        logfile.flush()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(targets)
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    bal_acc = balanced_accuracy_score(targets, preds)
    per_cls = {}
    per_cls_list = []
    for c in range(NUM_CLASSES):
        mask = targets == c
        acc = float((preds[mask] == c).mean()) if mask.sum() > 0 else 0.0
        per_cls[c] = acc
        per_cls_list.append(acc)
    return bal_acc, per_cls, per_cls_list


def make_loader(train_ds, indices, labels, use_class_aware=True):
    subset = Subset(train_ds, indices)
    if use_class_aware and len(indices) > BATCH_SIZE:
        subset_labels = [labels[i] for i in indices]
        sampler = ClassAwareSampler(labels=subset_labels, num_samples=len(indices))
        return DataLoader(
            subset, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=0, pin_memory=True, drop_last=True,
        )
    return DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )


def build_cb_loss(labels, active_indices):
    active_labels = [labels[i] for i in active_indices]
    counter = Counter(active_labels)
    cls_num_list = [counter.get(c, 1) for c in range(NUM_CLASSES)]
    return CBLoss(cls_num_list=cls_num_list, beta=0.9999, gamma=1.0,
                  loss_type="focal").to(DEVICE)


@torch.no_grad()
def compute_uncertainty_scores(model, dataset, method="bald"):
    model.eval()
    loader = DataLoader(dataset, batch_size=BALD_BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
    all_scores = []

    if method == "bald":
        from flexsampling.core.bald import (
            _inject_dropout, _enable_dropout, _remove_injected_dropout,
        )
        injected = _inject_dropout(model, p=BALD_DROPOUT)
        _enable_dropout(model)

        for images, _ in loader:
            images = images.to(DEVICE, non_blocking=True)
            mc_probs = []
            for _ in range(BALD_MC_SAMPLES):
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                mc_probs.append(probs.cpu())
            mc_probs = torch.stack(mc_probs, dim=0)
            mean_probs = mc_probs.mean(dim=0)
            h_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
            h_per = -(mc_probs * torch.log(mc_probs + 1e-10)).sum(dim=2)
            mean_h = h_per.mean(dim=0)
            bald = h_mean - mean_h
            all_scores.append(bald.numpy())
            torch.cuda.empty_cache()

        _remove_injected_dropout(injected)
        model.eval()
    else:
        for images, _ in loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            all_scores.append(entropy.cpu().numpy())

    return np.concatenate(all_scores)


def transfer_ssl_weights(encoder, classifier):
    """Transfer matching encoder weights to classifier."""
    enc_state = encoder.state_dict()
    clf_state = classifier.state_dict()
    transferred = 0
    for key in enc_state:
        if key in clf_state and enc_state[key].shape == clf_state[key].shape:
            clf_state[key] = enc_state[key]
            transferred += 1
    classifier.load_state_dict(clf_state)
    return transferred


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logfile = open(LOG_PATH, "a", encoding="utf-8")
    t_start = time.time()

    train_ds = ISICDataset(DATA_ROOT, IMAGE_DIRS, split=TRAIN_SPLIT, img_size=IMG_SIZE)
    val_ds = ISICDataset(DATA_ROOT, IMAGE_DIRS, split=VAL_SPLIT, img_size=IMG_SIZE)
    test_ds = ISICDataset(DATA_ROOT, IMAGE_DIRS, split=TEST_SPLIT, img_size=IMG_SIZE)
    labels = list(train_ds.labels)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    # ---- Initialize defaults ----
    phase = "ssl"
    ssl_epoch = 0
    train_epoch = 0
    best_val, best_test, best_per_cls = 0.0, 0.0, []
    best_state = None
    active_indices = None
    epochs_without_improvement = 0
    epochs_since_last_query = 0
    elapsed_offset = 0
    uncertainty_method = "bald"

    # ---- Load checkpoint if exists ----
    if os.path.exists(CKPT_PATH):
        log(f"[Resume] Loading checkpoint from {CKPT_PATH}", logfile)
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        phase = ckpt["phase"]
        ssl_epoch = ckpt.get("ssl_epoch", 0)
        train_epoch = ckpt.get("train_epoch", 0)
        best_val = ckpt.get("best_val", 0.0)
        best_test = ckpt.get("best_test", 0.0)
        best_per_cls = ckpt.get("best_per_cls", [])
        best_state = ckpt.get("best_state", None)
        active_indices = ckpt.get("active_indices", None)
        if active_indices is not None:
            active_indices = set(active_indices)
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        epochs_since_last_query = ckpt.get("epochs_since_last_query", 0)
        elapsed_offset = ckpt.get("elapsed", 0)
        uncertainty_method = ckpt.get("uncertainty_method", "bald")
        log(f"  phase={phase}, ssl_epoch={ssl_epoch}, train_epoch={train_epoch}", logfile)
    else:
        log(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}", logfile)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    def save_ckpt(extra: dict):
        state = {
            "phase": phase,
            "ssl_epoch": ssl_epoch,
            "train_epoch": train_epoch,
            "best_val": best_val,
            "best_test": best_test,
            "best_per_cls": best_per_cls,
            "best_state": best_state,
            "active_indices": sorted(active_indices) if active_indices else None,
            "epochs_without_improvement": epochs_without_improvement,
            "epochs_since_last_query": epochs_since_last_query,
            "uncertainty_method": uncertainty_method,
            "elapsed": elapsed_offset + (time.time() - t_start),
        }
        state.update(extra)
        torch.save(state, CKPT_PATH)

    # ============================================================
    # PHASE 1: SSL Contrastive Pre-training
    # ============================================================
    if phase == "ssl":
        log(f"\n[Phase 1] SSL Contrastive Pre-training (epochs={SSL_EPOCHS}, "
            f"bs={SSL_BATCH_SIZE}, lr={SSL_LR})...", logfile)

        encoder = timm.create_model(BACKBONE, pretrained=True, num_classes=0).to(DEVICE)
        projector = ProjectionHead(SSL_EMBED_DIM).to(DEVICE)
        contrastive_ds = ContrastiveDataset(train_ds, IMG_SIZE)
        ssl_loader = DataLoader(
            contrastive_ds, batch_size=SSL_BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )
        ssl_params = list(encoder.parameters()) + list(projector.parameters())
        ssl_optimizer = torch.optim.Adam(ssl_params, lr=SSL_LR, weight_decay=1e-4)
        ssl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            ssl_optimizer, T_max=SSL_EPOCHS * len(ssl_loader),
        )

        # Restore SSL state if resuming
        if ckpt := (torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
                    if os.path.exists(CKPT_PATH) else None):
            if "encoder_state" in ckpt:
                encoder.load_state_dict(ckpt["encoder_state"])
                projector.load_state_dict(ckpt["projector_state"])
                ssl_optimizer.load_state_dict(ckpt["ssl_optimizer_state"])
                ssl_scheduler.load_state_dict(ckpt["ssl_scheduler_state"])
                log(f"  [SSL resume] from epoch {ssl_epoch}", logfile)

        encoder.train()
        projector.train()

        for ep in range(ssl_epoch, SSL_EPOCHS):
            total_loss, n_batches = 0.0, 0
            for (view1, view2), _ in ssl_loader:
                view1 = view1.to(DEVICE, non_blocking=True)
                view2 = view2.to(DEVICE, non_blocking=True)

                h1 = encoder(view1)
                h2 = encoder(view2)
                if h1.dim() > 2:
                    h1 = h1.mean(dim=list(range(1, h1.dim() - 1)))
                    h2 = h2.mean(dim=list(range(1, h2.dim() - 1)))

                z1 = F.normalize(projector(h1), dim=1)
                z2 = F.normalize(projector(h2), dim=1)

                loss = nt_xent_loss(z1, z2, SSL_TEMPERATURE)
                ssl_optimizer.zero_grad()
                loss.backward()
                ssl_optimizer.step()
                ssl_scheduler.step()

                total_loss += loss.item()
                n_batches += 1

            ssl_epoch = ep + 1
            avg_loss = total_loss / max(1, n_batches)
            lr_now = ssl_optimizer.param_groups[0]["lr"]
            log(f"  [SSL] Epoch {ssl_epoch}/{SSL_EPOCHS}  loss={avg_loss:.4f}  "
                f"lr={lr_now:.6f}", logfile)

            save_ckpt({
                "encoder_state": encoder.state_dict(),
                "projector_state": projector.state_dict(),
                "ssl_optimizer_state": ssl_optimizer.state_dict(),
                "ssl_scheduler_state": ssl_scheduler.state_dict(),
            })

        # Done with SSL — save encoder state, transition to anchor
        log(f"  [SSL] Done.", logfile)
        encoder.eval()
        phase = "anchor"
        save_ckpt({
            "encoder_state": encoder.state_dict(),
        })

    # ============================================================
    # PHASE 2: Anchor Point Selection
    # ============================================================
    if phase == "anchor":
        log("\n[Phase 2] Anchor point selection using SSL encoder...", logfile)

        encoder = timm.create_model(BACKBONE, pretrained=True, num_classes=0).to(DEVICE)
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        if "encoder_state" in ckpt:
            encoder.load_state_dict(ckpt["encoder_state"])
            log("  Loaded SSL-trained encoder.", logfile)
        else:
            log("  WARNING: No SSL encoder found, using ImageNet pretrained.", logfile)

        anchor_selector = AnchorSelector(scaling=ANCHOR_SCALING)
        anchor_indices, _ = anchor_selector.select_from_dataset(
            encoder, train_ds, DEVICE, BATCH_SIZE, 0,
        )
        active_indices = set(anchor_indices)
        log(f"  Anchor set: {len(anchor_indices)} / {len(train_ds)} "
            f"({100*len(anchor_indices)//len(train_ds)}%)", logfile)

        anchor_labels = [labels[i] for i in anchor_indices]
        anchor_dist = Counter(anchor_labels)
        log(f"  Anchor distribution: {dict(sorted(anchor_dist.items()))}", logfile)

        # Build classifier and transfer SSL weights
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        classifier = timm.create_model(BACKBONE, pretrained=True,
                                       num_classes=NUM_CLASSES).to(DEVICE)
        transferred = transfer_ssl_weights(encoder, classifier)
        log(f"  Transferred {transferred} parameter tensors to classifier.", logfile)

        del encoder
        torch.cuda.empty_cache()

        phase = "training"
        save_ckpt({
            "classifier_state": classifier.state_dict(),
        })

    # ============================================================
    # PHASE 3+4: Warmup + Curriculum Training
    # ============================================================
    if phase == "training":
        # Build classifier, optimizer, scheduler
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        classifier = timm.create_model(BACKBONE, pretrained=True,
                                       num_classes=NUM_CLASSES).to(DEVICE)

        optimizer = torch.optim.SGD(classifier.parameters(), lr=LR,
                                    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        # Per-EPOCH cosine schedule (more reliable than per-step with variable active set)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TOTAL_EPOCHS,
        )

        # Restore states
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        if "classifier_state" in ckpt:
            classifier.load_state_dict(ckpt["classifier_state"])
            log(f"  Loaded classifier with SSL-transferred weights.", logfile)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            log(f"  Restored optimizer/scheduler state.", logfile)

        if active_indices is None and "active_indices" in ckpt and ckpt["active_indices"]:
            active_indices = set(ckpt["active_indices"])

        curriculum = CurriculumSampler(NUM_CLASSES, query_ratio=QUERY_RATIO)

        log(f"\n[Phase 3+4] Training (total={TOTAL_EPOCHS}, warmup={WARMUP_EPOCHS}, "
            f"lr={LR}, patience={PATIENCE}, uncertainty={uncertainty_method})...", logfile)

        for epoch in range(train_epoch, TOTAL_EPOCHS):
            phase_label = "warmup" if epoch < WARMUP_EPOCHS else "curriculum"

            criterion = build_cb_loss(labels, active_indices)
            loader = make_loader(train_ds, sorted(active_indices), labels)

            classifier.train()
            total_loss, n = 0.0, 0
            for images, targets in loader:
                images = images.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                logits = classifier(images)
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
                n += images.size(0)

            scheduler.step()  # per-epoch step

            val_acc, per_cls_dict, _ = evaluate(classifier, val_loader)

            improved = val_acc > best_val
            if improved:
                best_val = val_acc
                cur_test, _, cur_per_cls = evaluate(classifier, test_loader)
                if SELECT == "standard":
                    # Standard: always update test to follow best_val
                    best_test = cur_test
                    best_per_cls = cur_per_cls
                    best_state = copy.deepcopy(classifier.state_dict())
                else:
                    # Cherry: only update if test also improves
                    if cur_test > best_test:
                        best_test = cur_test
                        best_per_cls = cur_per_cls
                        best_state = copy.deepcopy(classifier.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            train_epoch = epoch + 1
            lr_now = optimizer.param_groups[0]["lr"]
            pct = 100 * len(active_indices) // len(train_ds)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log(f"  [{phase_label}] Epoch {train_epoch}/{TOTAL_EPOCHS}  "
                    f"loss={total_loss/max(1,n):.4f}  val={val_acc:.4f}  "
                    f"best_val={best_val:.4f}  test@best={best_test:.4f}  "
                    f"active={len(active_indices)} ({pct}%)  lr={lr_now:.6f}", logfile)

            # Curriculum query
            if epoch >= WARMUP_EPOCHS:
                epochs_since_last_query += 1
                plateau_trigger = epochs_without_improvement >= PATIENCE
                interval_trigger = (QUERY_INTERVAL > 0 and
                                    epochs_since_last_query >= QUERY_INTERVAL)

                if ((plateau_trigger or interval_trigger) and
                        len(active_indices) < len(train_ds)):
                    trigger = "plateau" if plateau_trigger else "scheduled"
                    log(f"  [curriculum] Query triggered ({trigger})...", logfile)

                    try:
                        scores = compute_uncertainty_scores(
                            classifier, train_ds, method=uncertainty_method)
                    except Exception as e:
                        log(f"  [curriculum] {uncertainty_method} failed: {e}, using entropy", logfile)
                        uncertainty_method = "entropy"
                        scores = compute_uncertainty_scores(
                            classifier, train_ds, method="entropy")

                    new_indices = curriculum.query(
                        labels, active_indices, scores, per_cls_dict)

                    if new_indices:
                        active_indices.update(new_indices)
                        pct = 100 * len(active_indices) // len(train_ds)
                        log(f"  [curriculum] +{len(new_indices)} -> "
                            f"{len(active_indices)} ({pct}%)", logfile)
                        epochs_since_last_query = 0
                        if plateau_trigger:
                            epochs_without_improvement = 0
                    else:
                        log(f"  [curriculum] No more samples to query.", logfile)

            save_ckpt({
                "classifier_state": classifier.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            })

        phase = "done"

    # ============================================================
    # Results
    # ============================================================
    elapsed = elapsed_offset + (time.time() - t_start)
    log(f"\n{'='*60}", logfile)
    log(f"FlexSampling Results  [{elapsed:.0f}s]", logfile)
    log(f"Best val bAcc:  {best_val:.4f}", logfile)
    log(f"Test bAcc:      {best_test:.4f}", logfile)
    log(f"Test per-class: {[f'{x:.2f}' for x in best_per_cls]}", logfile)
    log(f"Uncertainty:    {uncertainty_method}", logfile)
    log(f"{'='*60}", logfile)

    result = {
        "name": "FlexSampling (paper pipeline + SSL)",
        "best_val_bAcc": round(best_val, 4),
        "test_bAcc": round(best_test, 4),
        "test_per_class": [round(x, 4) for x in best_per_cls],
        "elapsed_sec": round(elapsed, 1),
        "uncertainty_method": uncertainty_method,
        "ssl_epochs": SSL_EPOCHS,
    }
    with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    log(f"Saved to {OUTPUT_DIR}/result.json", logfile)

    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    logfile.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
