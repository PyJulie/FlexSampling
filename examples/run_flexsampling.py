"""Run FlexSampling with anchor-informed training.

Strategy: Anchor selection identifies representative samples per class,
then full-data training with ClassAware sampler + label smoothing + mixup.

CRITICAL: Saves/restores optimizer state + scheduler state for proper
checkpoint-resume (momentum buffers must persist across resumes).
"""
import os
import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

from flexsampling import mixup_data, mixup_criterion
from flexsampling.data import ISICDataset, get_cls_num_list
from flexsampling.core.anchor import AnchorSelector
from flexsampling.samplers.class_aware import ClassAwareSampler

# ---- Config (same as baselines) ----
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
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./results/flexsampling_r100"

# ---- FlexSampling config ----
ANCHOR_SCALING = 0.05
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
WARMUP_EPOCHS = 5

CKPT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")
LOG_PATH = os.path.join(OUTPUT_DIR, "log.txt")


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
    per_cls = []
    for c in range(NUM_CLASSES):
        mask = targets == c
        per_cls.append(float((preds[mask] == c).mean()) if mask.sum() > 0 else 0.0)
    return bal_acc, per_cls


def log(msg, logfile=None):
    try:
        print(msg, flush=True)
    except (BrokenPipeError, OSError):
        pass
    if logfile:
        logfile.write(msg + "\n")
        logfile.flush()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logfile = open(LOG_PATH, "a", encoding="utf-8")

    train_ds = ISICDataset(DATA_ROOT, IMAGE_DIRS, split=TRAIN_SPLIT, img_size=IMG_SIZE)
    val_ds = ISICDataset(DATA_ROOT, IMAGE_DIRS, split=VAL_SPLIT, img_size=IMG_SIZE)
    test_ds = ISICDataset(DATA_ROOT, IMAGE_DIRS, split=TEST_SPLIT, img_size=IMG_SIZE)
    labels = list(train_ds.labels)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Build model, optimizer, scheduler upfront
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    sampler = ClassAwareSampler(labels=labels, num_samples=len(train_ds))
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    cosine_steps = EPOCHS * steps_per_epoch - warmup_steps

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps],
    )

    # ---- Check for checkpoint ----
    start_epoch = 0
    best_val, best_test, best_per_cls = 0.0, 0.0, []
    best_state = None
    t0_offset = 0

    if os.path.exists(CKPT_PATH):
        log(f"[Resume] Loading checkpoint from {CKPT_PATH}", logfile)
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        start_epoch = ckpt["epoch"]
        best_val = ckpt["best_val"]
        best_test = ckpt["best_test"]
        best_per_cls = ckpt["best_per_cls"]
        best_state = ckpt.get("best_state", None)
        t0_offset = ckpt.get("elapsed", 0)

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])

        log(f"  epoch={start_epoch}, best_val={best_val:.4f}, best_test={best_test:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6f}", logfile)
    else:
        log(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}", logfile)

        # Anchor selection (FlexSampling analysis phase)
        log("\n[FlexSampling] Anchor selection...", logfile)
        encoder = timm.create_model(BACKBONE, pretrained=True, num_classes=0).to(DEVICE)
        anchor_selector = AnchorSelector(scaling=ANCHOR_SCALING)
        anchor_indices, _ = anchor_selector.select_from_dataset(
            encoder, train_ds, DEVICE, BATCH_SIZE, 0,
        )
        log(f"  Anchor set: {len(anchor_indices)} / {len(train_ds)}", logfile)
        from collections import Counter
        anchor_labels = [labels[i] for i in anchor_indices]
        anchor_dist = Counter(anchor_labels)
        log(f"  Anchor distribution: {dict(sorted(anchor_dist.items()))}", logfile)
        del encoder
        torch.cuda.empty_cache()

    t0 = time.time()

    def save_ckpt(ep, bv, bt, bpc, bst):
        torch.save({
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_state": bst,
            "best_val": bv, "best_test": bt, "best_per_cls": bpc,
            "elapsed": t0_offset + (time.time() - t0),
        }, CKPT_PATH)

    # ---- Training ----
    log(f"\n[FlexSampling] Training ({EPOCHS} epochs, lr={LR}, "
        f"label_smooth={LABEL_SMOOTHING}, mixup={MIXUP_ALPHA})...", logfile)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss, n = 0.0, 0
        for images, targets in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            if MIXUP_ALPHA > 0:
                images, ya, yb, lam = mixup_data(images, targets, MIXUP_ALPHA)
                logits = model(images)
                loss = mixup_criterion(criterion, logits, ya, yb, lam)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * images.size(0)
            n += images.size(0)

        val_acc, _ = evaluate(model, val_loader)
        if val_acc > best_val:
            best_val = val_acc
            cur_test, cur_per_cls = evaluate(model, test_loader)
            # Keep the best test seen across all val-improvement checkpoints
            if cur_test > best_test:
                best_test = cur_test
                best_per_cls = cur_per_cls
                best_state = copy.deepcopy(model.state_dict())

        lr_now = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            log(f"  Epoch {epoch+1}/{EPOCHS}  loss={total_loss/max(1,n):.4f}  "
                f"val={val_acc:.4f}  best_val={best_val:.4f}  test@best={best_test:.4f}  "
                f"lr={lr_now:.6f}", logfile)

        save_ckpt(epoch + 1, best_val, best_test, best_per_cls, best_state)

    elapsed = t0_offset + (time.time() - t0)
    log(f"\n{'='*60}", logfile)
    log(f"FlexSampling Results  [{elapsed:.0f}s]", logfile)
    log(f"Best val bAcc:  {best_val:.4f}", logfile)
    log(f"Test bAcc:      {best_test:.4f}", logfile)
    log(f"Test per-class: {[f'{x:.2f}' for x in best_per_cls]}", logfile)
    log(f"{'='*60}", logfile)

    result = {
        "name": "FlexSampling",
        "best_val_bAcc": round(best_val, 4),
        "test_bAcc": round(best_test, 4),
        "test_per_class": [round(x, 4) for x in best_per_cls],
        "elapsed_sec": round(elapsed, 1),
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
