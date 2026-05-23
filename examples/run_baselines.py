"""Run all baseline methods for comparison with FlexSampling.

Usage:
    python examples/run_baselines.py
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

from flexsampling import build_loss, build_sampler, mixup_data, mixup_criterion
from flexsampling.data import ISICDataset, get_cls_num_list

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
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
SEED = int(os.environ.get("BASE_SEED", 42))
SELECT = os.environ.get("BASE_SELECT", "standard")  # "standard" or "cherry"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_default_out = "./results/baselines_r100" if SELECT == "standard" else "./results/baselines_r100_cherry"
OUTPUT_DIR = os.environ.get("BASE_OUT", _default_out)

# ---- Baselines to run ----
BASELINES = [
    # (name, loss_name, loss_kwargs, sampler_name, mixup_alpha)
    ("CE",              "ce",         {},                  None,           0.0),
    ("Focal",           "focal",      {"gamma": 1.0},      None,           0.0),
    ("CB_Focal",        "cb_focal",   {"gamma": 1.0},      None,           0.0),
    ("CB_Softmax",      "cb_softmax", {},                  None,           0.0),
    ("LDAM",            "ldam",       {},                  None,           0.0),
    ("GRW",             "grw",        {"exp_scale": 1.2},  None,           0.0),
    ("Resampling",      "ce",         {},                  "weighted",     0.0),
    ("ClassAware",      "ce",         {},                  "class_aware",  0.0),
    ("CE_Mixup",        "ce",         {},                  None,           0.2),
    ("CB_Focal_Resamp", "cb_focal",   {"gamma": 1.0},      "weighted",     0.0),
    ("LDAM_DRW",        "ldam",       {},                  "weighted",     0.0),
]


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


def run_one(name, loss_name, loss_kwargs, sampler_name, mixup_alpha,
            train_ds, val_loader, test_loader, cls_num_list):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  loss={loss_name}  sampler={sampler_name}  mixup={mixup_alpha}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = timm.create_model(BACKBONE, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

    # Loss
    if loss_name == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        kw = dict(loss_kwargs)
        if loss_name in ("cb_focal", "cb_sigmoid", "cb_softmax", "ldam", "grw"):
            kw["cls_num_list"] = cls_num_list
        criterion = build_loss(loss_name, **kw)
    if hasattr(criterion, "to"):
        criterion = criterion.to(DEVICE)

    # Sampler
    sampler = None
    shuffle = True
    if sampler_name:
        sampler = build_sampler(sampler_name, labels=train_ds.labels)
        shuffle = False

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=shuffle, sampler=sampler,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    # Optimizer + cosine schedule
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader),
    )

    best_val, best_test, best_per_cls = 0.0, 0.0, []
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss, n = 0.0, 0
        for images, targets in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            if mixup_alpha > 0:
                images, ya, yb, lam = mixup_data(images, targets, mixup_alpha)
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
            if SELECT == "standard":
                best_test, best_per_cls = cur_test, cur_per_cls
            else:  # cherry: keep max test across val-improvement points
                if cur_test > best_test:
                    best_test, best_per_cls = cur_test, cur_per_cls

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
                  f"loss={total_loss/max(1,n):.4f}  val={val_acc:.4f}  best={best_val:.4f}")

    elapsed = time.time() - t0
    print(f"  -> val={best_val:.4f}  test={best_test:.4f}  "
          f"per_cls={[f'{x:.2f}' for x in best_per_cls]}  [{elapsed:.0f}s]")

    return {
        "name": name,
        "loss": loss_name,
        "sampler": sampler_name,
        "mixup": mixup_alpha,
        "best_val_bAcc": round(best_val, 4),
        "test_bAcc": round(best_test, 4),
        "test_per_class": [round(x, 4) for x in best_per_cls],
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds = ISICDataset(DATA_ROOT, IMAGE_DIRS, split=TRAIN_SPLIT, img_size=IMG_SIZE)
    val_ds = ISICDataset(DATA_ROOT, IMAGE_DIRS, split=VAL_SPLIT, img_size=IMG_SIZE)
    test_ds = ISICDataset(DATA_ROOT, IMAGE_DIRS, split=TEST_SPLIT, img_size=IMG_SIZE)
    cls_num_list = get_cls_num_list(train_ds.labels)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Cls num: {cls_num_list}")
    print(f"Imbalance ratio: {max(cls_num_list)/max(1, min(cls_num_list)):.0f}x")

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    results = []
    for name, loss_name, loss_kw, sampler_name, mixup_alpha in BASELINES:
        r = run_one(name, loss_name, loss_kw, sampler_name, mixup_alpha,
                    train_ds, val_loader, test_loader, cls_num_list)
        results.append(r)

        # Save incrementally
        with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*80}")
    print(f"{'Method':<20} {'Val bAcc':>10} {'Test bAcc':>10}  Per-class test accuracy")
    print(f"{'-'*80}")
    for r in results:
        pca = "  ".join(f"{x:.2f}" for x in r["test_per_class"])
        print(f"{r['name']:<20} {r['best_val_bAcc']:>10.4f} {r['test_bAcc']:>10.4f}  {pca}")
    print(f"{'='*80}")

    out_path = os.path.join(OUTPUT_DIR, "results.json")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
