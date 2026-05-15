"""Training script for long-tailed skin lesion classification.

Usage:
    python examples/train.py --config examples/configs/isic_8class.yaml
    python examples/train.py --config examples/configs/isic_8class.yaml --loss.name ldam --sampler.name weighted
"""
import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, classification_report

from flexsampling import build_loss, build_sampler, mixup_data, mixup_criterion
from flexsampling.data import ISICDataset, get_cls_num_list


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override_config(cfg: dict, overrides: list) -> dict:
    """Apply --key.subkey value overrides to config dict."""
    for i in range(0, len(overrides), 2):
        keys = overrides[i].lstrip("-").split(".")
        val = overrides[i + 1]
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Auto-convert types
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        d[keys[-1]] = val
    return cfg


def build_model(cfg: dict) -> nn.Module:
    import timm
    model = timm.create_model(
        cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"],
    )
    return model


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    tcfg = cfg["training"]
    total_steps = tcfg["epochs"] * steps_per_epoch
    warmup_steps = tcfg.get("warmup_epochs", 0) * steps_per_epoch

    if tcfg.get("lr_schedule") == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        schedulers = []
        milestones = []
        if warmup_steps > 0:
            schedulers.append(LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps))
            milestones.append(warmup_steps)
        schedulers.append(CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps))
        if len(schedulers) > 1:
            return SequentialLR(optimizer, schedulers, milestones=milestones)
        return schedulers[0]
    else:
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = [m * steps_per_epoch for m in tcfg.get("step_milestones", [30, 40])]
        return MultiStepLR(optimizer, milestones=milestones, gamma=tcfg.get("step_gamma", 0.1))


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(targets)
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    bal_acc = balanced_accuracy_score(targets, preds)
    per_cls_acc = []
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            per_cls_acc.append((preds[mask] == c).mean())
        else:
            per_cls_acc.append(0.0)
    return {
        "balanced_acc": float(bal_acc),
        "per_class_acc": [float(a) for a in per_cls_acc],
        "preds": preds,
        "targets": targets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config)
    if overrides:
        cfg = override_config(cfg, overrides)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dcfg = cfg["data"]
    train_ds = ISICDataset(dcfg["root"], dcfg["image_dir"], split="train", img_size=dcfg.get("img_size", 224))
    val_ds = ISICDataset(dcfg["root"], dcfg["image_dir"], split="val", img_size=dcfg.get("img_size", 224))
    test_ds = ISICDataset(dcfg["root"], dcfg["image_dir"], split="test", img_size=dcfg.get("img_size", 224))

    cls_num_list = get_cls_num_list(train_ds.labels)
    num_classes = cfg["model"]["num_classes"]
    print(f"Classes: {num_classes}, samples per class: {cls_num_list}")
    print(f"Imbalance ratio: {max(cls_num_list)/max(1, min(cls_num_list)):.1f}x")

    # Sampler
    scfg = cfg.get("sampler", {})
    sampler_name = scfg.get("name")
    sampler = None
    shuffle = True
    if sampler_name and sampler_name != "null":
        sampler_kwargs = {k: v for k, v in scfg.items() if k != "name"}
        sampler = build_sampler(sampler_name, labels=train_ds.labels, **sampler_kwargs)
        shuffle = False

    tcfg = cfg["training"]
    bs = tcfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=shuffle, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = build_model(cfg).to(device)
    print(f"Model: {cfg['model']['backbone']}, params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    lcfg = cfg.get("loss", {})
    loss_name = lcfg.get("name", "focal")
    loss_kwargs = {k: v for k, v in lcfg.items() if k != "name"}
    if loss_name in ("cb_focal", "cb_sigmoid", "cb_softmax", "ldam", "grw"):
        loss_kwargs["cls_num_list"] = cls_num_list
    criterion = build_loss(loss_name, **loss_kwargs)
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)
    print(f"Loss: {loss_name}")

    # Optimizer & scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=tcfg["lr"],
        momentum=tcfg.get("momentum", 0.9),
        weight_decay=tcfg.get("weight_decay", 1e-4),
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    mixup_alpha = cfg.get("augmentation", {}).get("mixup_alpha", 0.0)

    # Training loop
    os.makedirs(cfg.get("output_dir", "./results"), exist_ok=True)
    best_val_acc = 0.0
    best_test_result = None
    t0 = time.time()

    for epoch in range(tcfg["epochs"]):
        model.train()
        total_loss, n_samples = 0.0, 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_alpha > 0:
                images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
                logits = model(images)
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)

        avg_loss = total_loss / max(1, n_samples)
        val_result = evaluate(model, val_loader, device, num_classes)
        val_acc = val_result["balanced_acc"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_result = evaluate(model, test_loader, device, num_classes)
            ckpt_path = os.path.join(cfg.get("output_dir", "./results"), "best.pth")
            torch.save(model.state_dict(), ckpt_path)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{tcfg['epochs']}  "
            f"loss={avg_loss:.4f}  val_bAcc={val_acc:.4f}  "
            f"best={best_val_acc:.4f}  lr={lr_now:.6f}"
        )

    elapsed = time.time() - t0

    if best_test_result is None:
        best_test_result = evaluate(model, test_loader, device, num_classes)

    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.0f}s")
    print(f"Best val balanced acc: {best_val_acc:.4f}")
    print(f"Test balanced acc:     {best_test_result['balanced_acc']:.4f}")
    print(f"Test per-class acc:    {best_test_result['per_class_acc']}")
    print(f"\nClassification report:")
    print(classification_report(
        best_test_result["targets"], best_test_result["preds"],
        zero_division=0,
    ))


if __name__ == "__main__":
    main()
