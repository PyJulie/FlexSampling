# FlexSampling

Official PyTorch implementation of **"Flexible Sampling for Long-tailed Skin Lesion Classification"** (MICCAI 2022).

[[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-16437-8_46) [[arXiv]](https://arxiv.org/abs/2204.03161)

## Method

FlexSampling is a curriculum-learning-based framework that addresses class imbalance by dynamically expanding the training set based on learning difficulty. Unlike conventional re-balancing methods that treat all classes equally, FlexSampling adapts sampling to each class's current learning status.

**Pipeline:**

1. **Self-supervised pre-training** — Contrastive learning extracts distribution-agnostic features, avoiding semantic bias toward majority classes.
2. **Anchor point selection** — Class prototypes (mean features) guide selection of representative samples nearest to each centroid, forming a less-imbalanced initial subset (controlled by scaling factor *s*).
3. **Warm-up training** — The classifier trains on the anchor set for a warm-up phase.
4. **Curriculum sampling** — When validation accuracy plateaus:
   - **Class-wise probability**: `p_cj = 1 - accuracy(cj)` — classes with lower accuracy receive higher sampling probability.
   - **Instance-wise selection**: BALD (Bayesian Active Learning by Disagreement) scores rank unsampled instances by uncertainty; the most uncertain are queried first.
   - New samples are merged into the active training set and the optimizer is re-initialized.

## Installation

```bash
pip install -e .

# With demo dependencies (timm, pyyaml, pandas, scikit-learn)
pip install -e ".[demo]"
```

## Quick Start

### Run FlexSampling (proposed method)

```bash
python examples/train.py --config examples/configs/isic_8class.yaml
```

### Run baseline comparison

```bash
python examples/train.py --config examples/configs/isic_8class.yaml --baseline.enabled true
```

### Use as a library

```python
from flexsampling import FlexSamplingTrainer

trainer = FlexSamplingTrainer(
    encoder=ssl_encoder,          # pre-trained feature extractor
    classifier=model,             # full model with classification head
    train_dataset=train_ds,
    val_dataset=val_ds,
    num_classes=8,
    device=device,
    anchor_scaling=0.1,           # controls initial subset imbalance
    warmup_epochs=30,
    total_epochs=100,
    patience=10,                  # epochs before curriculum query
    query_ratio=0.1,
)
history = trainer.run()
```

### Use individual components

```python
from flexsampling import build_loss, build_sampler, mixup_data, mixup_criterion
from flexsampling.core import AnchorSelector, BALDUncertainty, CurriculumSampler

# Anchor selection
selector = AnchorSelector(scaling=0.1)
anchor_indices, prototypes = selector.select_from_dataset(encoder, dataset, device)

# BALD uncertainty
bald = BALDUncertainty(n_samples=10)
scores = bald.score(model, dataset, device)

# Curriculum sampling
curriculum = CurriculumSampler(num_classes=8, query_ratio=0.1)
new_indices = curriculum.query(labels, active_set, scores, val_acc_per_class)

# Long-tailed losses
criterion = build_loss("cb_focal", cls_num_list=[500, 200, 100, 50, 20, 10, 5, 2])
criterion = build_loss("ldam", cls_num_list=[500, 200, 100, 50, 20, 10, 5, 2])

# Balanced samplers
sampler = build_sampler("class_aware", labels=train_labels)
```

### Available components

| Category | Name | Description |
|----------|------|-------------|
| **Core** | `FlexSamplingTrainer` | Full pipeline: anchor selection → warm-up → curriculum sampling |
| **Core** | `AnchorSelector` | Prototype-based anchor point selection |
| **Core** | `BALDUncertainty` | MC Dropout uncertainty via mutual information |
| **Core** | `CurriculumSampler` | Difficulty-aware dynamic sample querying |
| Loss | `focal` | Focal Loss (Lin et al., ICCV 2017) |
| Loss | `cb_focal` | Class-Balanced Focal Loss (Cui et al., CVPR 2019) |
| Loss | `cb_softmax` | Class-Balanced Softmax CE |
| Loss | `cb_sigmoid` | Class-Balanced Sigmoid CE |
| Loss | `ldam` | Label-Distribution-Aware Margin (Cao et al., NeurIPS 2019) |
| Loss | `grw` | Generalized Reweight Loss |
| Sampler | `weighted` | Inverse-frequency weighted random sampling |
| Sampler | `class_aware` | Uniform class sampling with per-class cycling |
| Aug | `mixup_data` | Mixup input interpolation (Zhang et al., ICLR 2018) |

## Dataset

Download ISIC 2019 images and place them in a directory. The NumPy split files define which images belong to each split and their labels.

```
dataset/
  8-class/
    dic.npy            # {filename: label_index}
    train_100.npy      # filenames for training (imbalance ratio 100)
    train_200.npy
    train_500.npy
    val_100.npy / val_200.npy / val_500.npy
    test_100.npy / test_200.npy / test_500.npy
  14-class/
    dic.npy
    train.npy / val.npy / test.npy
```

## Project Structure

```
FlexSampling/
  flexsampling/
    core/              # Proposed method
      anchor.py        #   Anchor point selection (Section 2.2)
      bald.py          #   BALD uncertainty estimation (Section 2.3.2)
      curriculum.py    #   Curriculum sampling module (Section 2.3)
      pipeline.py      #   Full FlexSampling trainer
    losses/            # Long-tailed loss functions
    samplers/          # Re-balancing samplers
    augmentations/     # Mixup
    data/              # ISIC dataset loader
  examples/
    train.py           # Training script (FlexSampling + baselines)
    configs/           # YAML configs
  dataset/             # Pre-processed splits (NumPy)
```

## Citation

```bibtex
@inproceedings{ju2022flexible,
  title={Flexible Sampling for Long-tailed Skin Lesion Classification},
  author={Ju, Lie and Wu, Yicheng and Wang, Lin and Yu, Zhen and Zhao, Xin and Wang, Xin and Bonnington, Paul and Ge, Zongyuan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={462--471},
  year={2022},
  organization={Springer}
}
```

## License

MIT
