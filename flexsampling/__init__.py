from flexsampling.core import (
    AnchorSelector,
    BALDUncertainty,
    CurriculumSampler,
    FlexSamplingTrainer,
)
from flexsampling.losses import build_loss, LOSS_REGISTRY
from flexsampling.samplers import build_sampler, SAMPLER_REGISTRY
from flexsampling.augmentations.mixup import mixup_data, mixup_criterion

__version__ = "0.2.0"

__all__ = [
    # Core method
    "AnchorSelector",
    "BALDUncertainty",
    "CurriculumSampler",
    "FlexSamplingTrainer",
    # Losses & samplers
    "build_loss",
    "build_sampler",
    "mixup_data",
    "mixup_criterion",
    "LOSS_REGISTRY",
    "SAMPLER_REGISTRY",
]
