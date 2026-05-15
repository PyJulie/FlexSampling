from typing import Dict, List
from torch.utils.data import Sampler

from flexsampling.samplers.resampling import WeightedResampler
from flexsampling.samplers.class_aware import ClassAwareSampler

SAMPLER_REGISTRY: Dict[str, type] = {
    "weighted": WeightedResampler,
    "class_aware": ClassAwareSampler,
}


def build_sampler(name: str, labels: List[int], **kwargs) -> Sampler:
    """Build a sampler by name.

    Examples::

        sampler = build_sampler("weighted", labels=train_labels)
        sampler = build_sampler("class_aware", labels=train_labels, num_samples_per_cls=4)
    """
    if name not in SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown sampler '{name}'. Available: {list(SAMPLER_REGISTRY.keys())}"
        )
    return SAMPLER_REGISTRY[name](labels=labels, **kwargs)


__all__ = [
    "WeightedResampler",
    "ClassAwareSampler",
    "build_sampler",
    "SAMPLER_REGISTRY",
]
