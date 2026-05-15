from typing import Any, Callable, Dict, List, Optional
import torch

from flexsampling.losses.focal import FocalLoss
from flexsampling.losses.cb_loss import CBLoss
from flexsampling.losses.ldam import LDAMLoss
from flexsampling.losses.grw import GRWLoss

LOSS_REGISTRY: Dict[str, type] = {
    "focal": FocalLoss,
    "cb_focal": CBLoss,
    "cb_sigmoid": CBLoss,
    "cb_softmax": CBLoss,
    "ldam": LDAMLoss,
    "grw": GRWLoss,
}

_CB_TYPE_MAP = {
    "cb_focal": "focal",
    "cb_sigmoid": "sigmoid",
    "cb_softmax": "softmax",
}


def build_loss(name: str, **kwargs) -> torch.nn.Module:
    """Build a loss function by name.

    Examples::

        loss = build_loss("focal", gamma=2.0)
        loss = build_loss("cb_focal", cls_num_list=[500, 100, 20], gamma=1.0)
        loss = build_loss("ldam", cls_num_list=[500, 100, 20], max_margin=0.5)
        loss = build_loss("grw", cls_num_list=[500, 100, 20], exp_scale=1.2)
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    if name in _CB_TYPE_MAP:
        kwargs["loss_type"] = _CB_TYPE_MAP[name]
    return LOSS_REGISTRY[name](**kwargs)


__all__ = [
    "FocalLoss",
    "CBLoss",
    "LDAMLoss",
    "GRWLoss",
    "build_loss",
    "LOSS_REGISTRY",
]
