"""Swin-T construction for the local FE-001 experiment."""

from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import Swin_T_Weights, swin_t


def build_model(*, num_classes: int = 4, pretrained: bool = True) -> tuple[nn.Module, str]:
    weights: Any = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = swin_t(weights=weights)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model, ("IMAGENET1K_V1" if pretrained else "none")

