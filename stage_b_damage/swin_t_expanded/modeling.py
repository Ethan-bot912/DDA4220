"""Swin-T construction for the local FE-001 experiment."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import Swin_T_Weights, swin_t


class SwinDamageModel(nn.Module):
    """Swin-T with a four-class head and optional ordinal auxiliary head."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_classes: int,
        head_dropout: float,
        ordinal: bool,
    ) -> None:
        super().__init__()
        in_features = int(backbone.head.in_features)
        backbone.head = nn.Identity()
        self.backbone = backbone
        if float(head_dropout) > 0.0:
            self.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(float(head_dropout)),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.head = nn.Linear(in_features, num_classes)
        self.ordinal_head = nn.Linear(in_features, 3) if ordinal else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.norm(x)
        x = self.backbone.permute(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor):
        features = self.forward_features(x)
        logits = self.head(features)
        if self.ordinal_head is None:
            return logits
        return {"logits": logits, "ordinal": self.ordinal_head(features)}


def build_model(
    *,
    num_classes: int = 4,
    pretrained: bool = True,
    head_dropout: float = 0.0,
    ordinal: bool = False,
) -> tuple[nn.Module, str]:
    weights: Any = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = swin_t(weights=weights)
    if ordinal:
        model = SwinDamageModel(
            backbone,
            num_classes=num_classes,
            head_dropout=head_dropout,
            ordinal=True,
        )
    else:
        in_features = backbone.head.in_features
        if float(head_dropout) > 0.0:
            backbone.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(float(head_dropout)),
                nn.Linear(in_features, num_classes),
            )
        else:
            backbone.head = nn.Linear(in_features, num_classes)
        model = backbone
    return model, ("IMAGENET1K_V1" if pretrained else "none")

