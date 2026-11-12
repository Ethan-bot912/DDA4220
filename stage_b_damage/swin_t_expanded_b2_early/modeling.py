"""Swin-T construction for the local Stage-B B1/B2 experiments."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import Swin_T_Weights, swin_t


def expand_swin_patch_embedding(backbone: nn.Module, input_channels: int) -> nn.Module:
    """Expand the RGB patch stem while preserving the pretrained response.

    The six-channel B2 input is ordered ``pre`` then ``post``.  Copying the
    RGB kernel into both halves and dividing by two makes the stem response
    identical to the original RGB stem when the two timestamps are equal.
    The constructor's RNG use is restored so adding the deterministic stem
    does not perturb subsequent model/training randomness.
    """
    input_channels = int(input_channels)
    if input_channels == 3:
        return backbone
    if input_channels != 6:
        raise ValueError(f"Only 3- or 6-channel Swin-T inputs are supported, got {input_channels}")

    stem = backbone.features[0][0]
    if not isinstance(stem, nn.Conv2d):
        raise TypeError(f"Unexpected Swin-T patch stem: {type(stem)!r}")
    if stem.in_channels == input_channels:
        return backbone
    if stem.in_channels != 3:
        raise ValueError(f"Expected an RGB patch stem, got {stem.in_channels} channels")

    cpu_rng = torch.get_rng_state()
    expanded = nn.Conv2d(
        input_channels,
        stem.out_channels,
        kernel_size=stem.kernel_size,
        stride=stem.stride,
        padding=stem.padding,
        dilation=stem.dilation,
        groups=stem.groups,
        bias=stem.bias is not None,
        padding_mode=stem.padding_mode,
        device=stem.weight.device,
        dtype=stem.weight.dtype,
    )
    torch.set_rng_state(cpu_rng)
    with torch.no_grad():
        expanded.weight.copy_(torch.cat((stem.weight, stem.weight), dim=1) / 2.0)
        if stem.bias is not None and expanded.bias is not None:
            expanded.bias.copy_(stem.bias)
    backbone.features[0][0] = expanded
    return backbone


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
    input_channels: int = 3,
) -> tuple[nn.Module, str]:
    weights: Any = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = swin_t(weights=weights)
    if ordinal:
        expand_swin_patch_embedding(backbone, input_channels)
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
        expand_swin_patch_embedding(backbone, input_channels)
        model = backbone
    return model, ("IMAGENET1K_V1" if pretrained else "none")

