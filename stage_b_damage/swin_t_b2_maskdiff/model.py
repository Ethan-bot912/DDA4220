"""Shared Swin-T with stage3/4 pre/post spatial difference and building-mask pooling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import Swin_T_Weights, swin_t


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_rgb(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (image - mean) / std


class MaskDiffSwinT(nn.Module):
    """Shared Swin-T with spatial differences and predicted-mask pooling."""

    def __init__(
        self,
        *,
        num_classes: int = 4,
        pretrained: bool = True,
        head_dropout: float = 0.2,
        gradient_checkpointing: bool = True,
        mask_dilate_pixels: int = 2,
        minimum_mask_fraction: float = 0.02,
    ) -> None:
        super().__init__()
        weights: Any = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = swin_t(weights=weights)
        self.encoder.head = nn.Identity()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.mask_dilate_pixels = int(mask_dilate_pixels)
        self.minimum_mask_fraction = float(minimum_mask_fraction)

        fused_channels = 192
        self.post_proj3 = nn.Conv2d(384, fused_channels, 1, bias=False)
        self.pre_proj3 = nn.Conv2d(384, fused_channels, 1, bias=False)
        self.post_proj4 = nn.Conv2d(768, fused_channels, 1, bias=False)
        self.pre_proj4 = nn.Conv2d(768, fused_channels, 1, bias=False)
        self.stage_norm = nn.GroupNorm(16, fused_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels * 4, fused_channels * 4, 1, bias=False),
            nn.GroupNorm(32, fused_channels * 4),
            nn.GELU(),
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(fused_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_channels * 4),
            nn.Dropout(float(head_dropout)),
            nn.Linear(fused_channels * 4, num_classes),
        )

    def _call(self, module: nn.Module, value: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(module, value, use_reentrant=False)
        return module(value)

    def stage_features(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stage-3 and stage-4 feature maps in NCHW form."""
        features = self.encoder.features
        value = self._call(features[0], image)
        value = self._call(features[1], value)
        value = self._call(features[2], value)
        value = self._call(features[3], value)
        value = self._call(features[4], value)
        value = self._call(features[5], value)
        stage3 = value.permute(0, 3, 1, 2).contiguous()
        value = self._call(features[6], value)
        value = self._call(features[7], value)
        stage4 = value.permute(0, 3, 1, 2).contiguous()
        return stage3, stage4

    def _combine_stages(
        self,
        post: torch.Tensor,
        pre: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The encoder is shared, so one concatenated forward half-batch is
        # mathematically equivalent and avoids duplicate dispatch overhead.
        paired = torch.cat((post, pre), dim=0)
        stages3, stages4 = self.stage_features(paired)
        p3, q3 = stages3.chunk(2, dim=0)
        p4, q4 = stages4.chunk(2, dim=0)
        post3 = self.post_proj3(p3)
        pre3 = self.pre_proj3(q3)
        post4 = F.interpolate(
            self.post_proj4(p4), size=post3.shape[-2:], mode="bilinear", align_corners=False
        )
        pre4 = F.interpolate(
            self.pre_proj4(q4), size=pre3.shape[-2:], mode="bilinear", align_corners=False
        )
        post_feature = self.stage_norm(post3 + post4)
        pre_feature = self.stage_norm(pre3 + pre4)
        return post_feature, pre_feature

    def predict_mask(self, post_feature: torch.Tensor, output_size: int) -> torch.Tensor:
        value = self.mask_head(post_feature)
        return F.interpolate(
            value, size=(output_size, output_size), mode="bilinear", align_corners=False
        )

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        post_feature, pre_feature = self._combine_stages(post, pre)
        difference = post_feature - pre_feature
        fused = self.fusion(
            torch.cat(
                (post_feature, pre_feature, difference, difference.abs()),
                dim=1,
            )
        )
        mask_logits = self.predict_mask(post_feature, int(post.shape[-1]))
        mask_probability = torch.sigmoid(mask_logits)
        weight = F.interpolate(
            mask_probability,
            size=fused.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if valid_mask is not None:
            valid = F.interpolate(
                valid_mask,
                size=fused.shape[-2:],
                mode="nearest",
            )
            weight = weight * valid
            fallback = valid
        else:
            fallback = torch.ones_like(weight)
        fraction = weight.sum(dim=(-2, -1), keepdim=True) / (
            fallback.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        )
        uniform = fallback / fallback.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        normalized_weight = weight / weight.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        pooling_weight = torch.where(
            fraction >= self.minimum_mask_fraction,
            normalized_weight,
            uniform,
        )
        pooled = (fused * pooling_weight).sum(dim=(-2, -1))
        logits = self.classifier(pooled)
        return {
            "logits": logits,
            "mask_logits": mask_logits,
            "mask_probability": mask_probability,
            "mask_fraction": fraction.squeeze(-1).squeeze(-1),
            "pooling_weight": pooling_weight,
        }


def load_b1_encoder(model: MaskDiffSwinT, checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = next(
        (checkpoint[key] for key in ("ema_model", "model") if checkpoint.get(key) is not None),
        checkpoint,
    )
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint in {path}")
    target = model.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []
    with torch.no_grad():
        for source_key, value in state.items():
            if not source_key.startswith("features."):
                skipped.append(source_key)
                continue
            target_key = f"encoder.{source_key}"
            if target_key not in target or target[target_key].shape != value.shape:
                skipped.append(source_key)
                continue
            target[target_key].copy_(value)
            loaded.append(target_key)
    model.load_state_dict(target, strict=True)
    return {
        "checkpoint": str(path),
        "loaded_count": len(loaded),
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:10],
    }


def build_model(config: dict[str, Any]) -> tuple[MaskDiffSwinT, dict[str, Any]]:
    model_config = config["model"]
    preprocessing = config["preprocessing"]
    model = MaskDiffSwinT(
        num_classes=int(model_config.get("num_classes", 4)),
        pretrained=bool(model_config.get("pretrained", True)),
        head_dropout=float(config.get("head_dropout", 0.2)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        mask_dilate_pixels=int(model_config["building_mask_head"].get("dilate_pixels", 2)),
        minimum_mask_fraction=float(config["inference"].get("minimum_mask_fraction", 0.02)),
    )
    report: dict[str, Any] = {
        "pretrained": bool(model_config.get("pretrained", True)),
        "spatial_fusion_stages": model_config.get("spatial_fusion_stages", []),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "normalization": preprocessing["normalization"]["name"],
    }
    warm_start = config.get("warm_start", {})
    if warm_start.get("enabled", False):
        checkpoint_path = (
            Path(config["_config_dir"]) / str(warm_start["checkpoint"])
        ).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"B1 warm-start checkpoint not found: {checkpoint_path}")
        report["warm_start"] = load_b1_encoder(model, checkpoint_path)
    return model, report
