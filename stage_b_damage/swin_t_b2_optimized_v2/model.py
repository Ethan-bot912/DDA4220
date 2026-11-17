"""Dual-stream Swin-T model for the optimized B2 experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import Swin_T_Weights, swin_t


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_rgb(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (image - mean) / std


class DualStreamSwinT(nn.Module):
    """Shared-weight pre/post Swin-T with difference fusion and a B1 residual gate."""

    def __init__(
        self,
        *,
        num_classes: int = 4,
        pretrained: bool = True,
        head_dropout: float = 0.2,
        gate_enabled: bool = True,
        gate_initial_bias: float = -3.0,
        ordinal: bool = True,
        binary_damage: bool = True,
    ) -> None:
        super().__init__()
        weights: Any = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = swin_t(weights=weights)
        feature_dim = int(self.encoder.head.in_features)
        self.encoder.head = nn.Identity()
        self.feature_dim = feature_dim
        self.gate_enabled = bool(gate_enabled)

        self.quality_projection = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 64),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(feature_dim * 4),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.GELU(),
            nn.Dropout(float(head_dropout)),
        )
        self.post_head = nn.Linear(feature_dim, num_classes)
        self.delta_head = nn.Linear(feature_dim + 64, num_classes)
        self.classifier = nn.Linear(feature_dim + 64, num_classes)
        self.ordinal_head = nn.Linear(feature_dim + 64, 3) if ordinal else None
        self.damage_head = nn.Linear(feature_dim + 64, 1) if binary_damage else None
        if self.gate_enabled:
            self.gate_head = nn.Sequential(
                nn.LayerNorm(feature_dim + 64),
                nn.Linear(feature_dim + 64, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
            self.gate_bias = nn.Parameter(torch.tensor(float(gate_initial_bias)))
        else:
            self.gate_head = None
            self.register_parameter("gate_bias", None)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        x = self.encoder.features(image)
        x = self.encoder.norm(x)
        x = self.encoder.permute(x)
        x = self.encoder.avgpool(x)
        return torch.flatten(x, 1)

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        quality: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if valid_mask is not None:
            pre = pre * valid_mask
            post = post * valid_mask
        f_pre = self.forward_features(pre)
        f_post = self.forward_features(post)
        fused_raw = torch.cat(
            (f_post, f_pre, (f_post - f_pre).abs(), f_post * f_pre), dim=1
        )
        fused = self.fusion(fused_raw)
        if quality is None:
            quality = fused.new_zeros((fused.shape[0], 4))
        quality = quality.to(dtype=fused.dtype)
        quality_features = self.quality_projection(quality)
        joint = torch.cat((fused, quality_features), dim=1)

        post_logits = self.post_head(f_post)
        delta_logits = self.delta_head(joint)
        if self.gate_enabled and self.gate_head is not None and self.gate_bias is not None:
            gate = torch.sigmoid(self.gate_head(joint) + self.gate_bias)
            logits = post_logits + gate * delta_logits
        else:
            gate = fused.new_ones((fused.shape[0], 1))
            logits = self.classifier(joint)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "post_logits": post_logits,
            "delta_logits": delta_logits,
            "gate": gate,
            "features": joint,
        }
        if self.ordinal_head is not None:
            output["ordinal"] = self.ordinal_head(joint)
        if self.damage_head is not None:
            output["damage"] = self.damage_head(joint).squeeze(1)
        return output


def load_b1_warm_start(model: DualStreamSwinT, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a compatible B1 encoder and classifier warm-start.

    The optimized model keeps a separate ``post_head`` so that the residual
    gate can start from the B1 post-image classifier and learn a delta on top
    of it.  Older versions only loaded the encoder and silently left
    ``post_head`` random, which made the advertised B1 fallback ineffective.
    The B1 four-class ``head.weight``/``head.bias`` are shape-compatible with
    ``post_head`` and are therefore copied when present.  Any incompatible
    head is safely skipped and reported.
    """
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema_model") or checkpoint.get("model") or checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint state in {path}")
    target = model.state_dict()
    loaded: list[str] = []
    encoder_loaded: list[str] = []
    head_loaded: list[str] = []
    skipped: list[str] = []
    with torch.no_grad():
        for source_key, value in state.items():
            # B1 checkpoints in this project use ``head.*``.  Also accept
            # common wrappers (``backbone.head.*``/``model.head.*``) without
            # accidentally mapping an encoder Identity head.
            normalized_key = source_key
            for prefix in ("module.", "model.", "backbone."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix):]
            if normalized_key in {"head.weight", "head.bias"}:
                candidates = [f"post_head.{normalized_key.split('.', 1)[1]}"]
            else:
                candidates = [
                    f"encoder.{source_key}",
                    source_key.replace("backbone.", "encoder.", 1),
                    source_key.replace("model.", "encoder.", 1),
                    source_key.replace("module.", "encoder.", 1),
                ]
            target_key = next((candidate for candidate in candidates if candidate in target), None)
            if target_key is None or target[target_key].shape != value.shape:
                skipped.append(source_key)
                continue
            target[target_key].copy_(value)
            loaded.append(target_key)
            if target_key.startswith("encoder."):
                encoder_loaded.append(target_key)
            elif target_key.startswith("post_head."):
                head_loaded.append(target_key)
    model.load_state_dict(target, strict=False)
    return {
        "checkpoint": str(path),
        "loaded_count": len(loaded),
        "encoder_loaded_count": len(encoder_loaded),
        "encoder_loaded_examples": encoder_loaded[:10],
        "head_loaded": bool(head_loaded),
        "head_loaded_keys": head_loaded,
        "loaded_examples": loaded[:10],
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:10],
    }


def build_model(config: dict[str, Any]) -> tuple[DualStreamSwinT, dict[str, Any]]:
    gate = config.get("gate", {})
    loss = config.get("loss", {})
    model = DualStreamSwinT(
        num_classes=int(config.get("num_classes", 4)),
        pretrained=bool(config.get("pretrained", True)),
        head_dropout=float(config.get("head_dropout", 0.2)),
        gate_enabled=bool(gate.get("enabled", True)),
        gate_initial_bias=float(gate.get("initial_bias", -3.0)),
        ordinal=float(loss.get("ordinal_weight", 0.0)) > 0.0,
        binary_damage=float(loss.get("damage_weight", 0.0)) > 0.0,
    )
    report: dict[str, Any] = {"pretrained": bool(config.get("pretrained", True))}
    checkpoint = str(config.get("b1_checkpoint", ""))
    if checkpoint:
        path = Path(config["_config_dir"]) / checkpoint
        if path.is_file():
            report["warm_start"] = load_b1_warm_start(model, path)
        else:
            report["warm_start"] = {"checkpoint": str(path), "status": "missing"}
    return model, report
