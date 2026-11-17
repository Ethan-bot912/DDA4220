"""Losses used by the optimized B2 training loop."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def class_weights(counts: dict[str, int], names: list[str], power: float) -> torch.Tensor:
    total = float(sum(counts.values()))
    values = [(total / max(float(counts.get(name, 0)), 1.0)) ** float(power) for name in names]
    weights = torch.tensor(values, dtype=torch.float32)
    return weights / weights.mean().clamp_min(1e-8)


def ordinal_targets(labels: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [(labels >= threshold).to(dtype=torch.float32) for threshold in (1, 2, 3)],
        dim=1,
    )


class OptimizedLoss(nn.Module):
    def __init__(
        self,
        *,
        class_weight: torch.Tensor,
        ordinal_weight: float = 0.2,
        damage_weight: float = 0.1,
        distill_weight: float = 0.05,
        focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weight", class_weight.float())
        self.ordinal_weight = float(ordinal_weight)
        self.damage_weight = float(damage_weight)
        self.distill_weight = float(distill_weight)
        self.focal_gamma = float(focal_gamma)

    def _classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, weight=self.class_weight, reduction="none")
        if self.focal_gamma <= 0.0:
            return ce.mean()
        probabilities = torch.softmax(logits.detach(), dim=1)
        focal_factor = (1.0 - probabilities.gather(1, labels[:, None]).squeeze(1)).pow(self.focal_gamma)
        return (ce * focal_factor).mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        hard_negative: torch.Tensor | None = None,
        distill_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = outputs["logits"]
        ce_per_item = F.cross_entropy(logits, labels, weight=self.class_weight, reduction="none")
        if self.focal_gamma > 0.0:
            with torch.no_grad():
                confidence = torch.softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
            ce_per_item = ce_per_item * (1.0 - confidence).pow(self.focal_gamma)
        if hard_negative is not None:
            ce_per_item = ce_per_item * (1.0 + hard_negative.to(ce_per_item.dtype))
        ce_loss = ce_per_item.mean()
        total = ce_loss
        values: dict[str, float] = {"ce": float(ce_loss.detach())}

        ordinal_loss = logits.new_zeros(())
        if "ordinal" in outputs and self.ordinal_weight > 0.0:
            ordinal_loss = F.binary_cross_entropy_with_logits(
                outputs["ordinal"], ordinal_targets(labels)
            )
            total = total + self.ordinal_weight * ordinal_loss
        values["ordinal"] = float(ordinal_loss.detach())

        damage_loss = logits.new_zeros(())
        if "damage" in outputs and self.damage_weight > 0.0:
            damaged = (labels > 0).to(dtype=outputs["damage"].dtype)
            damage_loss = F.binary_cross_entropy_with_logits(outputs["damage"], damaged)
            total = total + self.damage_weight * damage_loss
        values["damage"] = float(damage_loss.detach())

        distill_loss = logits.new_zeros(())
        if self.distill_weight > 0.0 and "post_logits" in outputs:
            if distill_mask is None:
                distill_mask = labels == 0
            if bool(distill_mask.any()):
                student = F.log_softmax(outputs["logits"][distill_mask], dim=1)
                teacher = F.softmax(outputs["post_logits"][distill_mask].detach(), dim=1)
                distill_loss = F.kl_div(student, teacher, reduction="batchmean")
                total = total + self.distill_weight * distill_loss
        values["distill"] = float(distill_loss.detach())
        values["total"] = float(total.detach())
        return total, values


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)
