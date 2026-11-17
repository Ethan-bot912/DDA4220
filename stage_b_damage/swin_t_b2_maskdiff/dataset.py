"""Paired dataset that adds a target-building polygon mask to the optimized B2 data."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFilter
from rasterio.windows import Window

from stage_b_damage.swin_t_b2_optimized_v2.dataset import OptimizedPairedDataset


POINT_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


def polygon_points(value: str) -> list[tuple[float, float]]:
    """Return pixel-coordinate vertices from a stored xBD WKT polygon."""
    if not value:
        return []
    payload = json.loads(value)
    wkt = payload.get("wkt", "") if isinstance(payload, dict) else ""
    return [(float(x), float(y)) for x, y in POINT_RE.findall(wkt)]


def rasterize_target_mask(
    geometry_json: str,
    window: Window,
    size: int,
    *,
    dilate_pixels: int = 0,
) -> torch.Tensor:
    """Rasterize the target building polygon in the same resized crop as the image."""
    points = polygon_points(geometry_json)
    if not points:
        raise ValueError("Missing or invalid post_geometry_json building polygon")
    if window.width <= 0 or window.height <= 0:
        raise ValueError(f"Invalid crop window: {window}")
    transformed = [
        (
            (x - window.col_off) * size / window.width,
            (y - window.row_off) * size / window.height,
        )
        for x, y in points
    ]
    image = Image.new("L", (int(size), int(size)), 0)
    ImageDraw.Draw(image).polygon(transformed, fill=1)
    if dilate_pixels > 0:
        image = image.filter(ImageFilter.MaxFilter(2 * int(dilate_pixels) + 1))
    array = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return array.reshape(1, int(size), int(size)).float()


class MaskDiffDataset(OptimizedPairedDataset):
    """Optimized paired crop with post-building polygon mask supervision."""

    def __init__(
        self,
        *args: Any,
        mask_dilate_pixels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mask_dilate_pixels = int(mask_dilate_pixels)
        if self.augmentation.get("enabled", False):
            raise ValueError(
                "The primary MaskDiff design has augmentation disabled; "
                "implement paired mask augmentation before enabling it"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        row = self.rows[index]
        pre_source = self._open(self._resolve_image(row["pre_image"]))
        post_source = self._open(self._resolve_image(row["post_image"]))
        _, post_window = self._windows(row, pre_source, post_source)
        building_mask = rasterize_target_mask(
            row.get("post_geometry_json", ""),
            post_window,
            self.crop_size,
            dilate_pixels=self.mask_dilate_pixels,
        )
        item["building_mask"] = building_mask
        return item
