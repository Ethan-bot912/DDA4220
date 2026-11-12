"""Lazy post-only and paired pre/post building-crop datasets."""

from __future__ import annotations

import csv
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torchvision.transforms.functional as TF
from rasterio.enums import Resampling
from rasterio.windows import Window
from torch.utils.data import Dataset


LABEL_TO_ID = {
    "no_damage": 0,
    "minor": 1,
    "major": 2,
    "destroyed": 3,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
REQUIRED_COLUMNS = {
    "split", "event", "tile_id", "building_index", "damage", "post_image",
    "post_bbox_xmin", "post_bbox_ymin", "post_bbox_xmax", "post_bbox_ymax",
}
PAIRED_REQUIRED_COLUMNS = REQUIRED_COLUMNS | {"pre_image"}


def _normalise_path(value: str) -> Path:
    return Path(value.replace("\\", os.sep))


class XBDPostDataset(Dataset):
    """Read the same post-only building windows as FE-001, on demand."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        project_root: str | Path,
        crop_size: int = 224,
        context: int = 16,
        max_rows: int = 0,
        return_meta: bool = False,
        max_open_files: int = 64,
        augmentation: dict | str | None = None,
        crop_geometry: dict | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.split = split
        self.crop_size = int(crop_size)
        self.context = int(context)
        self.return_meta = bool(return_meta)
        self.max_open_files = max(1, int(max_open_files))
        self.crop_geometry = dict(crop_geometry) if isinstance(crop_geometry, dict) else {}
        if isinstance(augmentation, dict):
            self.augmentation = dict(augmentation) if split == "train" else {}
        else:
            self.augmentation = {"enabled": False}
        self._handles: OrderedDict[str, rasterio.DatasetReader] = OrderedDict()
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest has no header: {self.manifest_path}")
            missing = REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Manifest missing columns: {sorted(missing)}")
            rows = list(reader)
        self.rows = [
            row for row in rows
            if row.get("split") == split and row.get("damage") in LABEL_TO_ID
        ]
        if max_rows:
            self.rows = self.rows[: int(max_rows)]
        if not self.rows:
            raise ValueError(f"No labelled rows found for split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in LABEL_TO_ID}
        for row in self.rows:
            counts[row["damage"]] += 1
        return counts

    def events(self) -> list[str]:
        return sorted({row["event"] for row in self.rows})

    def _resolve_image(self, value: str) -> Path:
        path = _normalise_path(value)
        return path if path.is_absolute() else self.project_root / path

    def _open(self, path: Path) -> rasterio.DatasetReader:
        key = str(path)
        if key in self._handles:
            handle = self._handles.pop(key)
            self._handles[key] = handle
            return handle
        handle = rasterio.open(path)
        self._handles[key] = handle
        while len(self._handles) > self.max_open_files:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def _window(self, row: dict[str, str]) -> Window:
        xmin = float(row["post_bbox_xmin"])
        ymin = float(row["post_bbox_ymin"])
        xmax = float(row["post_bbox_xmax"])
        ymax = float(row["post_bbox_ymax"])
        width = max(xmax - xmin, 1.0)
        height = max(ymax - ymin, 1.0)
        geometry = self.crop_geometry
        mode = str(geometry.get("mode", "fixed"))
        if mode == "relative":
            if self.split == "train":
                margins = [float(value) for value in geometry.get(
                    "train_margin_ratios", [0.5, 0.75, 1.0]
                )]
                if not margins:
                    raise ValueError("train_margin_ratios must not be empty")
                margin_ratio = margins[int(torch.randint(0, len(margins), ()).item())]
                jitter_range = geometry.get(
                    "train_scale_jitter", geometry.get("scale_jitter", [1.0, 1.0])
                )
                jitter_low, jitter_high = (
                    float(jitter_range[0]), float(jitter_range[1])
                )
                if jitter_low <= 0.0 or jitter_high < jitter_low:
                    raise ValueError("train_scale_jitter must be a positive [low, high] range")
                scale_jitter = float(
                    jitter_low + (jitter_high - jitter_low) * torch.rand(()).item()
                )
            else:
                margin_ratio = float(geometry.get("val_margin_ratio", 0.75))
                scale_jitter = 1.0
            if margin_ratio < 0.0:
                raise ValueError("relative crop margin_ratio must be non-negative")
            side = max(
                2,
                int(math.ceil(
                    max(width, height) * (1.0 + 2.0 * margin_ratio) * scale_jitter
                )),
            )
        elif mode == "fixed":
            side = max(2, int(math.ceil(max(width, height) + 2 * self.context)))
        else:
            raise ValueError(f"Unsupported crop_geometry mode: {mode}")
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        return Window(
            math.floor(center_x - side / 2),
            math.floor(center_y - side / 2),
            side,
            side,
        )

    @staticmethod
    def _read_crop(source: rasterio.DatasetReader, window: Window, size: int) -> torch.Tensor:
        array = source.read(
            out_shape=(source.count, size, size),
            window=window,
            boundless=True,
            fill_value=source.nodata if source.nodata is not None else -99,
            resampling=Resampling.bilinear,
        ).astype(np.float32, copy=False)
        if array.shape[0] < 3:
            raise ValueError(f"Expected RGB GeoTIFF, got shape={array.shape}")
        array = array[:3]
        array[array < 0] = 0
        array = np.clip(array, 0, 255) / 255.0
        return torch.from_numpy(np.ascontiguousarray(array))

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        """Apply label-preserving aerial-image augmentation before normalization."""
        if not self.augmentation.get("enabled", False):
            return image

        # A random D4 transform preserves the building damage label while
        # removing the model's dependence on arbitrary north/up orientation.
        if self.augmentation.get("d4", True):
            k = int(torch.randint(0, 4, ()).item())
            image = torch.rot90(image, k=k, dims=(-2, -1))
            if bool(torch.rand(()) < 0.5):
                image = torch.flip(image, dims=(-1,))

        probability = float(self.augmentation.get("color_jitter_probability", 0.0))
        if probability > 0.0 and bool(torch.rand(()) < probability):
            brightness = float(self.augmentation.get("brightness", 0.0))
            contrast = float(self.augmentation.get("contrast", 0.0))
            saturation = float(self.augmentation.get("saturation", 0.0))
            if brightness:
                factor = 1.0 + (2.0 * float(torch.rand(()).item()) - 1.0) * brightness
                image = TF.adjust_brightness(image, factor)
            if contrast:
                factor = 1.0 + (2.0 * float(torch.rand(()).item()) - 1.0) * contrast
                image = TF.adjust_contrast(image, factor)
            if saturation:
                factor = 1.0 + (2.0 * float(torch.rand(()).item()) - 1.0) * saturation
                image = TF.adjust_saturation(image, factor)
        return image.clamp(0.0, 1.0).contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self._resolve_image(row["post_image"])
        image = self._read_crop(self._open(image_path), self._window(row), self.crop_size)
        image = self._augment(image)
        item: dict[str, Any] = {
            "image": image,
            "label": torch.tensor(LABEL_TO_ID[row["damage"]], dtype=torch.long),
        }
        if self.return_meta:
            item["meta"] = {
                "event": row["event"],
                "tile_id": row["tile_id"],
                "building_index": row["building_index"],
                "damage": row["damage"],
            }
        return item

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class XBDPairedDataset(XBDPostDataset):
    """Read paired pre/post crops using one post-bbox-centered window.

    The first B2 control uses no augmentation, but the paired augmentation
    implementation deliberately applies identical spatial and photometric
    parameters to both timestamps so future experiments cannot accidentally
    create artificial change between the two views.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        missing = [
            index for index, row in enumerate(self.rows)
            if not row.get("pre_image")
        ]
        if missing:
            raise ValueError(
                "Paired manifest rows must contain pre_image; "
                f"missing rows include indices {missing[:3]}"
            )

    def _augment_pair(
        self, pre: torch.Tensor, post: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.augmentation.get("enabled", False):
            return pre, post

        # Draw every random parameter once and apply it to both timestamps.
        if self.augmentation.get("d4", True):
            k = int(torch.randint(0, 4, ()).item())
            do_flip = bool(torch.rand(()) < 0.5)

            def spatial(image: torch.Tensor) -> torch.Tensor:
                image = torch.rot90(image, k=k, dims=(-2, -1))
                return torch.flip(image, dims=(-1,)) if do_flip else image

            pre, post = spatial(pre), spatial(post)

        probability = float(self.augmentation.get("color_jitter_probability", 0.0))
        if probability > 0.0 and bool(torch.rand(()) < probability):
            brightness = float(self.augmentation.get("brightness", 0.0))
            contrast = float(self.augmentation.get("contrast", 0.0))
            saturation = float(self.augmentation.get("saturation", 0.0))
            brightness_factor = (
                1.0 + (2.0 * float(torch.rand(()).item()) - 1.0) * brightness
                if brightness else 1.0
            )
            contrast_factor = (
                1.0 + (2.0 * float(torch.rand(()).item()) - 1.0) * contrast
                if contrast else 1.0
            )
            saturation_factor = (
                1.0 + (2.0 * float(torch.rand(()).item()) - 1.0) * saturation
                if saturation else 1.0
            )

            def photometric(image: torch.Tensor) -> torch.Tensor:
                image = TF.adjust_brightness(image, brightness_factor)
                image = TF.adjust_contrast(image, contrast_factor)
                return TF.adjust_saturation(image, saturation_factor).clamp(0.0, 1.0)

            pre, post = photometric(pre), photometric(post)

        return pre.contiguous(), post.contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        window = self._window(row)
        pre_path = self._resolve_image(row["pre_image"])
        post_path = self._resolve_image(row["post_image"])
        pre = self._read_crop(self._open(pre_path), window, self.crop_size)
        post = self._read_crop(self._open(post_path), window, self.crop_size)
        pre, post = self._augment_pair(pre, post)
        item: dict[str, Any] = {
            "pre": pre,
            "post": post,
            "image": torch.cat((pre, post), dim=0),
            "label": torch.tensor(LABEL_TO_ID[row["damage"]], dtype=torch.long),
        }
        if self.return_meta:
            item["meta"] = {
                "event": row["event"],
                "tile_id": row["tile_id"],
                "building_index": row["building_index"],
                "damage": row["damage"],
            }
        return item

