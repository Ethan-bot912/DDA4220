"""Paired pre/post dataset with lightweight alignment and radiometric correction."""

from __future__ import annotations

import csv
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

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
    "split", "event", "tile_id", "building_index", "damage", "pre_image",
    "post_image", "post_bbox_xmin", "post_bbox_ymin", "post_bbox_xmax",
    "post_bbox_ymax",
}


def _normalise_path(value: str) -> Path:
    return Path(value.replace("\\", os.sep))


def _downsample(array: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    height = (array.shape[-2] // factor) * factor
    width = (array.shape[-1] // factor) * factor
    if factor == 1:
        return array
    if height < factor or width < factor:
        return array
    cropped = array[..., :height, :width]
    return cropped.reshape(*cropped.shape[:-2], height // factor, factor, width // factor, factor).mean(
        axis=(-3, -1)
    )


def estimate_phase_translation(
    pre: torch.Tensor,
    post: torch.Tensor,
    *,
    max_shift: int = 6,
    downsample: int = 4,
) -> tuple[int, int, float]:
    """Estimate an integer translation of ``pre`` toward ``post``.

    The phase-correlation estimate is intentionally conservative and returns a
    bounded integer shift.  It has no OpenCV dependency, which keeps the pilot
    runnable in the existing project environment.
    """
    pre_gray = pre.detach().float().mean(dim=0).cpu().numpy()
    post_gray = post.detach().float().mean(dim=0).cpu().numpy()
    pre_gray = _downsample(pre_gray, downsample)
    post_gray = _downsample(post_gray, downsample)
    pre_gray = pre_gray - float(pre_gray.mean())
    post_gray = post_gray - float(post_gray.mean())
    spectrum = np.fft.fft2(post_gray) * np.conjugate(np.fft.fft2(pre_gray))
    spectrum /= np.maximum(np.abs(spectrum), 1e-8)
    correlation = np.fft.ifft2(spectrum).real
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    height, width = correlation.shape
    shift_y = int(peak[0]) if peak[0] <= height // 2 else int(peak[0] - height)
    shift_x = int(peak[1]) if peak[1] <= width // 2 else int(peak[1] - width)
    shift_x = int(np.clip(shift_x * max(1, downsample), -max_shift, max_shift))
    shift_y = int(np.clip(shift_y * max(1, downsample), -max_shift, max_shift))
    peak_value = float(correlation[peak])
    baseline = float(np.mean(np.abs(correlation))) + 1e-8
    score = float(np.clip(peak_value / baseline / 10.0, 0.0, 1.0))
    return shift_x, shift_y, score


def warp_integer(image: torch.Tensor, shift_x: int, shift_y: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate an image without wrap-around and return a valid-pixel mask."""
    _, height, width = image.shape
    output = torch.zeros_like(image)
    mask = torch.zeros((1, height, width), dtype=image.dtype)
    src_y0, src_y1 = max(0, -shift_y), min(height, height - shift_y)
    src_x0, src_x1 = max(0, -shift_x), min(width, width - shift_x)
    dst_y0, dst_y1 = max(0, shift_y), min(height, height + shift_y)
    dst_x0, dst_x1 = max(0, shift_x), min(width, width + shift_x)
    if src_y1 > src_y0 and src_x1 > src_x0:
        output[:, dst_y0:dst_y1, dst_x0:dst_x1] = image[:, src_y0:src_y1, src_x0:src_x1]
        mask[:, dst_y0:dst_y1, dst_x0:dst_x1] = 1.0
    return output, mask


def robust_pair_normalize(
    pre: torch.Tensor,
    post: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    enabled: bool,
    clip: float = 3.0,
    min_scale: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Match robust pre statistics to post statistics, preserving post values."""
    if not enabled:
        return pre, post, 0.0
    mask = valid_mask[0] > 0.5
    if int(mask.sum()) < 16:
        return pre, post, 0.0
    pre_out = pre.clone()
    radiometric_shift = 0.0
    for channel in range(3):
        pre_values = pre[channel][mask]
        post_values = post[channel][mask]
        pre_median = torch.median(pre_values)
        post_median = torch.median(post_values)
        pre_scale = torch.quantile((pre_values - pre_median).abs(), 0.75).clamp_min(min_scale)
        post_scale = torch.quantile((post_values - post_median).abs(), 0.75).clamp_min(min_scale)
        z_score = ((pre[channel] - pre_median) / pre_scale).clamp(-float(clip), float(clip))
        pre_out[channel] = (z_score * post_scale + post_median).clamp(0.0, 1.0)
        radiometric_shift += float((pre_median - post_median).abs())
    return pre_out.clamp(0.0, 1.0), post, radiometric_shift / 3.0


def _apply_independent_photometric(
    image: torch.Tensor,
    *,
    brightness: float,
    contrast: float,
    gamma: float,
) -> torch.Tensor:
    if brightness:
        factor = 1.0 + (2.0 * float(torch.rand(()).item()) - 1.0) * brightness
        image = TF.adjust_brightness(image, factor)
    if contrast:
        factor = 1.0 + (2.0 * float(torch.rand(()).item()) - 1.0) * contrast
        image = TF.adjust_contrast(image, factor)
    if gamma:
        gamma_value = 1.0 + (2.0 * float(torch.rand(()).item()) - 1.0) * gamma
        image = image.clamp(0.0, 1.0).pow(1.0 / max(gamma_value, 1e-3))
    return image.clamp(0.0, 1.0)


class OptimizedPairedDataset(Dataset):
    """Lazy paired dataset used by the optimized Swin-T B2 pipeline."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        project_root: str | Path,
        crop_size: int = 224,
        context: int = 16,
        alignment: Mapping[str, Any] | None = None,
        radiometric: Mapping[str, Any] | None = None,
        augmentation: Mapping[str, Any] | None = None,
        hard_negative: Mapping[str, Any] | None = None,
        max_rows: int = 0,
        max_open_files: int = 64,
        hard_negative_keys: set[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.split = split
        self.crop_size = int(crop_size)
        self.context = int(context)
        self.alignment = dict(alignment or {})
        self.radiometric = dict(radiometric or {})
        self.augmentation = dict(augmentation or {}) if split == "train" else {}
        self.hard_negative = dict(hard_negative or {}) if split == "train" else {}
        self.max_open_files = max(1, int(max_open_files))
        self.hard_negative_keys = hard_negative_keys or set()
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

    def row_key(self, row: Mapping[str, str]) -> str:
        return f"{row['event']}|{row['tile_id']}|{row['building_index']}"

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

    @staticmethod
    def _window_from_bbox(xmin: float, ymin: float, xmax: float, ymax: float, context: int) -> Window:
        side = max(2, int(math.ceil(max(xmax - xmin, ymax - ymin, 1.0) + 2 * context)))
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        return Window(math.floor(center_x - side / 2), math.floor(center_y - side / 2), side, side)

    def _windows(
        self,
        row: Mapping[str, str],
        pre_source: rasterio.DatasetReader,
        post_source: rasterio.DatasetReader,
    ) -> tuple[Window, Window]:
        xmin = float(row["post_bbox_xmin"])
        ymin = float(row["post_bbox_ymin"])
        xmax = float(row["post_bbox_xmax"])
        ymax = float(row["post_bbox_ymax"])
        post_window = self._window_from_bbox(xmin, ymin, xmax, ymax, self.context)

        explicit_pre = all(
            key in row and row[key] not in (None, "")
            for key in (
                "pre_bbox_xmin", "pre_bbox_ymin", "pre_bbox_xmax", "pre_bbox_ymax"
            )
        )
        if explicit_pre:
            pre_window = self._window_from_bbox(
                float(row["pre_bbox_xmin"]),
                float(row["pre_bbox_ymin"]),
                float(row["pre_bbox_xmax"]),
                float(row["pre_bbox_ymax"]),
                self.context,
            )
            return post_window, pre_window

        # Convert the post-pixel bbox through world coordinates when both
        # rasters expose compatible transforms. This avoids assuming that the
        # same pixel indices represent the same ground location.
        if post_source.crs is not None and pre_source.crs is not None and post_source.crs != pre_source.crs:
            return post_window, post_window
        corners = [
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymin),
            (xmax, ymax),
        ]
        world = [post_source.transform * corner for corner in corners]
        pre_pixels = [(~pre_source.transform) * point for point in world]
        pre_x = [point[0] for point in pre_pixels]
        pre_y = [point[1] for point in pre_pixels]
        pre_window = self._window_from_bbox(
            min(pre_x), min(pre_y), max(pre_x), max(pre_y), self.context
        )
        return post_window, pre_window

    @staticmethod
    def _read_crop(source: rasterio.DatasetReader, window: Window, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        array = source.read(
            out_shape=(source.count, size, size),
            window=window,
            boundless=True,
            fill_value=source.nodata if source.nodata is not None else -99,
            resampling=Resampling.bilinear,
        ).astype(np.float32, copy=False)
        mask = source.read_masks(
            1,
            out_shape=(size, size),
            window=window,
            boundless=True,
            resampling=Resampling.nearest,
        ).astype(np.float32, copy=False)
        if array.shape[0] < 3:
            raise ValueError(f"Expected RGB GeoTIFF, got shape={array.shape}")
        array = np.clip(array[:3], 0.0, 255.0) / 255.0
        image = torch.from_numpy(np.ascontiguousarray(array))
        valid = torch.from_numpy(np.ascontiguousarray(mask[None] > 0)).float()
        return image, valid

    def _augment_pair(
        self, pre: torch.Tensor, post: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        options = self.augmentation
        if not options.get("enabled", False):
            return pre, post, valid
        if options.get("d4", True):
            k = int(torch.randint(0, 4, ()).item())
            flip = bool(torch.rand(()) < 0.5)

            def spatial(image: torch.Tensor) -> torch.Tensor:
                image = torch.rot90(image, k=k, dims=(-2, -1))
                return torch.flip(image, dims=(-1,)) if flip else image

            pre, post, valid = spatial(pre), spatial(post), spatial(valid)
        if options.get("independent_photometric", True):
            probability = float(options.get("photometric_probability", 0.0))
            if bool(torch.rand(()) < probability):
                kwargs = {
                    "brightness": float(options.get("brightness", 0.0)),
                    "contrast": float(options.get("contrast", 0.0)),
                    "gamma": float(options.get("gamma", 0.0)),
                }
                pre = _apply_independent_photometric(pre, **kwargs)
                post = _apply_independent_photometric(post, **kwargs)
        probability = float(options.get("misalignment_probability", 0.0))
        if bool(torch.rand(()) < probability):
            amount = int(options.get("misalignment_pixels", 0))
            dx = int(torch.randint(-amount, amount + 1, ()).item()) if amount else 0
            dy = int(torch.randint(-amount, amount + 1, ()).item()) if amount else 0
            pre, extra_mask = warp_integer(pre, dx, dy)
            valid = valid * extra_mask
        if bool(torch.rand(()) < float(options.get("pre_dropout_probability", 0.0))):
            pre = post.clone()
        return pre.contiguous(), post.contiguous(), valid.contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        pre_source = self._open(self._resolve_image(row["pre_image"]))
        post_source = self._open(self._resolve_image(row["post_image"]))
        post_window, pre_window = self._windows(row, pre_source, post_source)
        pre, pre_valid = self._read_crop(pre_source, pre_window, self.crop_size)
        post, post_valid = self._read_crop(post_source, post_window, self.crop_size)
        valid = pre_valid * post_valid
        shift_x = shift_y = 0
        registration_score = 1.0
        if bool(self.alignment.get("enabled", False)):
            shift_x, shift_y, registration_score = estimate_phase_translation(
                pre,
                post,
                max_shift=int(self.alignment.get("max_shift", 6)),
                downsample=int(self.alignment.get("downsample", 4)),
            )
            aligned, alignment_mask = warp_integer(pre, shift_x, shift_y)
            pre = aligned
            valid = valid * alignment_mask
        pre, post, radiometric_shift = robust_pair_normalize(
            pre,
            post,
            valid,
            enabled=bool(self.radiometric.get("enabled", False)),
            clip=float(self.radiometric.get("clip", 3.0)),
            min_scale=float(self.radiometric.get("min_scale", 0.03)),
        )
        pre, post, valid = self._augment_pair(pre, post, valid)
        change = (pre - post).abs()
        quality = torch.tensor(
            [
                float(registration_score),
                float(valid.mean()),
                float(math.sqrt(shift_x * shift_x + shift_y * shift_y) / max(1, int(self.alignment.get("max_shift", 6)))),
                float(radiometric_shift),
            ],
            dtype=torch.float32,
        )
        key = self.row_key(row)
        label_id = LABEL_TO_ID[row["damage"]]
        heuristic_hard = False
        if bool(self.hard_negative.get("enabled", False)):
            high_change = float(self.hard_negative.get("high_change_no_damage", 0.12))
            low_change = float(self.hard_negative.get("low_change_damage", 0.03))
            low_registration = float(self.hard_negative.get("low_registration", 0.10))
            heuristic_hard = (
                (label_id == 0 and float(change.mean()) >= high_change)
                or (label_id > 0 and float(change.mean()) <= low_change)
                or registration_score <= low_registration
            )
        return {
            "pre": pre,
            "post": post,
            "valid_mask": valid,
            "quality": quality,
            "change_score": change.mean().reshape(1),
            "hard_negative": torch.tensor(float(key in self.hard_negative_keys or heuristic_hard)),
            "label": torch.tensor(label_id, dtype=torch.long),
            "meta": {
                "event": row["event"],
                "tile_id": row["tile_id"],
                "building_index": row["building_index"],
                "damage": row["damage"],
                "key": key,
            },
        }

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
