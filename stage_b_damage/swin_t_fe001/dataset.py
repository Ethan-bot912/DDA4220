"""Lazy FE-001 post-disaster building-crop dataset."""

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
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.split = split
        self.crop_size = int(crop_size)
        self.context = int(context)
        self.return_meta = bool(return_meta)
        self.max_open_files = max(1, int(max_open_files))
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
        side = max(2, int(math.ceil(max(width, height) + 2 * self.context)))
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self._resolve_image(row["post_image"])
        image = self._read_crop(self._open(image_path), self._window(row), self.crop_size)
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

