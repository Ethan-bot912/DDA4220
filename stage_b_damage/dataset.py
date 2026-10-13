"""Lazy paired xBD building-crop dataset for Stage B experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.windows import Window
from torch.utils.data import Dataset


LABEL_TO_ID = {"no_damage": 0, "minor": 1, "major": 2, "destroyed": 3}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


class XBDPairedDataset(Dataset):
    """Read fixed-size paired crops on demand from a manifest.

    The manifest stores pixel-space building boxes from the GeoTIFF label
    files. One window is used for both pre- and post-disaster images, which
    preserves the spatial pairing required by the paired B2 classifier. The
    returned tensors are float32 in ``[0, 1]`` with shape
    ``(3, crop_size, crop_size)``.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        project_root: str | Path | None = None,
        crop_size: int = 128,
        context: int = 16,
        return_meta: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path.cwd().resolve()
        )
        self.crop_size = int(crop_size)
        self.context = int(context)
        self.return_meta = return_meta
        self._handles: dict[str, rasterio.DatasetReader] = {}

        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
        self.rows = [
            row
            for row in all_rows
            if row.get("split") == split and row.get("damage") in LABEL_TO_ID
        ]
        if not self.rows:
            raise ValueError(f"No labeled rows found for split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def _open(self, path: Path) -> rasterio.DatasetReader:
        key = str(path)
        if key not in self._handles:
            self._handles[key] = rasterio.open(path)
        return self._handles[key]

    def _window(self, row: dict[str, str]) -> Window:
        xmin = float(row["post_bbox_xmin"])
        ymin = float(row["post_bbox_ymin"])
        xmax = float(row["post_bbox_xmax"])
        ymax = float(row["post_bbox_ymax"])
        width = max(xmax - xmin, 1.0)
        height = max(ymax - ymin, 1.0)
        side = int(math.ceil(max(width, height) + 2 * self.context))
        side = max(side, 2)
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
        # xBD GeoTIFF pixels are 8-bit-like values stored as int16 and use
        # -99 for nodata. Clip after replacing nodata so padding is black.
        array[array < 0] = 0
        array = np.clip(array, 0, 255) / 255.0
        return torch.from_numpy(np.ascontiguousarray(array))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        window = self._window(row)
        pre_path = self._path(row["pre_image"])
        post_path = self._path(row["post_image"])
        pre = self._read_crop(self._open(pre_path), window, self.crop_size)
        post = self._read_crop(self._open(post_path), window, self.crop_size)
        item: dict[str, Any] = {
            "pre": pre,
            "post": post,
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
        state["_handles"] = {}
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
