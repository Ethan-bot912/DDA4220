"""Lazy paired xBD building-crop dataset for Stage B experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import OrderedDict
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
        max_open_files: int = 64,
        cache_dir: str | Path | None = None,
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
        self.max_open_files = int(max_open_files)
        if self.max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self._handles: "OrderedDict[str, rasterio.DatasetReader]" = OrderedDict()
        self._cache_pre: np.ndarray | None = None
        self._cache_post: np.ndarray | None = None
        self._cache_partial = False
        self._cache_limit = 0
        self._cache_dir: Path | None = None

        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle)
                         if row.get("split") == split and row.get("damage") in LABEL_TO_ID]
        if not self.rows:
            raise ValueError(f"No labeled rows found for split={split!r}")

        if cache_dir is not None:
            self._open_cache(Path(cache_dir), split)

    def __getstate__(self) -> dict:
        """Drop live raster handles and mmaps so DataLoader workers can spawn.

        ``rasterio`` dataset readers and ``np.memmap`` objects cannot be pickled
        on Windows (spawn), and pickling an mmap copies the whole cache into the
        state (gigabytes per worker). Store only the small scalar metadata so
        each worker reopens the memmaps lazily.
        """
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_cache_pre"] = None
        state["_cache_post"] = None
        state["cache_meta"] = dict(getattr(self, "_cache_meta_scalars", {}))
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._cache_meta_scalars = dict(getattr(self, "cache_meta", {}))
        if getattr(self, "_cache_dir", None) is not None:
            self._attach_cache_arrays()

    def _attach_cache_arrays(self) -> None:
        meta = self._cache_meta_scalars
        self._cache_pre = np.load(self._cache_dir / meta["pre_file"], mmap_mode="r")
        self._cache_post = np.load(self._cache_dir / meta["post_file"], mmap_mode="r")

    def _open_cache(self, cache_dir: Path, split: str) -> None:
        """Attach a pre-built crop cache, refusing anything inconsistent.

        The cache is only valid for the exact manifest bytes and crop geometry
        it was built from, so every mismatch is fatal rather than silently
        falling back to on-demand reads.
        """
        meta_path = cache_dir / f"{split}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No crop cache at {meta_path}; run tools/build_crop_cache.py first"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checks = {
            "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
            "crop_size": self.crop_size,
            "context": self.context,
        }
        for key, expected in checks.items():
            if meta.get(key) != expected:
                raise ValueError(
                    f"Crop cache mismatch for {key!r}: cache has {meta.get(key)!r}, "
                    f"expected {expected!r}. Rebuild with tools/build_crop_cache.py"
                )
        if meta.get("rows") != len(self.rows):
            if meta.get("row_limit"):
                self._cache_partial = True
                self._cache_limit = int(meta["rows"])
            else:
                raise ValueError(
                    f"Crop cache covers {meta.get('rows')} rows but split {split!r} has "
                    f"{len(self.rows)}. Rebuild with tools/build_crop_cache.py"
                )
        self._cache_pre = np.load(cache_dir / meta["pre_file"], mmap_mode="r")
        self._cache_post = np.load(cache_dir / meta["post_file"], mmap_mode="r")
        self._cache_dir = cache_dir
        self.cache_meta = meta
        self._cache_meta_scalars = {
            key: value
            for key, value in meta.items()
            if key in {"pre_file", "post_file", "dtype", "crop_size", "context", "rows", "shape"}
        }

    def __len__(self) -> int:
        return len(self.rows)
    def _path(self, value: str) -> Path:
        # Preserve the original manifest bytes/hash while accepting its Windows
        # relative separators on Linux and macOS as well.
        path = Path(value.replace("\\", "/"))
        return path if path.is_absolute() else self.project_root / path

    def _open(self, path: Path) -> rasterio.DatasetReader:
        key = str(path)
        if key not in self._handles:
            if len(self._handles) >= self.max_open_files:
                _, oldest = self._handles.popitem(last=False)
                oldest.close()
            self._handles[key] = rasterio.open(path)
        self._handles.move_to_end(key)
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
        use_cache = self._cache_pre is not None and (
            not self._cache_partial or index < self._cache_limit
        )
        if use_cache:
            pre = torch.from_numpy(
                np.asarray(self._cache_pre[index], dtype=np.float32) / np.float32(255.0)
                if self._cache_pre.dtype == np.uint8
                else np.array(self._cache_pre[index], dtype=np.float32)
            )
            post = torch.from_numpy(
                np.asarray(self._cache_post[index], dtype=np.float32) / np.float32(255.0)
                if self._cache_post.dtype == np.uint8
                else np.array(self._cache_post[index], dtype=np.float32)
            )
        else:
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

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
