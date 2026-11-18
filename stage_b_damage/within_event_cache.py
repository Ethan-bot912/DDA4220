"""Versioned, lossless train/val mmap cache (PERF-002 design, stricter validation).

Historical dataset/trainer sources are deliberately unchanged. Test images are
never opened by this module, including source fingerprint verification.
"""
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from stage_b_damage.dataset import LABEL_TO_ID, XBDPairedDataset
from tools.train_expanded import sha256


def metadata_identity(config, root):
    result = {}
    for key in ("manifest", "split_map", "data_lock"):
        actual = sha256(root / config[key])
        if actual != config[key + "_sha256"]:
            raise ValueError(f"Frozen {key} changed")
        result[key + "_sha256"] = actual
    return result


def allowed_sources(config, root):
    with (root / config["manifest"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    allowed, forbidden = set(), set()
    for row in rows:
        paths = {row[k].replace("\\", "/") for k in ("pre_image", "post_image")}
        if row["split"] in ("train", "val"):
            if row["event"] in config["sealed_test_events"]:
                raise ValueError("Sealed event in development data")
            allowed.update(paths)
        else:
            forbidden.update(paths)
    if allowed & forbidden:
        raise ValueError("Development and sealed splits share an image")
    return sorted(allowed)


def source_identity(config, root):
    locked = json.loads((root / config["data_lock"]).read_text(encoding="utf-8"))
    entries = {r["path"].replace("\\", "/"): r["sha256"] for r in locked["files"]}
    result = {}
    for name in allowed_sources(config, root):
        actual = sha256(root / name)
        if entries.get(name) != actual:
            raise ValueError(f"Source differs from historical data lock: {name}")
        result[name] = actual
    return result


def cache_code():
    root = Path(__file__).resolve().parents[1]
    return {name: sha256(root / name) for name in
            ("stage_b_damage/dataset.py", "stage_b_damage/within_event_cache.py", "tools/build_within_event_cache.py")}


def cache_versions():
    return {"numpy": str(np.__version__), "rasterio": str(rasterio.__version__),
            "gdal": str(rasterio.__gdal_version__), "torch": str(torch.__version__)}


def verify_cache(config, root):
    identity = metadata_identity(config, root)
    directory = root / config["cache_dir"]
    meta = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    expected = {**identity, "crop_size": config["crop_size"], "context": config["context"],
                "dtype": config["cache_dtype"], "cache_code": cache_code(), "versions": cache_versions(), "format": 1}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Cache {key} mismatch")
    if meta["sources"] != source_identity(config, root):
        raise ValueError("Cache source identity mismatch")
    names = {f"{split}_{phase}.npy" for split in ("train", "val") for phase in ("pre", "post")}
    if set(meta["arrays"]) != names:
        raise ValueError("Cache must contain exactly train/val arrays")
    for name, digest in meta["arrays"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Cache checksum mismatch: {name}")
        split = name.split("_")[0]
        array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        shape = (config[f"expected_{split}_rows"], 3, config["crop_size"], config["crop_size"])
        if array.shape != shape or str(array.dtype) != config["cache_dtype"]:
            raise ValueError(f"Cache shape/dtype mismatch: {name}")
        del array
    return sha256(directory / "complete.json")


class CachedDataset(XBDPairedDataset):
    """Must be constructed after verify_cache at the process entry point."""
    def __init__(self, manifest_path, split, *, cache_dir, **kwargs):
        if split not in ("train", "val"):
            raise ValueError("Only train/val caching is permitted")
        super().__init__(manifest_path, split, **kwargs)
        self.cache_dir, self.split = Path(cache_dir), split
        self._arrays = {}

    def __getitem__(self, index):
        if not self._arrays:
            self._arrays = {key: np.load(self.cache_dir / f"{self.split}_{key}.npy", mmap_mode="r", allow_pickle=False)
                            for key in ("pre", "post")}
        row = self.rows[index]
        item = {"label": torch.tensor(LABEL_TO_ID[row["damage"]], dtype=torch.long)}
        for key, array in self._arrays.items():
            value = np.array(array[index], dtype=np.float32, copy=True)
            if array.dtype == np.uint8:
                value /= np.float32(255)
            item[key] = torch.from_numpy(value)
        if self.return_meta:
            item["meta"] = {k: row[k] for k in ("event", "tile_id", "building_index", "damage")}
        return item

    def __getstate__(self):
        state = super().__getstate__()
        state["_arrays"] = {}
        return state

    def close(self):
        self._arrays = {}
        super().close()
