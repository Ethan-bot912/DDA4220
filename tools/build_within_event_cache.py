"""Build both development splits, never test; explicit dtype, atomic completion."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from stage_b_damage.dataset import XBDPairedDataset
from stage_b_damage.within_event_cache import metadata_identity, source_identity, cache_code, cache_versions
from tools.train_expanded import ROOT, atomic_json, preflight, sha256


def encode(array, dtype):
    if dtype == "float32":
        return array.astype(np.float32, copy=False)
    if dtype != "uint8":
        raise ValueError("Only explicit uint8 or float32 storage is supported")
    encoded = np.rint(array * np.float32(255)).clip(0, 255).astype(np.uint8)
    if not np.array_equal(encoded.astype(np.float32) / np.float32(255), array):
        raise ValueError("Crop cannot round-trip exactly; build a NEW float32 cache with updated config")
    return encoded


def build(config, root):
    if config["cache_dtype"] not in ("uint8", "float32"):
        raise ValueError("Unsupported cache dtype")
    identity = metadata_identity(config, root)
    preflight(config, root, root / config["manifest"])
    sources = source_identity(config, root)
    destination = root / config["cache_dir"]
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError("Cache or partial build exists; choose a new path, preserve previous evidence")
    partial.mkdir(parents=True)
    started = time.monotonic()
    arrays = {}
    for split in ("train", "val"):
        ds = XBDPairedDataset(root / config["manifest"], split, project_root=root,
                              crop_size=config["crop_size"], context=config["context"], max_open_files=2)
        try:
            shape = (len(ds), 3, config["crop_size"], config["crop_size"])
            # Building in tile order saves source decoding; storage remains manifest order.
            indices = sorted(range(len(ds)), key=lambda i: (ds.rows[i]["event"], ds.rows[i]["tile_id"]))
            for phase in ("pre", "post"):
                name = f"{split}_{phase}.npy"
                array = np.lib.format.open_memmap(partial / name, mode="w+", dtype=config["cache_dtype"], shape=shape)
                for n, i in enumerate(indices, 1):
                    row = ds.rows[i]
                    crop = ds._read_crop(ds._open(ds._path(row[phase + "_image"])), ds._window(row), ds.crop_size).numpy()
                    array[i] = encode(crop, config["cache_dtype"])
                    if n % 2000 == 0:
                        print(f"{split}/{phase}: {n}/{len(ds)}", flush=True)
                array.flush()
                del array
                arrays[name] = sha256(partial / name)
        finally:
            ds.close()
    if source_identity(config, root) != sources or metadata_identity(config, root) != identity:
        raise ValueError("Data changed during cache construction")
    meta = {**identity, "format": 1, "crop_size": config["crop_size"], "context": config["context"],
            "dtype": config["cache_dtype"], "cache_code": cache_code(), "versions": cache_versions(), "sources": sources, "arrays": arrays,
            "seconds": time.monotonic() - started, "test_images_read": False, "exact_roundtrip": True}
    atomic_json(partial / "complete.json", meta)
    partial.rename(destination)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    build(json.loads(args.config.read_text(encoding="utf-8-sig")), args.project_root.resolve())


if __name__ == "__main__":
    main()
