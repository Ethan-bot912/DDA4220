"""Pre-compute Stage B building crops into on-disk uint8 memmaps.

The training pipeline re-decodes the same GeoTIFF windows every epoch, which
dominates wall-clock time on CPU. The crops themselves are deterministic and
do not change between epochs, so they can be materialised once.

Losslessness is verified rather than assumed: the in-memory tensors are float32
in [0, 1] produced by ``np.clip(array, 0, 255) / 255.0``. When the underlying
window happens to scale 1:1 (integer-valued pixels), ``x * 255`` is an exact
integer and uint8 storage round-trips bit-for-bit. When resampling produces
genuine fractions that would be quantised, the builder fails loudly instead of
silently writing lossy data; pass ``--allow-float32`` to store float32 instead.

Output layout (one pair of files per split):

    <cache-dir>/<split>_pre.npy    (n, 3, size, size) uint8
    <cache-dir>/<split>_post.npy   (n, 3, size, size) uint8
    <cache-dir>/<split>_meta.json  provenance + losslessness record

Rows are read in tile order so each GeoTIFF is opened once, but every sample is
written at its original manifest row index so that ``Subset(indices)`` keeps
its meaning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage_b_damage.dataset import XBDPairedDataset  # noqa: E402


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_uint8(array: np.ndarray, tol: float) -> np.ndarray | None:
    """Exact-integer uint8 view of a [0,1] float32 crop, else ``None``."""
    scaled = np.asarray(array, dtype=np.float32) * np.float32(255.0)
    if np.abs(scaled - np.round(scaled)).max(initial=0.0) > tol:
        return None
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def build(
    *,
    manifest: Path,
    split: str,
    cache_dir: Path,
    crop_size: int,
    context: int,
    project_root: Path,
    order: str,
    tol: float,
    verify_every: int,
    allow_float32: bool,
    limit: int,
    progress_every: int,
) -> dict:
    dataset = XBDPairedDataset(
        manifest,
        split,
        project_root=project_root,
        crop_size=crop_size,
        context=context,
        return_meta=False,
    )
    total = len(dataset)
    n = total if not limit else min(limit, total)

    indices = list(range(n))
    if order == "tile":
        rows = dataset.rows
        indices.sort(key=lambda i: (rows[i].get("event", ""), rows[i].get("tile_id", "")))

    cache_dir.mkdir(parents=True, exist_ok=True)
    shape = (n, 3, crop_size, crop_size)
    pre_path = cache_dir / f"{split}_pre.npy"
    post_path = cache_dir / f"{split}_post.npy"

    # Probe the first sample to decide the storage dtype before allocating.
    t0 = time.perf_counter()
    probe = dataset[indices[0]]
    fill_seconds = time.perf_counter() - t0

    uint8_ok = all(to_uint8(probe[key].numpy(), tol) is not None for key in ("pre", "post"))
    if not uint8_ok and not allow_float32:
        dataset.close()
        raise SystemExit(
            f"[{split}] crop {indices[0]} is not integer-valued at 1/255 resolution; "
            "uint8 storage would be lossy. Re-run with --allow-float32 to store float32."
        )
    dtype = np.uint8 if uint8_ok else np.float32
    print(f"[{split}] n={n} dtype={dtype.__name__} probe={fill_seconds * 1000:.1f}ms", flush=True)

    pre = np.lib.format.open_memmap(pre_path, mode="w+", dtype=dtype, shape=shape)
    post = np.lib.format.open_memmap(post_path, mode="w+", dtype=dtype, shape=shape)

    checked = 0
    worst_error = 0.0
    started = time.perf_counter()
    for position, index in enumerate(indices):
        item = dataset[index]
        for key, target in (("pre", pre), ("post", post)):
            original = item[key].numpy()
            quantised = to_uint8(original, tol)
            if quantised is not None:
                target[index] = quantised
                if verify_every and position % verify_every == 0:
                    restored = quantised.astype(np.float32) / np.float32(255.0)
                    if not np.array_equal(original, restored):
                        dataset.close()
                        raise SystemExit(
                            f"[{split}] uint8 round-trip mismatch at index {index} ({key}); "
                            "refusing to write a lossy cache"
                        )
                    checked += 1
            elif dtype is np.float32:
                target[index] = original
                checked += 1
            else:
                dataset.close()
                raise SystemExit(
                    f"[{split}] index {index} ({key}) unexpected non-integer crop; "
                    "storage dtype was chosen from the probe sample only"
                )
            if dtype is np.float32:
                recovered = np.asarray(target[index], dtype=np.float32)
                if verify_every and position % verify_every == 0:
                    worst_error = max(
                        worst_error, float(np.abs(recovered - original).max(initial=0.0))
                    )
        if progress_every and position % progress_every == 0:
            done = position + 1
            rate = done / max(time.perf_counter() - started, 1e-9)
            print(
                f"[{split}] {done}/{n}  {rate:.1f} crops/s  "
                f"eta={(n - done) / rate / 60:.1f}min",
                flush=True,
            )

    pre.flush()
    post.flush()
    elapsed = time.perf_counter() - started
    dataset.close()

    counts = Counter(dataset.rows[i].get("damage", "") for i in indices)
    meta = {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256(manifest),
        "split": split,
        "rows": n,
        "total_rows_in_split": total,
        "row_limit": limit,
        "order": order,
        "crop_size": crop_size,
        "context": context,
        "dtype": np.dtype(dtype).name,
        "shape": list(shape),
        "pre_file": pre_path.name,
        "post_file": post_path.name,
        "bytes_per_file": int(np.prod(shape)) * np.dtype(dtype).itemsize,
        "lossless": bool(np.dtype(dtype) == np.uint8),
        "round_trip_tolerance": tol,
        "samples_bitwise_verified": checked,
        "verify_every": verify_every,
        "float32_max_abs_error": worst_error,
        "label_counts": dict(counts),
        "build_seconds": elapsed,
        "crops_per_second": n / elapsed if elapsed else None,
        "torch_version": __import__("torch").__version__,
        "numpy_version": np.__version__,
    }
    (cache_dir / f"{split}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[{split}] done in {elapsed / 60:.1f}min  dtype={meta['dtype']}  "
        f"lossless={meta['lossless']}  verified={checked}",
        flush=True,
    )
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "dataset/pilot_tier3_4events/manifest_classification.csv",
    )
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache/pilot_tier3_4events")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--order", choices=("manifest", "tile"), default="tile")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--verify-every", type=int, default=50)
    parser.add_argument("--allow-float32", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    splits = args.split or ["val"]
    for split in splits:
        build(
            manifest=args.manifest,
            split=split,
            cache_dir=args.cache_dir,
            crop_size=args.crop_size,
            context=args.context,
            project_root=args.project_root,
            order=args.order,
            tol=args.tol,
            verify_every=args.verify_every,
            allow_float32=args.allow_float32,
            limit=args.limit,
            progress_every=args.progress_every,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
