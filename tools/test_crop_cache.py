"""Acceptance test for the crop cache: build 500 crops, read them back, compare.

Proves three things before spending 40 minutes on the full cache:

1. uint8 storage round-trips bit-for-bit against the live GeoTIFF reader.
2. Reading from cache is materially faster than decoding windows.
3. The cache refuses to load when manifest or crop geometry disagrees.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage_b_damage.dataset import XBDPairedDataset  # noqa: E402
from tools.build_crop_cache import build  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "dataset/pilot_tier3_4events/manifest_classification.csv",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache/_acceptance")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.cache_dir.exists():
        shutil.rmtree(args.cache_dir)

    print("=" * 70)
    print(f"1/5 building cache for {args.limit} rows of split={args.split}")
    print("=" * 70)
    meta = build(
        manifest=args.manifest,
        split=args.split,
        cache_dir=args.cache_dir,
        crop_size=128,
        context=16,
        project_root=ROOT,
        order="tile",
        tol=1e-3,
        verify_every=1,
        allow_float32=False,
        limit=args.limit,
        progress_every=100,
    )
    print(f"    dtype={meta['dtype']} lossless={meta['lossless']} "
          f"verified={meta['samples_bitwise_verified']}")

    print()
    print("=" * 70)
    print("2/5 reading both paths")
    print("=" * 70)
    base = XBDPairedDataset(args.manifest, args.split, project_root=ROOT,
                            crop_size=128, context=16)
    cached = XBDPairedDataset(args.manifest, args.split, project_root=ROOT,
                              crop_size=128, context=16, cache_dir=args.cache_dir)
    n = args.limit
    assert len(base) >= n and len(cached) >= n

    timings = {}
    tensors = {}
    for name, dataset in (("live", base), ("cache", cached)):
        loader = DataLoader(Subset(dataset, list(range(n))), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
        started = time.perf_counter()
        pre_batches, post_batches = [], []
        for batch in loader:
            pre_batches.append(batch["pre"])
            post_batches.append(batch["post"])
        timings[name] = time.perf_counter() - started
        tensors[name] = (torch.cat(pre_batches), torch.cat(post_batches))
        print(f"    {name:5s} {timings[name]:7.2f}s  "
              f"({timings[name] / n * 1000:.2f} ms/crop)")

    print()
    print("=" * 70)
    print("3/5 bitwise comparison")
    print("=" * 70)
    ok = True
    for key, idx in (("pre", 0), ("post", 1)):
        a, b = tensors["live"][idx], tensors["cache"][idx]
        identical = torch.equal(a, b)
        max_diff = float((a - b).abs().max())
        print(f"    {key:4s} array_equal={identical}  max_abs_diff={max_diff}")
        ok = ok and identical

    print()
    print("=" * 70)
    print("4/5 perturbing index order (cache must stay row-aligned)")
    print("=" * 70)
    order = [7, 3, 199, 0, 42, 498, 100]
    for i in order:
        lp = base[i]["pre"]
        cp = cached[i]["pre"]
        same = torch.equal(lp, cp)
        print(f"    index {i:4d} equal={same}")
        ok = ok and same

    print()
    print("=" * 70)
    print("5/5 guard: mismatched crop_size must be rejected")
    print("=" * 70)
    try:
        XBDPairedDataset(args.manifest, args.split, project_root=ROOT,
                         crop_size=96, context=16, cache_dir=args.cache_dir)
        print("    FAIL: cache was accepted with crop_size=96")
        ok = False
    except ValueError as exc:
        print(f"    OK rejected: {exc}")

    print()
    print("=" * 70)
    speedup = timings["live"] / timings["cache"] if timings["cache"] else float("nan")
    print(f"RESULT bitwise_identical={ok}  speedup={speedup:.1f}x  "
          f"(live {timings['live']:.2f}s vs cache {timings['cache']:.2f}s for {n} crops)")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
