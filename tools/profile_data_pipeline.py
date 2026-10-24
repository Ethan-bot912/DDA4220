"""Measure the xBD loader ladder: baseline -> +cache -> +grouping -> +workers.

Every configuration uses the frozen ``XBDPairedDataset`` read path (same window,
same bilinear resampling); the only things that change are GDAL's block cache
size, the order in which buildings are requested, and the number of loader
workers. The script also verifies that the decoded tensors are bit-identical,
so a speed-up can never come from silently changing the data.

    python tools/profile_data_pipeline.py ^
      --manifest dataset\pilot_tier3_4events\manifest_classification.csv ^
      --project-root . --output output\profile_loader.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import rasterio
import torch
from torch.utils.data import DataLoader

from stage_b_damage.dataset import XBDPairedDataset
from stage_b_damage.fast_dataset import TileGroupedBatchSampler, distinct_tiles_for_batch


BASE_CACHE_MB = 128  # what the published expanded run used
BIG_CACHE_MB = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--equivalence-samples", type=int, default=256)
    parser.add_argument("--max-open-files", type=int, default=256)
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> XBDPairedDataset:
    return XBDPairedDataset(
        args.manifest,
        args.split,
        project_root=args.project_root,
        crop_size=args.crop_size,
        context=args.context,
        max_open_files=args.max_open_files,
        return_meta=True,
    )


def measure(config: str, loader: DataLoader, dataset, batches: int, warmup: int) -> dict:
    iterator = iter(loader)
    for _ in range(warmup):
        try:
            next(iterator)
        except StopIteration:
            break
    times: list[float] = []
    tile_counts: list[int] = []
    samples = 0
    started = time.perf_counter()
    for _ in range(batches):
        begin = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        times.append(time.perf_counter() - begin)
        meta = batch.get("meta")
        if isinstance(meta, dict) and meta:
            count = len(next(iter(meta.values())))
            tile_counts.append(
                len({(meta["event"][i], meta["tile_id"][i]) for i in range(count)})
            )
        samples += int(batch["label"].numel())
    elapsed = time.perf_counter() - started
    if not times:
        raise RuntimeError(f"{config}: no batches were produced")
    ordered = sorted(times)

    def pct(q: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[index]

    return {
        "config": config,
        "batches": len(times),
        "samples": samples,
        "total_seconds": elapsed,
        "samples_per_second": samples / elapsed if elapsed else 0.0,
        # Throughput-based mean: with prefetching workers the per-call wait is
        # often ~0, so the wall-clock average is the meaningful number.
        "batch_mean_seconds": elapsed / len(times),
        "batch_wait_p50_seconds": pct(0.50),
        "batch_wait_p90_seconds": pct(0.90),
        "mean_distinct_tiles_per_batch": statistics.fmean(tile_counts) if tile_counts else None,
    }


def tensors_for(dataset, indices: list[int]) -> dict[int, torch.Tensor]:
    """Read identical index -> post tensor under whatever GDAL env is active."""
    out = {}
    for index in indices:
        item = dataset[index]
        out[index] = torch.cat((item["pre"].flatten(), item["post"].flatten()))
    return out


def main() -> int:
    args = parse_args()
    # Worker processes are spawned, so the GDAL cache budget has to be visible
    # through the environment before they import rasterio.
    os.environ["GDAL_CACHEMAX"] = str(BIG_CACHE_MB * 1024 * 1024)
    args.manifest = args.manifest.resolve()
    args.project_root = args.project_root.resolve()
    results: list[dict] = []

    with rasterio.Env(GDAL_CACHEMAX=BASE_CACHE_MB * 1024 * 1024):
        base_ds = build_dataset(args)
        base_loader = DataLoader(
            base_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=0,
        )
        results.append(measure("baseline (128MB gdal, random order, workers=0)", base_loader, base_ds, args.batches, args.warmup))

        step = max(1, len(base_ds) // max(args.equivalence_samples, 1))
        probe_indices = list(range(0, len(base_ds), step))[: args.equivalence_samples]
        base_tensors = tensors_for(base_ds, probe_indices)

    with rasterio.Env(GDAL_CACHEMAX=BIG_CACHE_MB * 1024 * 1024):
        big_ds = build_dataset(args)
        big_loader = DataLoader(
            big_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=0,
        )
        results.append(measure("+LRU/gdal4GB (random order, workers=0)", big_loader, big_ds, args.batches, args.warmup))
        big_tensors = tensors_for(big_ds, probe_indices)
        max_diff_random = max(
            float((base_tensors[i] - big_tensors[i]).abs().max()) for i in probe_indices
        )

        grouped_ds = build_dataset(args)
        grouped_sampler = TileGroupedBatchSampler(grouped_ds.rows, args.batch_size, seed=args.seed)
        grouped_loader = DataLoader(grouped_ds, batch_sampler=grouped_sampler, num_workers=0)
        results.append(measure("+grouped tiles (4GB gdal, workers=0)", grouped_loader, grouped_ds, args.batches, args.warmup))
        grouped_tensors = tensors_for(grouped_ds, probe_indices)
        max_diff_grouped = max(
            float((base_tensors[i] - grouped_tensors[i]).abs().max()) for i in probe_indices
        )

        random_worker_ds = build_dataset(args)
        random_worker_loader = DataLoader(
            random_worker_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None,
        )
        results.append(measure(f"+workers={args.workers} only (random order)", random_worker_loader, random_worker_ds, args.batches, args.warmup))

        worker_ds = build_dataset(args)
        worker_sampler = TileGroupedBatchSampler(worker_ds.rows, args.batch_size, seed=args.seed)
        worker_loader = DataLoader(
            worker_ds,
            batch_sampler=worker_sampler,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None,
        )
        results.append(measure(f"+grouped+workers={args.workers}", worker_loader, worker_ds, args.batches, args.warmup))

        wide_ds = build_dataset(args)
        wide_loader = DataLoader(
            wide_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=2,
        )
        results.append(measure("+workers=8 only (random order)", wide_loader, wide_ds, args.batches, args.warmup))

    baseline = results[0]
    for row in results:
        row["speedup_vs_baseline"] = (
            baseline["batch_mean_seconds"] / row["batch_mean_seconds"]
            if row["batch_mean_seconds"]
            else None
        )
        row["io_share_estimate"] = None

    report = {
        "manifest": str(args.manifest),
        "split": args.split,
        "rows": len(base_ds),
        "batch_size": args.batch_size,
        "crop_size": args.crop_size,
        "context": args.context,
        "seed": args.seed,
        "results": results,
        "equivalence": {
            "samples_checked": len(probe_indices),
            "max_abs_diff_random_order": max_diff_random,
            "max_abs_diff_grouped_order": max_diff_grouped,
            "bit_identical_random_order": max_diff_random == 0.0,
            "bit_identical_grouped_order": max_diff_grouped == 0.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
