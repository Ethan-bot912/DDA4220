"""Validate the frozen expanded B2 config and paired manifest without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


LABELS = {"no_damage", "minor", "major", "destroyed"}
REQUIRED_COLUMNS = {
    "split", "event", "tile_id", "building_index", "damage", "pre_image",
    "post_image", "post_bbox_xmin", "post_bbox_ymin", "post_bbox_xmax",
    "post_bbox_ymax",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_config(path: Path) -> tuple[dict, Path, Path]:
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    project_root = (path.parent / config["project_root"]).resolve()
    manifest = (project_root / config["manifest"]).resolve()
    return config, project_root, manifest


def resolve_manifest_path(value: str, project_root: Path) -> Path:
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else project_root / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("..") / ".." / "configs" / "swin_t_expanded_b2_early_seed42_bs16.json",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Check train/val pre and post paths only; sealed test paths are never inspected.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Read this many train and val paired samples after contract checks (0 disables reads).",
    )
    args = parser.parse_args()

    config, project_root, manifest = resolve_config(args.config)
    if config.get("mode") != "early" or config.get("input_channels") != 6:
        raise RuntimeError("B2 config must declare mode=early and input_channels=6")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    actual_hash = sha256(manifest)
    if actual_hash != config["manifest_sha256"]:
        raise RuntimeError(f"Manifest SHA256 mismatch: {actual_hash} != {config['manifest_sha256']}")

    b1_reference = config.get("b1_reference", {})
    b1_config_path = (args.config.resolve().parent / b1_reference["config"]).resolve()
    b1_config = json.loads(b1_config_path.read_text(encoding="utf-8"))
    b1_output = (args.config.resolve().parent / b1_config["output_dir"]).resolve()
    b2_output = (args.config.resolve().parent / config["output_dir"]).resolve()
    if b1_output == b2_output:
        raise RuntimeError("B2 output directory must be separate from the B1 output directory")
    b1_checkpoint = (args.config.resolve().parent / b1_reference["checkpoint"]).resolve()
    if not b1_checkpoint.is_file():
        raise FileNotFoundError(f"B1 reference checkpoint not found: {b1_checkpoint}")

    with manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Manifest has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing paired columns: {sorted(missing)}")
        rows = list(reader)

    labelled = [row for row in rows if row.get("damage") in LABELS]
    split_counts = Counter(row["split"] for row in labelled)
    expected_counts = {
        "train": int(config["expected_train_rows"]),
        "val": int(config["expected_val_rows"]),
    }
    if {key: split_counts[key] for key in expected_counts} != expected_counts:
        raise RuntimeError(
            f"Train/val row counts differ: {dict(split_counts)} != {expected_counts}"
        )
    actual_events = {
        split: sorted({row["event"] for row in labelled if row["split"] == split})
        for split in expected_counts
    }
    expected_events = {
        "train": sorted(config["train_events"]),
        "val": sorted(config["val_events"]),
    }
    if actual_events != expected_events:
        raise RuntimeError(f"Event split differs: {actual_events} != {expected_events}")

    train_counts = Counter(
        row["damage"] for row in labelled if row["split"] == "train"
    )
    if dict(train_counts) != config["expected_train_counts"]:
        raise RuntimeError(
            f"Train class counts differ: {dict(train_counts)} != {config['expected_train_counts']}"
        )

    checked_paths = 0
    if args.check_images:
        for row in labelled:
            if row["split"] not in {"train", "val"}:
                continue
            for field in ("pre_image", "post_image"):
                image = resolve_manifest_path(row[field], project_root)
                if not image.is_file():
                    raise FileNotFoundError(f"Missing {field}: {image}")
                checked_paths += 1

    sampled_shapes: list[dict] = []
    if args.sample:
        if args.sample < 0:
            raise ValueError("--sample must be non-negative")
        # Import lazily so plain manifest validation does not initialize rasterio.
        from dataset import XBDPairedDataset

        paired_crop = config.get("paired_crop", {})
        crop_size = int(config.get("crop_size", paired_crop.get("crop_size", 224)))
        context = int(config.get("context", paired_crop.get("context", 16)))
        for split in ("train", "val"):
            dataset = XBDPairedDataset(
                manifest,
                split,
                project_root=project_root,
                crop_size=crop_size,
                context=context,
                max_rows=args.sample,
                max_open_files=int(config.get("max_open_files", 64)),
                return_meta=True,
            )
            for index in range(min(args.sample, len(dataset))):
                item = dataset[index]
                for name in ("pre", "post", "image"):
                    tensor = item[name]
                    if not bool(tensor.isfinite().all()) or float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
                        raise RuntimeError(f"Invalid {split} sample {index} tensor {name}")
                sampled_shapes.append({
                    "split": split,
                    "index": index,
                    "pre": list(item["pre"].shape),
                    "post": list(item["post"].shape),
                    "image": list(item["image"].shape),
                })

    print(json.dumps({
        "status": "ok",
        "config": str(args.config.resolve()),
        "manifest": str(manifest),
        "manifest_sha256": actual_hash,
        "b1_reference_checkpoint": str(b1_checkpoint),
        "b1_output": str(b1_output),
        "b2_output": str(b2_output),
        "rows": dict(split_counts),
        "events": actual_events,
        "train_counts": dict(train_counts),
        "checked_image_paths": checked_paths,
        "sampled_shapes": sampled_shapes,
        "test_inspected": False,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
