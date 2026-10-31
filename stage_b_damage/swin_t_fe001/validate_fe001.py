"""Validate the FE-001 manifest/config without loading a model or training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


LABELS = ["no_damage", "minor", "major", "destroyed"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(config_path: Path) -> tuple[dict, Path, Path]:
    config_path = config_path.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    project_root = (config_dir / cfg["project_root"]).resolve()
    manifest = (project_root / cfg["manifest"]).resolve()
    return cfg, project_root, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("..\..\configs\swin_t_fe001_b1.json"))
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    cfg, project_root, manifest = resolve(args.config)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    actual_hash = sha256(manifest)
    if actual_hash != cfg["manifest_sha256"]:
        raise RuntimeError(f"Manifest SHA256 mismatch: {actual_hash} != {cfg['manifest_sha256']}")

    with manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    split_rows = Counter(row["split"] for row in rows if row.get("damage") in LABELS)
    expected_rows = {
        "train": cfg["expected_train_rows"],
        "val": cfg["expected_val_rows"],
        "test": cfg["expected_test_rows"],
    }
    if dict(split_rows) != expected_rows:
        raise RuntimeError(f"Split counts differ: {dict(split_rows)} != {expected_rows}")

    expected_events = {
        "train": sorted(cfg["train_events"]),
        "val": sorted(cfg["val_events"]),
        "test": sorted(cfg["test_events"]),
    }
    actual_events = {
        split: sorted({
            row["event"] for row in rows
            if row.get("split") == split and row.get("damage") in LABELS
        })
        for split in ("train", "val", "test")
    }
    if actual_events != expected_events:
        raise RuntimeError(f"Event split differs: {actual_events} != {expected_events}")

    train_counts = Counter(
        row["damage"] for row in rows
        if row.get("split") == "train" and row.get("damage") in LABELS
    )
    if dict(train_counts) != cfg["expected_train_counts"]:
        raise RuntimeError(
            f"Train class counts differ: {dict(train_counts)} != {cfg['expected_train_counts']}"
        )

    if args.check_images:
        missing = []
        for row in rows:
            if row.get("damage") not in LABELS:
                continue
            image = Path(row["post_image"].replace("\\", "/"))
            image = image if image.is_absolute() else project_root / image
            if not image.is_file():
                missing.append(str(image))
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} post images; first={missing[:3]}")

    print(json.dumps({
        "status": "ok",
        "manifest": str(manifest),
        "manifest_sha256": actual_hash,
        "rows": dict(split_rows),
        "events": actual_events,
        "train_counts": dict(train_counts),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

