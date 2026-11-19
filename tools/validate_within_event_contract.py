"""Strict preflight for a frozen within-event manifest and one model config."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-b1-checkpoint", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = json.loads(args.split_report.read_text(encoding="utf-8"))
    if not report["acceptance"]["passed"]:
        raise RuntimeError("Split report did not pass")
    manifest_hash = sha256_file(args.manifest)
    if manifest_hash != config["manifest_sha256"]:
        raise RuntimeError(
            f"Manifest hash mismatch: {manifest_hash} != {config['manifest_sha256']}"
        )
    if manifest_hash != report["outputs"]["manifest_classification.csv"]["sha256"]:
        raise RuntimeError("Manifest hash differs from frozen split report")
    if config.get("eval_test"):
        raise RuntimeError("Preflight config must keep eval_test=false")

    by_split = Counter()
    by_split_class = defaultdict(Counter)
    tile_split: dict[tuple[str, str], str] = {}
    building_keys = set()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "split", "event", "tile_id", "building_index", "damage",
            "pre_image", "post_image", "pre_label", "post_label",
        }
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"Manifest missing fields: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            split = row["split"]
            if split not in {"train", "val", "test"}:
                raise RuntimeError(f"Unexpected split: {split}")
            building_key = (row["event"], row["tile_id"], row["building_index"])
            if building_key in building_keys:
                raise RuntimeError(f"Duplicate building key: {building_key}")
            building_keys.add(building_key)
            tile_key = (row["event"], row["tile_id"])
            previous = tile_split.setdefault(tile_key, split)
            if previous != split:
                raise RuntimeError(f"Tile crosses splits: {tile_key}: {previous}, {split}")
            by_split[split] += 1
            by_split_class[split][row["damage"]] += 1

    pooled = report["pooled"]
    for split in ("train", "val", "test"):
        if by_split[split] != pooled[split]["rows"]:
            raise RuntimeError(f"{split} row mismatch: {by_split[split]}")
        if dict(by_split_class[split]) != pooled[split]["class_counts"]:
            raise RuntimeError(f"{split} class mismatch: {dict(by_split_class[split])}")
    for split, expected_rows in (
        ("train", config["expected_train_rows"]),
        ("val", config["expected_val_rows"]),
        ("test", config["expected_test_rows"]),
    ):
        if by_split[split] != expected_rows:
            raise RuntimeError(f"Config {split} count mismatch: {by_split[split]} != {expected_rows}")
    for split, expected_counts in (
        ("train", config["expected_train_counts"]),
        ("val", config["expected_val_counts"]),
        ("test", config["expected_test_counts"]),
    ):
        if dict(by_split_class[split]) != expected_counts:
            raise RuntimeError(f"Config {split} class mismatch")

    checkpoint = config.get("b1_checkpoint")
    checkpoint_path = None
    if checkpoint:
        checkpoint_path = (args.config.resolve().parent / checkpoint).resolve()
        if args.require_b1_checkpoint and not checkpoint_path.is_file():
            raise FileNotFoundError(f"Required same-seed B1 checkpoint missing: {checkpoint_path}")

    result = {
        "status": "ok",
        "config": str(args.config.resolve()),
        "manifest_sha256": manifest_hash,
        "rows": dict(by_split),
        "class_counts": {split: dict(counts) for split, counts in by_split_class.items()},
        "tiles": dict(Counter(tile_split.values())),
        "building_keys": len(building_keys),
        "eval_test": False,
        "b1_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "b1_checkpoint_exists": bool(checkpoint_path and checkpoint_path.is_file()),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
