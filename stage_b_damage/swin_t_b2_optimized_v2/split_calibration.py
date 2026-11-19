"""Create an event/tile-grouped calibration manifest from a training manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def bucket(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def make_split(source: Path, destination: Path, fraction: float, seed: int) -> dict:
    # ``fraction=0`` is the fair-training mode: preserve the original train
    # split and do not sacrifice examples for post-hoc calibration.  Positive
    # fractions remain available for a separate calibration diagnostic.
    if not 0.0 <= fraction < 0.5:
        raise ValueError("fraction must be between 0 (disabled) and 0.5")
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Manifest has no header")
        rows = list(reader)
        fields = list(reader.fieldnames)
    if fraction == 0.0:
        # No calibration split is requested: preserve the frozen manifest bytes.
        # Rewriting the CSV would change its SHA256 on Windows and break the
        # data-lock/config contract without changing any row.
        counts = Counter(row.get("split", "unknown") for row in rows)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return {
            "source": str(source),
            "destination": str(destination),
            "fraction": fraction,
            "seed": seed,
            "counts": dict(counts),
            "bytes_preserved": True,
        }
    counts = Counter()
    for row in rows:
        if row.get("split") == "train":
            group = f"{row['event']}|{row['tile_id']}"
            if fraction > 0.0 and bucket(group, seed) < fraction:
                row["split"] = "calibration"
                counts["calibration"] += 1
            else:
                counts["train"] += 1
        else:
            counts[row.get("split", "unknown")] += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"source": str(source), "destination": str(destination), "fraction": fraction, "seed": seed, "counts": dict(counts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = make_split(args.manifest, args.output, args.fraction, args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
