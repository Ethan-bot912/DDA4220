"""Event/class-balanced and hard-negative sampling utilities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler


def row_key(row: dict[str, str]) -> str:
    return f"{row['event']}|{row['tile_id']}|{row['building_index']}"


def load_hard_negative_keys(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    path = Path(path)
    if not path.is_file():
        return set()
    keys: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("key") or row.get("sample_key")
            if key:
                keys.add(key)
    return keys


def manifest_keys(path: str | Path, split: str = "train") -> set[str]:
    """Return sample keys from one manifest split.

    This is used as a leakage guard when a hard-negative CSV was produced by
    an earlier run: only keys that are still in the current training split are
    allowed into the sampler.  Validation and sealed-test keys are discarded.
    """
    keys: set[str] = set()
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") == split:
                keys.add(row_key(row))
    return keys


def filter_hard_negative_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    allowed_keys: set[str],
) -> dict[str, int]:
    """Filter a mined CSV to keys in the current training split.

    The output is deliberately reduced to ``key,reason`` so provenance is
    simple and the file cannot accidentally carry labels from validation/test.
    """
    input_csv, output_csv = Path(input_csv), Path(output_csv)
    selected: dict[str, str] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("key") or row.get("sample_key") or ""
            if key in allowed_keys:
                selected[key] = row.get("reason") or "mined_hard_negative"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "reason"])
        writer.writeheader()
        for key in sorted(selected):
            writer.writerow({"key": key, "reason": selected[key]})
    return {"input": sum(1 for _ in read_csv_rows(input_csv)), "selected": len(selected)}


def read_csv_rows(path: Path):
    """Yield rows without retaining a potentially large prediction table."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _bucket(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def fallback_manifest_hard_negatives(
    manifest: str | Path,
    output_csv: str | Path,
    *,
    split: str = "train",
    fraction: float = 0.25,
    seed: int = 42,
) -> dict[str, int | str]:
    """Create a deterministic, training-only fallback hard-negative file.

    If no model-prediction CSV is available, emphasize a hash-stable subset of
    the two ambiguous middle grades.  This is intentionally a fallback (not a
    claim of mined model errors); a prediction-derived CSV is preferred by the
    pipeline.  Restricting selection to ``split=train`` keeps validation and
    test sealed.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    output_csv = Path(output_csv)
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    with Path(manifest).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split or row.get("damage") not in {"minor", "major"}:
                continue
            key = row_key(row)
            if _bucket(key, seed) < fraction and key not in seen:
                seen.add(key)
                selected.append({"key": key, "reason": "manifest_middle_grade_fallback"})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "reason"])
        writer.writeheader()
        writer.writerows(sorted(selected, key=lambda row: row["key"]))
    return {"source": str(manifest), "split": split, "selected": len(selected), "fraction": fraction}


def build_weighted_sampler(
    dataset: Any,
    config: dict[str, Any],
    generator: torch.Generator,
) -> tuple[WeightedRandomSampler | None, dict[str, Any], set[str]]:
    settings = config.get("event_sampler", {})
    if not isinstance(settings, dict) or not bool(settings.get("enabled", False)):
        return None, {"enabled": False}, set()
    hard_keys = load_hard_negative_keys(config.get("hard_negative_file"))
    event_counts = Counter(row["event"] for row in dataset.rows)
    class_counts = Counter(row["damage"] for row in dataset.rows)
    max_event = max(event_counts.values())
    max_class = max(class_counts.values())
    event_power = float(settings.get("power", 0.5))
    class_power = float(settings.get("class_power", 0.35))
    event_cap = float(settings.get("max_weight", 2.0))
    hard_multiplier = float(settings.get("hard_negative_multiplier", 2.0))
    weights = []
    for row in dataset.rows:
        event_weight = min((max_event / max(event_counts[row["event"]], 1)) ** event_power, event_cap)
        class_weight = (max_class / max(class_counts[row["damage"]], 1)) ** class_power
        hard_weight = hard_multiplier if row_key(row) in hard_keys else 1.0
        weights.append(event_weight * class_weight * hard_weight)
    tensor = torch.tensor(weights, dtype=torch.double)
    tensor = tensor / tensor.mean().clamp_min(1e-8)
    requested = int(settings.get("samples_per_epoch", 0))
    samples_per_epoch = len(dataset) if requested <= 0 else requested
    sampler = WeightedRandomSampler(
        tensor,
        num_samples=samples_per_epoch,
        replacement=bool(settings.get("replacement", True)),
        generator=generator,
    )
    metadata = {
        "enabled": True,
        "event_power": event_power,
        "class_power": class_power,
        "event_cap": event_cap,
        "hard_negative_multiplier": hard_multiplier,
        "hard_negative_count": len(hard_keys),
        "samples_per_epoch": samples_per_epoch,
        "event_counts": dict(sorted(event_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "mean_weight": float(tensor.mean()),
    }
    return sampler, metadata, hard_keys


def mine_hard_negatives(input_csv: str | Path, output_csv: str | Path) -> dict[str, int]:
    """Create a hard-negative key file from training predictions.

    The input must contain ``key``, ``label`` and ``prediction``. Optional
    ``change_score`` and ``registration_score`` columns make the heuristic more
    selective. This utility is intended for train/out-of-fold predictions only.
    """
    input_csv, output_csv = Path(input_csv), Path(output_csv)
    selected: list[dict[str, str]] = []
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("key", "")
            label = int(row.get("label", -1))
            prediction = int(row.get("prediction", -1))
            change = float(row.get("change_score", 0.0) or 0.0)
            reg = float(row.get("registration_score", 1.0) or 1.0)
            false_alarm = label == 0 and prediction != 0
            missed_damage = label > 0 and prediction == 0
            low_quality = reg < 0.10
            high_change_intact = label == 0 and change >= 0.12
            if key and (false_alarm or missed_damage or low_quality or high_change_intact):
                selected.append({"key": key, "reason": "hard_negative"})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "reason"])
        writer.writeheader()
        writer.writerows(selected)
    return {"selected": len(selected)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(mine_hard_negatives(args.input, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
