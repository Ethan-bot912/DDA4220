"""Run the complete optimized-B2 train -> raw validation pipeline.

Calibration remains an opt-in diagnostic when ``calibration_fraction`` is
positive; it is never applied in the default fair-training run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from split_calibration import make_split
    from sampler import (
        fallback_manifest_hard_negatives,
        filter_hard_negative_csv,
        manifest_keys,
        mine_hard_negatives,
    )
except ImportError:  # pragma: no cover
    from .split_calibration import make_split
    from .sampler import (
        fallback_manifest_hard_negatives,
        filter_hard_negative_csv,
        manifest_keys,
        mine_hard_negatives,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config_b2_optimized.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/b2_optimized_full"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--calibration-fraction", type=float, default=None)
    parser.add_argument(
        "--hard-negative-input",
        type=Path,
        default=None,
        help="Training-only prediction/key CSV used to build hard-negative keys.",
    )
    parser.add_argument(
        "--disable-hard-negative-bootstrap",
        action="store_true",
        help="Do not reuse a prior train-only hard-negative CSV; use the manifest fallback.",
    )
    parser.add_argument(
        "--fallback-hard-negative-fraction",
        type=float,
        default=0.25,
        help="Fraction of minor/major training rows selected by the fallback.",
    )
    return parser.parse_args()


def _resolve_optional(path: Path | None, config_dir: Path) -> Path | None:
    if path is None:
        return None
    return (path if path.is_absolute() else config_dir / path).resolve()


def _candidate_hard_negative_inputs(config_dir: Path, output_dir: Path) -> list[Path]:
    """Find prior train-only artifacts without inspecting validation/test data."""
    candidates = [
        config_dir / "hard_negatives_seed42_oof.csv",
        config_dir / "runs" / "hard_negatives_seed42_oof.csv",
        config_dir / "runs" / "b2_optimized_full_seed42" / "calibration" / "predictions_calibration.csv",
    ]
    for path in sorted((config_dir / "runs").glob("*/calibration/predictions_calibration.csv")):
        candidates.append(path)
    # Never select an artifact inside the run currently being created.
    current = output_dir.resolve()
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate == current or current in candidate.parents:
            continue
        if candidate.is_file() and str(candidate).lower() not in seen:
            seen.add(str(candidate).lower())
            unique.append(candidate)
    return unique


def _csv_fieldnames(path: Path) -> set[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            return set(next(reader))
        except StopIteration:
            return set()


def prepare_hard_negative_file(
    *,
    source_manifest: Path,
    current_manifest: Path,
    output_dir: Path,
    config_dir: Path,
    explicit_input: Path | None,
    disabled: bool,
    fallback_fraction: float,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    """Prepare a train-only hard-negative key file with leakage filtering."""
    output_csv = output_dir / "hard_negatives.csv"
    allowed = manifest_keys(current_manifest, split="train")
    source: Path | None = _resolve_optional(explicit_input, config_dir)
    if source is None and not disabled:
        candidates = _candidate_hard_negative_inputs(config_dir, output_dir)
        source = candidates[0] if candidates else None

    provenance: dict[str, Any] = {
        "output": str(output_csv.resolve()),
        "allowed_train_keys": len(allowed),
        "source": None,
        "mode": "manifest_fallback",
    }
    if source is not None and source.is_file():
        fields = _csv_fieldnames(source)
        mined_csv = output_dir / "hard_negatives_mined.csv"
        if {"label", "prediction"}.issubset(fields):
            mined = mine_hard_negatives(source, mined_csv)
            candidate = mined_csv
            provenance["mode"] = "prediction_mined"
            provenance["mined"] = mined
        elif "key" in fields or "sample_key" in fields:
            candidate = source
            provenance["mode"] = "key_file_filtered"
        else:
            candidate = None
        if candidate is not None:
            filtered = filter_hard_negative_csv(candidate, output_csv, allowed)
            provenance["source"] = str(source.resolve())
            provenance["filtered"] = filtered
            if filtered["selected"] > 0:
                return output_csv, provenance

    fallback = fallback_manifest_hard_negatives(
        current_manifest,
        output_csv,
        split="train",
        fraction=float(fallback_fraction),
        seed=seed,
    )
    provenance["fallback"] = fallback
    return output_csv, provenance


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    project_root = (config_dir / config["project_root"]).resolve()
    source_manifest = (project_root / config["manifest"]).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fraction = float(
        args.calibration_fraction
        if args.calibration_fraction is not None
        else config.get("calibration_fraction", 0.0)
    )
    if not 0.0 <= fraction < 0.5:
        raise ValueError("calibration fraction must be between 0 (disabled) and 0.5")
    calibration_manifest = output_dir / "manifest_with_calibration.csv"
    split_report = make_split(
        source_manifest,
        calibration_manifest,
        fraction,
        int(config.get("seed", 42)),
    )

    configured_hard_input = (
        args.hard_negative_input
        if args.hard_negative_input is not None
        else (Path(str(config["hard_negative_file"])) if config.get("hard_negative_file") else None)
    )
    hard_negative_path, hard_negative_report = prepare_hard_negative_file(
        source_manifest=source_manifest,
        current_manifest=calibration_manifest,
        output_dir=output_dir,
        config_dir=config_dir,
        explicit_input=configured_hard_input,
        disabled=bool(args.disable_hard_negative_bootstrap),
        fallback_fraction=float(args.fallback_hard_negative_fraction),
        seed=int(config.get("seed", 42)),
    )

    resolved = dict(config)
    resolved["project_root"] = str(project_root)
    resolved["manifest"] = str(calibration_manifest)
    resolved["manifest_sha256"] = sha256_file(calibration_manifest)
    resolved["output_dir"] = str((output_dir / "train").resolve())
    resolved["calibration_fraction"] = fraction
    resolved["hard_negative_file"] = str(hard_negative_path.resolve())
    resolved["hard_negative_provenance"] = hard_negative_report
    resolved["_generated_by"] = "run_full_pipeline.py"
    for key in ("b1_checkpoint", "hard_negative_file", "hard_negative_source"):
        value = str(resolved.get(key, ""))
        if value:
            resolved[key] = str((config_dir / value).resolve())
    resolved_hard_path = Path(resolved["hard_negative_file"]) if resolved.get("hard_negative_file") else None
    if resolved_hard_path is not None and resolved_hard_path.is_file():
        resolved["hard_negative_file_sha256"] = sha256_file(resolved_hard_path)
    resolved_config = output_dir / "resolved_config.json"
    resolved_config.write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")

    python = sys.executable
    train_command = [
        python,
        str(config_dir / "train.py"),
        "--config", str(resolved_config),
        "--device", args.device,
    ]
    if args.epochs is not None:
        train_command += ["--epochs", str(args.epochs)]
    if args.max_train is not None:
        train_command += ["--max-train", str(args.max_train)]
    if args.max_val is not None:
        train_command += ["--max-val", str(args.max_val)]
    run(train_command)

    checkpoint = output_dir / "train" / "best.pt"
    validation_dir = output_dir / "validation_raw"
    run([
        python,
        str(config_dir / "evaluate.py"),
        "--config", str(resolved_config),
        "--manifest", str(calibration_manifest),
        "--checkpoint", str(checkpoint),
        "--split", "val",
        "--output-dir", str(validation_dir),
        "--device", args.device,
    ])

    # Calibration is an optional diagnostic.  In the fair-training mode
    # (fraction=0), there is no held-out calibration split and no global
    # class-offset is fitted or applied to validation predictions.
    calibration_json: Path | None = None
    if fraction > 0.0:
        calibration_dir = output_dir / "calibration"
        run([
            python,
            str(config_dir / "evaluate.py"),
            "--config", str(resolved_config),
            "--manifest", str(calibration_manifest),
            "--checkpoint", str(checkpoint),
            "--split", "calibration",
            "--output-dir", str(calibration_dir),
            "--device", args.device,
        ])
        calibration_json = output_dir / "calibration.json"
        run([
            python,
            str(config_dir / "calibrate.py"),
            "--input", str(calibration_dir / "predictions_calibration.csv"),
            "--apply-to", str(validation_dir / "predictions_val.csv"),
            "--calibrated-csv", str(output_dir / "predictions_val_calibrated.csv"),
            "--output", str(calibration_json),
        ])

    summary: dict[str, Any] = {
        "config": str(resolved_config),
        "source_manifest": str(source_manifest),
        "calibration_manifest": str(calibration_manifest),
        "split_report": split_report,
        "checkpoint": str(checkpoint),
        "hard_negative": hard_negative_report,
        "calibration": str(calibration_json) if calibration_json is not None else None,
        "calibration_applied_to_validation": bool(calibration_json is not None),
        "test_evaluated": False,
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
