"""Compare two validation prediction exports without touching the sealed test set.

The evaluator writes one row per building.  This script deliberately aligns the
rows by ``(event, tile_id, building_index)`` instead of relying on CSV order,
then recomputes pooled, per-event, per-class, and confusion-matrix metrics for
B1 and B2.  It also reports B2-minus-B1 deltas and paired correctness counts.

Only the two prediction CSVs supplied on the command line are read.  If a
neighbouring ``metrics.json`` exists, its split is checked and must be ``val``;
CSV files containing a ``split`` column are checked in the same way.  No test
manifest, checkpoint, or sealed-test prediction is opened by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence

import numpy as np

from metrics import LABELS, metrics_from_predictions


KEY_COLUMNS = ("event", "tile_id", "building_index")
PROB_COLUMNS = tuple(f"p_{label}" for label in LABELS)
REQUIRED_COLUMNS = set(KEY_COLUMNS) | {"actual", "predicted"} | set(PROB_COLUMNS)
SCALAR_METRICS = ("accuracy", "macro_f1", "balanced_accuracy", "macro_ap")
OUTPUT_FIELDS = (
    "section",
    "group",
    "metric",
    "label",
    "actual",
    "predicted",
    "B1",
    "B2",
    "B2_minus_B1",
)


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _check_val_metadata(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bool:
    """Reject test rows and optionally verify the evaluator sidecar metadata."""

    checked = False
    if "split" in fields:
        checked = True
        splits = {str(row.get("split", "")).strip().lower() for row in rows}
        if splits != {"val"}:
            _fail(f"{path} is not validation-only; split values={sorted(splits)}")

    sidecar = path.with_name("metrics.json")
    if sidecar.exists():
        checked = True
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"cannot read validation metadata {sidecar}: {exc}")
        split = str(metadata.get("split", "")).strip().lower()
        if split != "val":
            _fail(f"{sidecar} declares split={split!r}; refusing non-validation input")
        declared_n = metadata.get("metrics", {}).get("n")
        if declared_n is not None and int(declared_n) != len(rows):
            _fail(f"{sidecar} declares n={declared_n}, but {path} has {len(rows)} rows")
    return checked


def _sidecar_manifest_sha256(path: Path) -> Optional[str]:
    """Return a sidecar manifest hash when one is available."""

    sidecar = path.with_name("metrics.json")
    if not sidecar.exists():
        return None
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    value = metadata.get("manifest_sha256")
    return str(value) if value else None


def read_predictions(path: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], bool]:
    """Read and validate one prediction CSV, preserving key strings exactly."""

    if not path.exists():
        _fail(f"prediction CSV does not exist: {path}")
    if path.suffix.lower() != ".csv":
        _fail(f"prediction input must be a CSV: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = sorted(REQUIRED_COLUMNS - set(fields))
        if missing:
            _fail(f"{path} is missing required columns: {missing}")
        for row in reader:
            # DictReader may return None for a malformed row with too few cells.
            if any(row.get(column) is None for column in REQUIRED_COLUMNS):
                _fail(f"malformed row in {path}: {row}")
            rows.append(row)  # type: ignore[arg-type]

    metadata_checked = _check_val_metadata(path, fields, rows)
    parsed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        key = tuple(str(row[column]) for column in KEY_COLUMNS)
        if any(value == "" for value in key):
            _fail(f"empty alignment key at {path}:{row_number}: {key}")
        if key in parsed:
            _fail(f"duplicate alignment key in {path}: {key}")

        actual = str(row["actual"])
        predicted = str(row["predicted"])
        if actual not in LABELS or predicted not in LABELS:
            _fail(f"unknown label at {path}:{row_number}: actual={actual!r}, predicted={predicted!r}")

        probabilities: list[float] = []
        for column in PROB_COLUMNS:
            try:
                value = float(row[column])
            except (TypeError, ValueError) as exc:
                _fail(f"invalid probability at {path}:{row_number}, {column}: {exc}")
            if not math.isfinite(value) or value < 0.0:
                _fail(f"invalid probability at {path}:{row_number}, {column}: {value!r}")
            probabilities.append(value)
        if abs(sum(probabilities) - 1.0) > 1e-3:
            _fail(
                f"probabilities do not sum to one at {path}:{row_number}: "
                f"sum={sum(probabilities):.8f}"
            )

        parsed[key] = {
            "event": key[0],
            "tile_id": key[1],
            "building_index": key[2],
            "actual": actual,
            "predicted": predicted,
            "probabilities": probabilities,
        }
    return parsed, metadata_checked


def _label_counts(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[str(row[field])] += 1
    return counts


def metrics_for_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actual = [LABELS.index(str(row["actual"])) for row in rows]
    predicted = [LABELS.index(str(row["predicted"])) for row in rows]
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
    result = metrics_from_predictions(actual, predicted, probabilities)
    result["actual_counts"] = _label_counts(rows, "actual")
    result["predicted_counts"] = _label_counts(rows, "predicted")
    result["predicted_fraction"] = {
        label: result["predicted_counts"][label] / max(result["n"], 1)
        for label in LABELS
    }
    return result


def summarize(rows_by_key: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> dict[str, Any]:
    ordered_rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    pooled = metrics_for_rows(ordered_rows)
    events: dict[str, dict[str, Any]] = {}
    for event in sorted({key[0] for key in rows_by_key}):
        event_rows = [row for key, row in rows_by_key.items() if key[0] == event]
        events[event] = metrics_for_rows(event_rows)

    event_f1 = [value["macro_f1"] for value in events.values()]
    event_balanced = [value["balanced_accuracy"] for value in events.values()]
    event_accuracy = [value["accuracy"] for value in events.values()]
    return {
        "n": len(ordered_rows),
        "pooled": pooled,
        "per_event": events,
        "event_macro_f1_mean": float(np.mean(event_f1)) if event_f1 else 0.0,
        "event_macro_f1_std": float(np.std(event_f1)) if event_f1 else 0.0,
        "event_balanced_accuracy_mean": float(np.mean(event_balanced)) if event_balanced else 0.0,
        "event_accuracy_mean": float(np.mean(event_accuracy)) if event_accuracy else 0.0,
    }


def _subtract_nested(b2: Mapping[str, Any], b1: Mapping[str, Any]) -> dict[str, Any]:
    """Return numeric B2-minus-B1 values for one metrics object."""

    result: dict[str, Any] = {}
    for metric in ("n",) + SCALAR_METRICS:
        if metric in b1 and metric in b2:
            result[metric] = b2[metric] - b1[metric]
    result["per_class"] = {}
    for label in LABELS:
        result["per_class"][label] = {}
        for metric in ("precision", "recall", "f1", "support"):
            result["per_class"][label][metric] = (
                b2["per_class"][label][metric] - b1["per_class"][label][metric]
            )
    result["confusion"] = [
        [int(b2["confusion"][actual][predicted] - b1["confusion"][actual][predicted]) for predicted in range(4)]
        for actual in range(4)
    ]
    return result


def delta_summary(b1: Mapping[str, Any], b2: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pooled": _subtract_nested(b2["pooled"], b1["pooled"]),
        "per_event": {},
        "event_macro_f1_mean": b2["event_macro_f1_mean"] - b1["event_macro_f1_mean"],
        "event_macro_f1_std": b2["event_macro_f1_std"] - b1["event_macro_f1_std"],
        "event_balanced_accuracy_mean": (
            b2["event_balanced_accuracy_mean"] - b1["event_balanced_accuracy_mean"]
        ),
        "event_accuracy_mean": b2["event_accuracy_mean"] - b1["event_accuracy_mean"],
    }
    for event in sorted(b1["per_event"]):
        result["per_event"][event] = _subtract_nested(
            b2["per_event"][event], b1["per_event"][event]
        )
    return result


def paired_summary(
    b1: Mapping[tuple[str, str, str], Mapping[str, Any]],
    b2: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    outcomes = {"both_correct": 0, "B1_only_correct": 0, "B2_only_correct": 0, "both_wrong": 0}
    by_actual = {
        label: {name: 0 for name in outcomes}
        for label in LABELS
    }
    transition = [[0 for _ in LABELS] for _ in LABELS]
    for key in sorted(b1):
        row1, row2 = b1[key], b2[key]
        b1_correct = row1["actual"] == row1["predicted"]
        b2_correct = row2["actual"] == row2["predicted"]
        if b1_correct and b2_correct:
            outcome = "both_correct"
        elif b1_correct:
            outcome = "B1_only_correct"
        elif b2_correct:
            outcome = "B2_only_correct"
        else:
            outcome = "both_wrong"
        outcomes[outcome] += 1
        by_actual[row1["actual"]][outcome] += 1
        transition[LABELS.index(row1["predicted"])][LABELS.index(row2["predicted"])] += 1
    n = len(b1)
    return {
        **outcomes,
        "disagreement_count": n - outcomes["both_correct"] - outcomes["both_wrong"],
        "disagreement_rate": (
            (n - outcomes["both_correct"] - outcomes["both_wrong"]) / n if n else 0.0
        ),
        "outcome_by_actual": by_actual,
        "prediction_transition_B1_rows_B2_columns": transition,
    }


def _row(
    section: str,
    group: str,
    metric: str,
    b1: Any,
    b2: Any,
    *,
    label: str = "",
    actual: str = "",
    predicted: str = "",
) -> dict[str, Any]:
    return {
        "section": section,
        "group": group,
        "metric": metric,
        "label": label,
        "actual": actual,
        "predicted": predicted,
        "B1": b1,
        "B2": b2,
        "B2_minus_B1": b2 - b1 if isinstance(b1, (int, float)) and isinstance(b2, (int, float)) else "",
    }


def comparison_rows(b1: Mapping[str, Any], b2: Mapping[str, Any], paired: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, left, right in (
        ("pooled", b1["pooled"], b2["pooled"]),
    ):
        for metric in ("n",) + SCALAR_METRICS:
            rows.append(_row("pooled", group, metric, left[metric], right[metric]))
        for label in LABELS:
            for metric in ("precision", "recall", "f1", "support"):
                rows.append(
                    _row(
                        "per_class",
                        group,
                        metric,
                        left["per_class"][label][metric],
                        right["per_class"][label][metric],
                        label=label,
                    )
                )
        for actual_index, actual in enumerate(LABELS):
            for predicted_index, predicted in enumerate(LABELS):
                rows.append(
                    _row(
                        "confusion",
                        group,
                        "count",
                        left["confusion"][actual_index][predicted_index],
                        right["confusion"][actual_index][predicted_index],
                        actual=actual,
                        predicted=predicted,
                    )
                )

    for event in sorted(b1["per_event"]):
        left, right = b1["per_event"][event], b2["per_event"][event]
        for metric in ("n",) + SCALAR_METRICS:
            rows.append(_row("per_event", event, metric, left[metric], right[metric]))
        for label in LABELS:
            for metric in ("precision", "recall", "f1", "support"):
                rows.append(
                    _row(
                        "per_event_class",
                        event,
                        metric,
                        left["per_class"][label][metric],
                        right["per_class"][label][metric],
                        label=label,
                    )
                )
        for actual_index, actual in enumerate(LABELS):
            for predicted_index, predicted in enumerate(LABELS):
                rows.append(
                    _row(
                        "per_event_confusion",
                        event,
                        "count",
                        left["confusion"][actual_index][predicted_index],
                        right["confusion"][actual_index][predicted_index],
                        actual=actual,
                        predicted=predicted,
                    )
                )

    for metric in (
        "event_macro_f1_mean",
        "event_macro_f1_std",
        "event_balanced_accuracy_mean",
        "event_accuracy_mean",
    ):
        rows.append(_row("event_aggregate", "all_events", metric, b1[metric], b2[metric]))
    for metric in (
        "both_correct",
        "B1_only_correct",
        "B2_only_correct",
        "both_wrong",
        "disagreement_count",
        "disagreement_rate",
    ):
        rows.append(_row("paired", "all_rows", metric, "", paired[metric]))
    return rows


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalar values recursively for strict JSON output."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1", type=Path, required=True, help="B1 validation predictions.csv")
    parser.add_argument("--b2", type=Path, required=True, help="B2 validation predictions.csv")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Allow replacing existing output files")
    args = parser.parse_args()

    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()
    input_paths = {args.b1.resolve(), args.b2.resolve()}
    if output_json in input_paths or output_csv in input_paths:
        parser.error("output paths must not overwrite either prediction input")
    if not args.force:
        existing = [str(path) for path in (output_json, output_csv) if path.exists()]
        if existing:
            parser.error(f"output exists (use --force to replace): {existing}")

    b1, b1_metadata_checked = read_predictions(args.b1.resolve())
    b2, b2_metadata_checked = read_predictions(args.b2.resolve())
    if not b1_metadata_checked or not b2_metadata_checked:
        _fail(
            "validation provenance is missing: each prediction CSV must either "
            "contain a split=val column or sit beside a metrics.json declaring split=val"
        )
    b1_manifest_sha256 = _sidecar_manifest_sha256(args.b1.resolve())
    b2_manifest_sha256 = _sidecar_manifest_sha256(args.b2.resolve())
    if b1_manifest_sha256 and b2_manifest_sha256 and b1_manifest_sha256 != b2_manifest_sha256:
        _fail(
            "B1/B2 validation sidecars report different manifest SHA256 values: "
            f"B1={b1_manifest_sha256}, B2={b2_manifest_sha256}"
        )
    b1_keys, b2_keys = set(b1), set(b2)
    if b1_keys != b2_keys:
        missing = sorted(b1_keys - b2_keys)[:5]
        extra = sorted(b2_keys - b1_keys)[:5]
        _fail(
            "B1/B2 key sets differ: "
            f"missing_in_B2={len(b1_keys - b2_keys)} {missing}; "
            f"extra_in_B2={len(b2_keys - b1_keys)} {extra}"
        )
    actual_mismatches = [key for key in sorted(b1_keys) if b1[key]["actual"] != b2[key]["actual"]]
    if actual_mismatches:
        _fail(f"actual-label mismatch for {len(actual_mismatches)} aligned rows; first={actual_mismatches[:5]}")
    if not b1_keys:
        _fail("prediction files are empty")
    if {key[0] for key in b1_keys} != {key[0] for key in b2_keys}:
        _fail("event sets differ after key alignment")

    b1_summary = summarize(b1)
    b2_summary = summarize(b2)
    paired = paired_summary(b1, b2)
    result = {
        "scope": "validation_only",
        "validation_only": True,
        "test_rows_read": 0,
        "key_columns": list(KEY_COLUMNS),
        "inputs": {
            "B1_predictions": str(args.b1.resolve()),
            "B2_predictions": str(args.b2.resolve()),
            "B1_val_metadata_checked": b1_metadata_checked,
            "B2_val_metadata_checked": b2_metadata_checked,
            "manifest_sha256": b1_manifest_sha256 or b2_manifest_sha256,
        },
        "alignment": {
            "B1_rows": len(b1),
            "B2_rows": len(b2),
            "aligned_rows": len(b1_keys),
            "unique_keys": len(b1_keys),
            "key_sets_match": True,
            "actual_labels_match": True,
            "events": sorted({key[0] for key in b1_keys}),
        },
        "B1": b1_summary,
        "B2": b2_summary,
        "B2_minus_B1": delta_summary(b1_summary, b2_summary),
        "paired": paired,
    }
    rows = comparison_rows(b1_summary, b2_summary, paired)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_json_safe(result), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "written": [str(output_json), str(output_csv)],
                "aligned_rows": len(b1_keys),
                "B1": b1_summary["pooled"],
                "B2": b2_summary["pooled"],
                "B2_minus_B1": result["B2_minus_B1"]["pooled"],
                "paired": paired,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
