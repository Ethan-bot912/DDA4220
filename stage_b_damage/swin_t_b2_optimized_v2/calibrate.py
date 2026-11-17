"""Fit post-hoc temperature/class-offset calibration on prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


LABEL_NAMES = ["no_damage", "minor", "major", "destroyed"]


def read_prediction_csv(path: str | Path) -> tuple[list[dict[str, str]], torch.Tensor, torch.Tensor]:
    rows: list[dict[str, str]] = []
    logits: list[list[float]] = []
    labels: list[int] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        logit_fields = [f"logit_{index}" for index in range(4)]
        missing = set(logit_fields) - set(fields)
        if missing:
            raise ValueError(f"Prediction CSV is missing {sorted(missing)}")
        for row in reader:
            rows.append(row)
            logits.append([float(row[name]) for name in logit_fields])
            labels.append(int(row["label"]))
    return rows, torch.tensor(logits, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def fit_calibrator(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int = 80,
) -> tuple[float, list[float]]:
    """Fit one temperature and four class offsets using held-out calibration data."""
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    offsets = torch.nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.LBFGS(
        [log_temperature, offsets],
        lr=0.2,
        max_iter=int(steps),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy((logits + offsets) / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.exp().clamp(0.05, 20.0).detach())
    return temperature, offsets.detach().tolist()


def apply_calibrator(logits: torch.Tensor, temperature: float, offsets: list[float]) -> torch.Tensor:
    return torch.softmax((logits + logits.new_tensor(offsets)) / max(float(temperature), 1e-4), dim=1)


def metrics(labels: torch.Tensor, probabilities: torch.Tensor) -> dict[str, Any]:
    predictions = probabilities.argmax(dim=1)
    matrix = torch.zeros((4, 4), dtype=torch.long)
    for actual, predicted in zip(labels.tolist(), predictions.tolist()):
        matrix[actual, predicted] += 1
    f1: list[float] = []
    per_class: dict[str, dict[str, float]] = {}
    for index, name in enumerate(LABEL_NAMES):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - matrix[index, index])
        fn = float(matrix[index, :].sum() - matrix[index, index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": score}
        f1.append(score)
    return {
        "n": int(labels.numel()),
        "accuracy": float((predictions == labels).float().mean()),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion": matrix.tolist(),
    }


def write_calibrated_csv(
    rows: list[dict[str, str]],
    probabilities: torch.Tensor,
    output: str | Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["label"]
    for index in range(4):
        fieldnames.append(f"calibrated_prob_{index}")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability in zip(rows, probabilities.tolist()):
            result = dict(row)
            for index, value in enumerate(probability):
                result[f"calibrated_prob_{index}"] = f"{value:.9g}"
            writer.writerow(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply-to", type=Path, default=None)
    parser.add_argument("--calibrated-csv", type=Path, default=None)
    args = parser.parse_args()

    rows, logits, labels = read_prediction_csv(args.input)
    temperature, offsets = fit_calibrator(logits, labels)
    before = metrics(labels, torch.softmax(logits, dim=1))
    after_probabilities = apply_calibrator(logits, temperature, offsets)
    after = metrics(labels, after_probabilities)
    payload: dict[str, Any] = {
        "source": str(args.input),
        "temperature": temperature,
        "class_offsets": offsets,
        "calibration_metrics_before": before,
        "calibration_metrics_after": after,
    }
    if args.apply_to is not None:
        eval_rows, eval_logits, eval_labels = read_prediction_csv(args.apply_to)
        eval_probabilities = apply_calibrator(eval_logits, temperature, offsets)
        payload["apply_to"] = str(args.apply_to)
        payload["apply_to_metrics"] = metrics(eval_labels, eval_probabilities)
        if args.calibrated_csv is not None:
            write_calibrated_csv(eval_rows, eval_probabilities, args.calibrated_csv)
            payload["calibrated_csv"] = str(args.calibrated_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
