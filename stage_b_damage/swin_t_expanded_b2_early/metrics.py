"""Dependency-light four-class classification metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np


LABELS = ["no_damage", "minor", "major", "destroyed"]


def confusion_matrix(labels: Iterable[int], predictions: Iterable[int], num_classes: int = 4) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def _binary_average_precision(target: np.ndarray, score: np.ndarray) -> float:
    positives = int(target.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    target = target[order]
    true_positives = np.cumsum(target, dtype=np.float64)
    false_positives = np.cumsum(1 - target, dtype=np.float64)
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / positives
    previous_recall = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous_recall) * precision))


def macro_average_precision(labels: Iterable[int], probabilities: np.ndarray) -> float:
    labels_array = np.asarray(list(labels), dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS):
        raise ValueError("probabilities must have shape (n, 4)")
    values = []
    for class_index in range(len(LABELS)):
        target = (labels_array == class_index).astype(np.int64)
        values.append(_binary_average_precision(target, probabilities[:, class_index]))
    return float(np.mean(values)) if values else 0.0


def metrics_from_predictions(
    labels: Iterable[int], predictions: Iterable[int], probabilities: np.ndarray | None = None
) -> dict:
    labels_list = list(labels)
    predictions_list = list(predictions)
    result = metrics_from_confusion(confusion_matrix(labels_list, predictions_list, num_classes=4))
    if probabilities is not None:
        result["macro_ap"] = macro_average_precision(labels_list, probabilities)
    return result


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=np.int64)
    total = int(matrix.sum())
    per_class: dict[str, dict] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    for index, name in enumerate(LABELS):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            recalls.append(recall)
            f1_values.append(f1)
    return {
        "n": total,
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "per_class": per_class,
        "confusion": matrix.tolist(),
    }

