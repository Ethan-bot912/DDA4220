"""Compare optimized B2 predictions with the published Swin-T B1/B2 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


LABELS = ["no_damage", "minor", "major", "destroyed"]
NAME_TO_ID = {name: index for index, name in enumerate(LABELS)}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ap_one(target: np.ndarray, score: np.ndarray) -> float:
    positives = int(target.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    target = target[order]
    tp = np.cumsum(target, dtype=np.float64)
    fp = np.cumsum(1 - target, dtype=np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / positives
    previous = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous) * precision))


def metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    matrix = np.zeros((4, 4), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        matrix[int(actual), int(predicted)] += 1
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for index, name in enumerate(LABELS):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - matrix[index, index])
        fn = float(matrix[index, :].sum() - matrix[index, index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index].sum())}
        f1_values.append(f1)
        recalls.append(recall)
    result: dict[str, Any] = {
        "n": int(matrix.sum()),
        "accuracy": float(np.trace(matrix) / max(int(matrix.sum()), 1)),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class": per_class,
        "confusion": matrix.tolist(),
    }
    if probabilities is not None:
        result["macro_ap"] = float(np.mean([
            ap_one((labels == index).astype(np.int64), probabilities[:, index])
            for index in range(4)
        ]))
    return result


def key(row: dict[str, str]) -> str:
    return f"{row['event']}|{row['tile_id']}|{row['building_index']}"


def load_old(path: Path) -> tuple[dict[str, dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rows = read_rows(path)
    table: dict[str, dict[str, Any]] = {}
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    for row in rows:
        actual = NAME_TO_ID[row["actual"]]
        predicted = NAME_TO_ID[row["predicted"]]
        probability = [float(row[f"p_{name}"]) for name in LABELS]
        table[f"{row['event']}|{row['tile_id']}|{row['building_index']}"] = {
            "label": actual,
            "prediction": predicted,
            "event": row["event"],
        }
        labels.append(actual)
        predictions.append(predicted)
        probabilities.append(probability)
    return table, np.asarray(labels), np.asarray(predictions), np.asarray(probabilities)


def load_optimized(path: Path, calibrated: bool = False) -> tuple[dict[str, dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rows = read_rows(path)
    table: dict[str, dict[str, Any]] = {}
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    for row in rows:
        label = int(row["label"])
        if calibrated:
            probability = [float(row[f"calibrated_prob_{index}"]) for index in range(4)]
            prediction = int(np.argmax(probability))
        else:
            logits = np.asarray([float(row[f"logit_{index}"]) for index in range(4)], dtype=np.float64)
            logits = logits - logits.max()
            probability = np.exp(logits) / np.exp(logits).sum()
            prediction = int(row["prediction"])
        sample_key = key(row)
        table[sample_key] = {
            "label": label,
            "prediction": prediction,
            "event": row["event"],
            "change_score": float(row.get("change_score", 0.0)),
            "registration_score": float(row.get("registration_score", 1.0)),
            "gate": float(row.get("gate", 0.0)),
        }
        labels.append(label)
        predictions.append(prediction)
        probabilities.append(probability)
    return table, np.asarray(labels), np.asarray(predictions), np.asarray(probabilities)


def event_breakdown(table: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for event in sorted({item["event"] for item in table.values()}):
        values = [item for item in table.values() if item["event"] == event]
        labels = np.asarray([item["label"] for item in values])
        predictions = np.asarray([item["prediction"] for item in values])
        result[event] = metrics(labels, predictions)
    return result


def quality_breakdown(table: dict[str, dict[str, Any]], field: str, bins: int = 5) -> dict[str, dict[str, Any]]:
    values = np.asarray([item[field] for item in table.values()], dtype=np.float64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    result: dict[str, dict[str, Any]] = {}
    for index in range(len(edges) - 1):
        low, high = float(edges[index]), float(edges[index + 1])
        selected = [item for item in table.values() if (item[field] >= low and (item[field] <= high if index == len(edges) - 2 else item[field] < high))]
        labels = np.asarray([item["label"] for item in selected])
        predictions = np.asarray([item["prediction"] for item in selected])
        result[f"q{index + 1}"] = {"low": low, "high": high, "n": len(selected), **metrics(labels, predictions)}
    return result


def paired_comparison(
    optimized: dict[str, dict[str, Any]],
    b1: dict[str, dict[str, Any]],
    b2: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    common = sorted(set(optimized) & set(b1) & set(b2))
    counts = {"common": len(common), "optimized_only": 0, "b1_only": 0, "b2_only": 0, "all_wrong": 0}
    by_label: dict[str, dict[str, int]] = {name: {"optimized": 0, "b1": 0, "b2": 0} for name in LABELS}
    for sample_key in common:
        label = optimized[sample_key]["label"]
        opt_correct = optimized[sample_key]["prediction"] == label
        b1_correct = b1[sample_key]["prediction"] == label
        b2_correct = b2[sample_key]["prediction"] == label
        by_label[LABELS[label]]["optimized"] += int(opt_correct)
        by_label[LABELS[label]]["b1"] += int(b1_correct)
        by_label[LABELS[label]]["b2"] += int(b2_correct)
        if opt_correct and not b1_correct and not b2_correct:
            counts["optimized_only"] += 1
        elif b1_correct and not opt_correct and not b2_correct:
            counts["b1_only"] += 1
        elif b2_correct and not opt_correct and not b1_correct:
            counts["b2_only"] += 1
        elif not opt_correct and not b1_correct and not b2_correct:
            counts["all_wrong"] += 1
    return {"counts": counts, "by_label_correct": by_label}


def markdown_table(rows: dict[str, dict[str, Any]]) -> str:
    lines = ["| Split/model | Accuracy | Macro-F1 | Balanced Acc. | Macro-AP |", "|---|---:|---:|---:|---:|"]
    for name, value in rows.items():
        lines.append(
            f"| {name} | {value.get('accuracy', 0.0):.4f} | {value.get('macro_f1', 0.0):.4f} | {value.get('balanced_accuracy', 0.0):.4f} | {value.get('macro_ap', 0.0):.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--calibrated", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--b2", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    opt_table, opt_labels, opt_predictions, opt_probabilities = load_optimized(args.optimized)
    cal_table, cal_labels, cal_predictions, cal_probabilities = load_optimized(args.calibrated, calibrated=True)
    b1_table, b1_labels, b1_predictions, b1_probabilities = load_old(args.b1)
    b2_table, b2_labels, b2_predictions, b2_probabilities = load_old(args.b2)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    global_metrics = {
        "published_B1": metrics(b1_labels, b1_predictions, b1_probabilities),
        "published_B2_early": metrics(b2_labels, b2_predictions, b2_probabilities),
        "optimized_raw": metrics(opt_labels, opt_predictions, opt_probabilities),
        "optimized_calibrated": metrics(cal_labels, cal_predictions, cal_probabilities),
    }
    payload = {
        "training": {
            "status": summary.get("status"),
            "completed_epochs": summary.get("completed_epochs"),
            "best_epoch": summary.get("best_epoch"),
            "best_val": summary.get("best_val"),
            "training_seconds": summary.get("training_seconds"),
            "train_rows": summary.get("train_rows"),
            "val_rows": summary.get("val_rows"),
        },
        "global": global_metrics,
        "event_raw": event_breakdown(opt_table),
        "event_calibrated": event_breakdown(cal_table),
        "quality_raw_change_score": quality_breakdown(opt_table, "change_score"),
        "quality_raw_registration_score": quality_breakdown(opt_table, "registration_score"),
        "paired_raw": paired_comparison(opt_table, b1_table, b2_table),
        "paired_calibrated": paired_comparison(cal_table, b1_table, b2_table),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    table = markdown_table(global_metrics)
    report = [
        "# Optimized B2 结果分析",
        "",
        "## 训练状态",
        "",
        f"- 状态：`{summary.get('status')}`；完成 {summary.get('completed_epochs')} 轮；最佳 epoch={summary.get('best_epoch')}。",
        f"- 训练样本：{summary.get('train_rows')}；validation：{summary.get('val_rows')}。",
        f"- 训练耗时：{float(summary.get('training_seconds', 0.0)) / 3600:.2f} 小时。",
        "- test 未读取。",
        "",
        "## 总体指标",
        "",
        table,
        "",
        "## 解释",
        "",
        "- `optimized_raw` 是模型原始 argmax；`optimized_calibrated` 使用 train 内部留出的 calibration split 拟合 temperature/class offsets，再应用到原 validation 预测。",
        "- 事件、质量分层和 B1/B2 配对统计见同目录 `analysis.json`。",
        "- 任何 calibration 后下降都应解释为跨事件校准失配，不应把 calibration 子集上的分数当作最终泛化成绩。",
    ]
    (args.output_dir / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(table)
    print(json.dumps({"output_dir": str(args.output_dir), "global": global_metrics}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
