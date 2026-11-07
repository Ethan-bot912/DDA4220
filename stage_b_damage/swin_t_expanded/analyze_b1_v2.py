"""Recompute and compare the Swin-T B1-v2 validation results.

This is deliberately independent of the training history: all headline
metrics are recomputed from the saved prediction CSVs, aligned by the stable
``(event, tile_id, building_index)`` key.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from metrics import LABELS, metrics_from_predictions


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "expanded_b1_v2_seed42_bs16"
FILES = {
    "v0": ROOT / "runs" / "expanded_eval_best_epoch6" / "predictions.csv",
    "v1_ema": ROOT / "runs" / "expanded_b1_fast_v1_eval_ema" / "predictions.csv",
    "v1_online": ROOT / "runs" / "expanded_b1_fast_v1_eval_online" / "predictions.csv",
    "v2_ema": ROOT / "runs" / "expanded_b1_v2_eval_ema" / "predictions.csv",
    "v2_online": ROOT / "runs" / "expanded_b1_v2_eval_online" / "predictions.csv",
}
LABEL_TO_ID = {name: index for index, name in enumerate(LABELS)}
PROB_COLUMNS = [f"p_{name}" for name in LABELS]
KEY_COLUMNS = ("event", "tile_id", "building_index")


def read_predictions(path: Path) -> tuple[dict[tuple[str, str, str], dict], list[str]]:
    rows: dict[tuple[str, str, str], dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = set(KEY_COLUMNS) | {"actual", "predicted"} | set(PROB_COLUMNS)
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for row in reader:
            key = tuple(row[column] for column in KEY_COLUMNS)
            if key in rows:
                raise ValueError(f"duplicate prediction key in {path}: {key}")
            if row["actual"] not in LABEL_TO_ID or row["predicted"] not in LABEL_TO_ID:
                raise ValueError(f"unknown label in {path}: {row}")
            rows[key] = row
    return rows, fields


def metrics_for_rows(rows: list[dict]) -> dict:
    actual = [LABEL_TO_ID[row["actual"]] for row in rows]
    predicted = [LABEL_TO_ID[row["predicted"]] for row in rows]
    probabilities = np.asarray(
        [[float(row[column]) for column in PROB_COLUMNS] for row in rows], dtype=np.float64
    )
    result = metrics_from_predictions(actual, predicted, probabilities)
    result["predicted_counts"] = {
        name: int(sum(1 for value in predicted if value == index))
        for index, name in enumerate(LABELS)
    }
    result["predicted_fraction"] = {
        name: result["predicted_counts"][name] / max(result["n"], 1)
        for name in LABELS
    }
    return result


def summarize_model(rows_by_key: dict[tuple[str, str, str], dict]) -> dict:
    rows = list(rows_by_key.values())
    pooled = metrics_for_rows(rows)
    events = {}
    for event in sorted({key[0] for key in rows_by_key}):
        event_rows = [row for key, row in rows_by_key.items() if key[0] == event]
        events[event] = metrics_for_rows(event_rows)
    event_f1 = [value["macro_f1"] for value in events.values()]
    event_balanced = [value["balanced_accuracy"] for value in events.values()]
    event_accuracy = [value["accuracy"] for value in events.values()]
    return {
        "n": len(rows),
        "pooled": pooled,
        "per_event": events,
        "event_macro_f1_mean": mean(event_f1),
        "event_macro_f1_std": pstdev(event_f1),
        "event_balanced_accuracy_mean": mean(event_balanced),
        "event_accuracy_mean": mean(event_accuracy),
    }


def metric_delta(left: dict, right: dict) -> dict:
    result = {}
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "macro_ap"):
        if key in left and key in right:
            result[key] = left[key] - right[key]
    result["per_class_f1"] = {
        label: left["per_class"][label]["f1"] - right["per_class"][label]["f1"]
        for label in LABELS
    }
    result["per_class_recall"] = {
        label: left["per_class"][label]["recall"] - right["per_class"][label]["recall"]
        for label in LABELS
    }
    return result


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def markdown_report(all_results: dict, history: list[dict]) -> str:
    best = max(history, key=lambda item: item["val"]["macro_f1"])
    final = history[-1]
    lines = [
        "# B1-v2 完整训练结果分析",
        "",
        "> 独立验证结果由保存的 `predictions.csv` 重新计算；比较时使用 `(event, tile_id, building_index)` 对齐，避免按行误配。",
        "",
        "## 1. 训练状态与配置",
        "",
        f"- 实验：`swin_t_expanded_b1_v2_seed42_bs16`；训练状态：`20/20 complete`。",
        f"- 数据：train 104,152，val 45,002；post-only，Swin-T ImageNet 预训练，224 crop，context=16，batch size=16，梯度累积=2。",
        f"- 增强：D4（90° 旋转/翻转）+ 0.3 概率的轻度 brightness/contrast/saturation jitter。",
        f"- 损失：class-weighted CE（power=0.70）+ 0.20 ordinal BCE；事件 capped sampler（sqrt inverse frequency，最大权重 2.0）；EMA decay=0.999。",
        f"- 训练最佳轮次：epoch {best['epoch']}；最终轮次：epoch {final['epoch']}。",
        "",
        "## 2. 验证集主指标（独立重算）",
        "",
        "| 版本 | Accuracy | Macro-F1 | Balanced Acc | Macro-AP | 事件 Macro-F1 均值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("v0", "v1_ema", "v1_online", "v2_ema", "v2_online"):
        result = all_results[name]
        pooled = result["pooled"]
        lines.append(
            f"| {name} | {fmt(pooled['accuracy'])} | {fmt(pooled['macro_f1'])} | "
            f"{fmt(pooled['balanced_accuracy'])} | {fmt(pooled['macro_ap'])} | "
            f"{fmt(result['event_macro_f1_mean'])} |"
        )
    lines += [
        "",
        "B1-v2 EMA 相对旧 B1-v0：",
        "",
        f"- Accuracy {fmt(all_results['v2_ema']['pooled']['accuracy'] - all_results['v0']['pooled']['accuracy'], 4)}，"
        f"Macro-F1 {fmt(all_results['v2_ema']['pooled']['macro_f1'] - all_results['v0']['pooled']['macro_f1'], 4)}，"
        f"Balanced Accuracy {fmt(all_results['v2_ema']['pooled']['balanced_accuracy'] - all_results['v0']['pooled']['balanced_accuracy'], 4)}，"
        f"Macro-AP {fmt(all_results['v2_ema']['pooled']['macro_ap'] - all_results['v0']['pooled']['macro_ap'], 4)}。",
        f"相对 B1-fast-v1 EMA：Macro-F1 {fmt(all_results['v2_ema']['pooled']['macro_f1'] - all_results['v1_ema']['pooled']['macro_f1'], 4)}，"
        f"Macro-AP {fmt(all_results['v2_ema']['pooled']['macro_ap'] - all_results['v1_ema']['pooled']['macro_ap'], 4)}。",
        "",
        "## 3. B1-v2 EMA 的类别表现",
        "",
        "| 类别 | Precision | Recall | F1 | 支持数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        value = all_results["v2_ema"]["pooled"]["per_class"][label]
        lines.append(
            f"| {label} | {fmt(value['precision'])} | {fmt(value['recall'])} | {fmt(value['f1'])} | {value['support']} |"
        )
    v2_pooled = all_results["v2_ema"]["pooled"]
    lines += [
        "",
        "### 预测分布与混淆矩阵",
        "",
        "| 类别 | 实际占比 | 预测占比 | 预测数 |",
        "|---|---:|---:|---:|",
    ]
    for label in LABELS:
        support = v2_pooled["per_class"][label]["support"]
        lines.append(
            f"| {label} | {fmt(support / v2_pooled['n'])} | "
            f"{fmt(v2_pooled['predicted_fraction'][label])} | {v2_pooled['predicted_counts'][label]} |"
        )
    lines += [
        "",
        "| actual\\pred | no_damage | minor | major | destroyed |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in zip(LABELS, v2_pooled["confusion"]):
        lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")
    lines += [
        "",
        "相对 v0，v2 主要减少了 no_damage/minor 被误判为 destroyed 的数量，从而显著提高 destroyed precision；"
        "代价是 no_damage 的 precision、minor 的 recall 和 major 的 F1 下降。",
        "",
        "### 类别 F1 变化",
        "",
        "| 类别 | v2-v0 F1 | v2-v1-EMA F1 |",
        "|---|---:|---:|",
    ]
    for label in LABELS:
        lines.append(
            f"| {label} | {fmt(all_results['v2_ema']['pooled']['per_class'][label]['f1'] - all_results['v0']['pooled']['per_class'][label]['f1'])} | "
            f"{fmt(all_results['v2_ema']['pooled']['per_class'][label]['f1'] - all_results['v1_ema']['pooled']['per_class'][label]['f1'])} |"
        )
    lines += [
        "",
        "解释：no_damage 与 destroyed 已经是相对稳定的两类；minor 的召回为 "
        f"{fmt(all_results['v2_ema']['pooled']['per_class']['minor']['recall'])}，major 的召回为 "
        f"{fmt(all_results['v2_ema']['pooled']['per_class']['major']['recall'])}，这两个中间等级仍是主要瓶颈。",
        "",
        "## 4. 事件泛化",
        "",
        "| 事件 | n | Accuracy | Macro-F1 | Balanced Acc | Macro-AP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for event, value in all_results["v2_ema"]["per_event"].items():
        pooled = value
        lines.append(
            f"| {event} | {pooled['n']} | {fmt(pooled['accuracy'])} | {fmt(pooled['macro_f1'])} | "
            f"{fmt(pooled['balanced_accuracy'])} | {fmt(pooled['macro_ap'])} |"
        )
    lines += [
        "",
        "| 事件 | no_damage F1 | minor F1 | major F1 | destroyed F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for event, value in all_results["v2_ema"]["per_event"].items():
        lines.append(
            f"| {event} | "
            + " | ".join(fmt(value["per_class"][label]["f1"]) for label in LABELS)
            + " |"
        )
    lines += [
        "",
        f"事件 Macro-F1 无权均值={fmt(all_results['v2_ema']['event_macro_f1_mean'])}，"
        f"标准差={fmt(all_results['v2_ema']['event_macro_f1_std'])}。这比单一 pooled 指标更能反映跨灾害事件的稳定性。",
        "",
        "## 5. 训练曲线与选择结论",
        "",
        f"- epoch 1→4：Macro-F1 从 {fmt(history[0]['val']['macro_f1'])} 升至 {fmt(best['val']['macro_f1'])}；"
        f"随后长期回落，epoch 20 为 {fmt(final['val']['macro_f1'])}。",
        f"- train loss 从 {fmt(history[0]['train_loss'])} 降至 {fmt(final['train_loss'])}，而验证 Macro-F1 在 epoch 4 后下降，说明后期主要是过拟合/域适配过度，而不是欠拟合。",
        "- 因此后续 B2 或最终基线应使用 `best.pt`（epoch 4 EMA），不要使用 `last.pt`。",
        "- online 权重的 Macro-F1 略低于 EMA，但 minor F1 更高、major F1 更低；EMA 仍是更稳妥的默认部署权重。",
        "",
        "## 6. 当前结论与下一步",
        "",
        "B1-v2 已完整跑通，且相对 B1-fast-v1 的 Macro-AP 有提升；但 pooled Macro-F1 只略高于/接近旧 B1-v0，"
        "并未形成明显的整体跃升。模型的关键限制仍是 minor/major 的可分性和跨事件域偏移。下一轮应优先围绕中间等级的阈值/损失权重、事件留一验证和更针对性的 hard-example 采样做小规模验证，"
        "而不是继续把同一配置训练更久。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    loaded = {name: read_predictions(path)[0] for name, path in FILES.items()}
    reference_keys = set(loaded["v2_ema"])
    alignment = {}
    for name, rows in loaded.items():
        keys = set(rows)
        if keys != reference_keys:
            raise ValueError(
                f"key mismatch for {name}: missing={len(reference_keys - keys)}, extra={len(keys - reference_keys)}"
            )
        for key in reference_keys:
            if rows[key]["actual"] != loaded["v2_ema"][key]["actual"]:
                raise ValueError(f"actual-label mismatch at {key} for {name}")
        alignment[name] = {"n": len(rows), "keys_match_v2_ema": True}

    all_results = {name: summarize_model(rows) for name, rows in loaded.items()}
    deltas = {
        "v2_ema_minus_v0": metric_delta(all_results["v2_ema"]["pooled"], all_results["v0"]["pooled"]),
        "v2_ema_minus_v1_ema": metric_delta(all_results["v2_ema"]["pooled"], all_results["v1_ema"]["pooled"]),
        "v2_online_minus_v2_ema": metric_delta(all_results["v2_online"]["pooled"], all_results["v2_ema"]["pooled"]),
    }
    history = [json.loads(line) for line in (RUN / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    output = {
        "manifest_sha256": "77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2",
        "alignment": alignment,
        "results": all_results,
        "deltas": deltas,
        "history_summary": {
            "epochs": len(history),
            "best_epoch": max(history, key=lambda item: item["val"]["macro_f1"])["epoch"],
            "best_macro_f1": max(item["val"]["macro_f1"] for item in history),
            "final_macro_f1": history[-1]["val"]["macro_f1"],
        },
    }
    (RUN / "analysis_comparison.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    comparison_fields = [
        "model",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "macro_ap",
        "event_macro_f1_mean",
        "event_macro_f1_std",
    ]
    with (RUN / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        for name, result in all_results.items():
            pooled = result["pooled"]
            writer.writerow(
                {
                    "model": name,
                    "accuracy": pooled["accuracy"],
                    "macro_f1": pooled["macro_f1"],
                    "balanced_accuracy": pooled["balanced_accuracy"],
                    "macro_ap": pooled.get("macro_ap"),
                    "event_macro_f1_mean": result["event_macro_f1_mean"],
                    "event_macro_f1_std": result["event_macro_f1_std"],
                }
            )
    (RUN / "analysis_report.md").write_text(
        markdown_report(all_results, history), encoding="utf-8"
    )
    print(json.dumps({"written": [str(RUN / "analysis_comparison.json"), str(RUN / "comparison.csv"), str(RUN / "analysis_report.md")], "pooled": {name: value["pooled"] for name, value in all_results.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
