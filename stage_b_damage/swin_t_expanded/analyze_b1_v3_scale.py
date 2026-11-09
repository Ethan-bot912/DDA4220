"""Independent analysis for the B1-v3 relative-scale experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from analyze_b1_v2 import LABELS, metric_delta, read_predictions, summarize_model


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "expanded_b1_v3_scale_seed42_bs16"
FILES = {
    "v0": ROOT / "runs" / "expanded_eval_best_epoch6" / "predictions.csv",
    "v1_ema": ROOT / "runs" / "expanded_b1_fast_v1_eval_ema" / "predictions.csv",
    "v2_ema": ROOT / "runs" / "expanded_b1_v2_eval_ema" / "predictions.csv",
    "v3_scale_ema": ROOT / "runs" / "expanded_b1_v3_scale_eval_ema" / "predictions.csv",
    "v3_scale_online": ROOT / "runs" / "expanded_b1_v3_scale_eval_online" / "predictions.csv",
}


def f(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    loaded = {name: read_predictions(path)[0] for name, path in FILES.items()}
    reference = set(loaded["v3_scale_ema"])
    for name, rows in loaded.items():
        if set(rows) != reference:
            raise ValueError(f"prediction keys differ for {name}")
        for key in reference:
            if rows[key]["actual"] != loaded["v3_scale_ema"][key]["actual"]:
                raise ValueError(f"actual label mismatch for {name} at {key}")
    results = {name: summarize_model(rows) for name, rows in loaded.items()}
    history = [
        json.loads(line)
        for line in (RUN / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    best = max(history, key=lambda item: item["val"]["macro_f1"])
    deltas = {
        "v3_scale_ema_minus_v0": metric_delta(results["v3_scale_ema"]["pooled"], results["v0"]["pooled"]),
        "v3_scale_ema_minus_v2_ema": metric_delta(results["v3_scale_ema"]["pooled"], results["v2_ema"]["pooled"]),
        "v3_scale_online_minus_v3_scale_ema": metric_delta(results["v3_scale_online"]["pooled"], results["v3_scale_ema"]["pooled"]),
    }
    output = {
        "manifest_sha256": "77e78d056d2a7494a739c5a8debaa93e11ad4357f6afcaf97aca4538c2769ed2",
        "results": results,
        "deltas": deltas,
        "history_summary": {
            "completed_epochs": len(history),
            "best_epoch": best["epoch"],
            "best_macro_f1": best["val"]["macro_f1"],
            "stopping_reason": "early_stopping",
        },
    }
    (RUN / "analysis_comparison.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (RUN / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "model",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "macro_ap",
            "event_macro_f1_mean",
            "event_macro_f1_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
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
    lines = [
        "# B1-v3-scale 实验结果分析",
        "",
        "## 训练状态",
        "",
        f"- 完成 {len(history)} 个 epoch，早停；最佳 epoch={best['epoch']}。",
        "- 唯一训练主轴：将固定像素 context 改为相对 bbox margin，并在训练端加入 0.75–1.35 scale jitter；其余 v2 配方保持不变。",
        "- 验证集使用固定 margin=0.75、无随机 jitter；数据划分、seed、batch 和 manifest 与 v2 相同。",
        "",
        "## Pooled validation（独立重算）",
        "",
        "| 版本 | Accuracy | Macro-F1 | Balanced Acc | Macro-AP | 事件 Macro-F1 均值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("v0", "v1_ema", "v2_ema", "v3_scale_ema", "v3_scale_online"):
        r = results[name]
        p = r["pooled"]
        lines.append(
            f"| {name} | {f(p['accuracy'])} | {f(p['macro_f1'])} | {f(p['balanced_accuracy'])} | {f(p['macro_ap'])} | {f(r['event_macro_f1_mean'])} |"
        )
    lines += [
        "",
        f"相对 v2 EMA：Accuracy {f(results['v3_scale_ema']['pooled']['accuracy'] - results['v2_ema']['pooled']['accuracy'])}，"
        f"Macro-F1 {f(results['v3_scale_ema']['pooled']['macro_f1'] - results['v2_ema']['pooled']['macro_f1'])}，"
        f"Balanced Accuracy {f(results['v3_scale_ema']['pooled']['balanced_accuracy'] - results['v2_ema']['pooled']['balanced_accuracy'])}，"
        f"Macro-AP {f(results['v3_scale_ema']['pooled']['macro_ap'] - results['v2_ema']['pooled']['macro_ap'])}。",
        "",
        "## v3-scale EMA 类别结果",
        "",
        "| 类别 | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for label in LABELS:
        v = results["v3_scale_ema"]["pooled"]["per_class"][label]
        lines.append(f"| {label} | {f(v['precision'])} | {f(v['recall'])} | {f(v['f1'])} |")
    lines += [
        "",
        "### 事件结果",
        "",
        "| 事件 | n | Accuracy | Macro-F1 | Balanced Acc | Macro-AP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for event, v in results["v3_scale_ema"]["per_event"].items():
        lines.append(f"| {event} | {v['n']} | {f(v['accuracy'])} | {f(v['macro_f1'])} | {f(v['balanced_accuracy'])} | {f(v['macro_ap'])} |")
    lines += [
        "",
        "| 事件 | v2 Macro-F1 | v3-scale Macro-F1 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for event in results["v3_scale_ema"]["per_event"]:
        v2 = results["v2_ema"]["per_event"][event]["macro_f1"]
        v3 = results["v3_scale_ema"]["per_event"][event]["macro_f1"]
        lines.append(f"| {event} | {f(v2)} | {f(v3)} | {f(v3 - v2)} |")
    matrix = results["v3_scale_ema"]["pooled"]["confusion"]
    lines += [
        "",
        "### v3-scale EMA 混淆矩阵（actual 行 / predicted 列）",
        "",
        "| actual\\pred | no_damage | minor | major | destroyed |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in zip(LABELS, matrix):
        lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")
    lines += [
        "",
        "## 训练曲线",
        "",
        "| epoch | train loss | val Macro-F1 | val Balanced Acc | val Macro-AP |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in history:
        lines.append(
            f"| {item['epoch']} | {f(item['train_loss'])} | {f(item['val']['macro_f1'])} | {f(item['val']['balanced_accuracy'])} | {f(item['val']['macro_ap'])} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "相对 v2，relative-scale crop 提高了 Balanced Accuracy 和 minor F1，但 pooled Macro-F1 下降 0.0053，Matthew Macro-F1 下降 0.0034；Moore 只提升 0.0031。它没有达到预设成功标准（Matthew ≥0.38、pooled ≥0.515），因此不能替代 v2 作为主 B1。下一步应优先去掉随机 scale jitter 做纯 relative-margin 消融，或直接转向 pre/post 双时相输入与标签/配准审计。",
        "",
    ]
    (RUN / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"best_epoch": best["epoch"], "best": best["val"], "event": results["v3_scale_ema"]["per_event"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
