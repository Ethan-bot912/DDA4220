"""Compare completed expanded B1/B2 summaries across sessions or computers.

Location/session differences are allowed; protocol, samples, training code and
software versions must match. Hardware times remain separate observations.
"""
import argparse
import csv
import json
from pathlib import Path


def compare(b1, b2):
    if (b1["mode"], b2["mode"]) != ("post", "early"):
        raise ValueError("Expected B1=post, B2=early")
    for row in (b1, b2):
        if row["status"] != "complete" or row["completed_epochs"] != row["config"]["epochs"]:
            raise ValueError("Both runs must finish the planned epochs")
        if row["config"]["max_train"] or row["config"]["max_val"] or row["test_evaluated"]:
            raise ValueError("Smoke runs or test-selected runs cannot enter this comparison")
    for key in ("config", "manifest_sha256", "train_indices_sha256", "val_indices_sha256", "code_sha256", "versions", "events"):
        if b1[key] != b2[key]:
            raise ValueError(f"Uncontrolled comparison: {key} differs")
    def scores(x, y):
        return {"B1": x, "B2": y, "B2_minus_B1": {
            k: y[k] - x[k] for k in ("accuracy", "macro_f1", "balanced_accuracy")}}
    if set(b1["best_per_event"]) != set(b2["best_per_event"]):
        raise ValueError("Validation event sets differ")
    return {"scope": "single-seed best pooled validation selection; not held-out test performance",
            "manifest_sha256": b1["manifest_sha256"], "config": b1["config"],
            "pooled": scores(b1["best_val"], b2["best_val"]),
            "per_event": {e: scores(b1["best_per_event"][e], b2["best_per_event"][e]) for e in b1["best_per_event"]},
            "resources": {"B1": {k: b1[k] for k in ("hardware", "training_seconds", "best_epoch")},
                          "B2": {k: b2[k] for k in ("hardware", "training_seconds", "best_epoch")}},
            "hardware_note": "Sessions and system load may differ; recorded timings are not a controlled architecture speed comparison. See each run's hardware metadata.",
            "test_evaluated": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b1", type=Path, required=True)
    p.add_argument("--b2", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix(".csv").exists():
        raise FileExistsError("Choose a new comparison output")
    result = compare(json.loads(a.b1.read_text()), json.loads(a.b2.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with a.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "metric", "B1", "B2", "B2_minus_B1"])
        for scope, data in [("pooled", result["pooled"]), *result["per_event"].items()]:
            for metric, delta in data["B2_minus_B1"].items():
                writer.writerow([scope, metric, data["B1"][metric], data["B2"][metric], delta])
    print(json.dumps(result["pooled"]["B2_minus_B1"]))


if __name__ == "__main__":
    main()
