"""Validation-only export and strict paired comparison for class-weight runs.

Outputs belong under output/, not Git. Comparison reads all four predeclared
run directories, including explicit missing/paused/failed status entries.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stage_b_damage.dataset import LABEL_TO_ID
from stage_b_damage.train_baseline import build_model, make_input, metrics_from_confusion
from stage_b_damage.within_event_cache import CachedDataset, verify_cache
from tools.train_class_weight import code_fingerprint
from tools.train_expanded import ROOT, atomic_json, sha256


def export(run, output, root, device):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("Only completed runs may be exported as experiment results")
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    c = saved["config"]
    if saved["code_sha256"] != code_fingerprint() or saved["cache_sha256"] != verify_cache(c, root):
        raise ValueError("Training source or data cache changed")
    for key in ("config", "mode", "initialization_sha256", "cache_sha256", "code_sha256"):
        if saved[key] != summary[key]:
            raise ValueError(f"Checkpoint/summary mismatch: {key}")
    if saved["epoch"] != summary["best_epoch"] or saved["val"] != summary["best_val"]:
        raise ValueError("Checkpoint does not match selected best")
    ds = CachedDataset(root / c["manifest"], "val", project_root=root, crop_size=c["crop_size"],
                       context=c["context"], cache_dir=root / c["cache_dir"], return_meta=True)
    model = build_model(c["backbone"], saved["mode"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    matrices = {"pooled": torch.zeros(4, 4, dtype=torch.long)}
    predictions = []
    try:
        with torch.inference_mode():
            for batch in DataLoader(ds, batch_size=c["batch_size"], shuffle=False):
                probabilities = model(make_input(batch, saved["mode"]).to(device)).softmax(1).cpu()
                for i, (actual, predicted) in enumerate(zip(batch["label"].tolist(), probabilities.argmax(1).tolist())):
                    row = {k: batch["meta"][k][i] for k in ("event", "tile_id", "building_index")}
                    row.update(actual=actual, predicted=predicted, confidence=float(probabilities[i, predicted]))
                    predictions.append(row)
                    matrices.setdefault(row["event"], torch.zeros(4, 4, dtype=torch.long))[actual, predicted] += 1
                    matrices["pooled"][actual, predicted] += 1
    finally:
        ds.close()
    scores = {key: metrics_from_confusion(value) for key, value in matrices.items()}
    if scores["pooled"] != summary["best_val"] or {k: v for k, v in scores.items() if k != "pooled"} != summary["best_per_event"]:
        raise ValueError("Recomputed predictions differ from selected validation scores")
    output.mkdir(parents=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    atomic_json(output / "evaluation.json", {"split": "val", "test_evaluated": False, "scores": scores,
                "checkpoint_sha256": sha256(run / "best.pt"), "summary_sha256": sha256(run / "summary.json"),
                "predictions_sha256": sha256(output / "predictions.csv"), "rows": len(predictions),
                "evaluation_code_sha256": sha256(Path(__file__))})


def validate_pair(a, b):
    for run in (a, b):
        if run["status"] != "complete" or run["completed_epochs"] != run["config"]["epochs"] or run["test_evaluated"]:
            raise ValueError("Incomplete or wrong-scope run")
    for key in ("mode", "manifest_sha256", "split_map_sha256", "data_lock_sha256", "cache_sha256",
                "initialization_sha256", "code_sha256", "versions", "hardware", "train_indices_sha256", "val_indices_sha256"):
        if a[key] != b[key]:
            raise ValueError(f"Unmatched A/B: {key}")
    ca, cb = (dict(x["config"]) for x in (a, b))
    if ca.pop("class_weight_mode") != "inverse_frequency" or cb.pop("class_weight_mode") != "equal":
        raise ValueError("Expected inverse-frequency A and equal B")
    if ca.pop("experiment") != "baseline" or cb.pop("experiment") != "class_weight":
        raise ValueError("Incorrect experiment identity")
    # Attempts are explicit caller-selected run identities, not scientific knobs.
    ca.pop("attempt")
    cb.pop("attempt")
    if ca != cb:
        raise ValueError("A/B configs differ beyond the declared factor")
    if len(a["history"]) != a["completed_epochs"] or len(b["history"]) != b["completed_epochs"]:
        raise ValueError("Incomplete history")
    if [r["sample_order_sha256"] for r in a["history"]] != [r["sample_order_sha256"] for r in b["history"]]:
        raise ValueError("A/B sample orders differ")


def read_predictions(run):
    directory = run / "validation"
    ev = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    if ev["split"] != "val" or ev["test_evaluated"] or ev["summary_sha256"] != sha256(run / "summary.json") or ev["checkpoint_sha256"] != sha256(run / "best.pt") or ev["predictions_sha256"] != sha256(directory / "predictions.csv"):
        raise ValueError("Validation evidence provenance mismatch")
    with (directory / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {(r["event"], r["tile_id"], r["building_index"]): r for r in rows}
    if len(result) != len(rows) or len(rows) != ev["rows"]:
        raise ValueError("Duplicate or missing prediction IDs")
    return result


def diagnostics(rows):
    events = sorted({key[0] for key in rows})
    output = {}
    for event in ["pooled", *events]:
        matrix = torch.zeros(4, 4, dtype=torch.long)
        for key, row in rows.items():
            if event == "pooled" or key[0] == event:
                matrix[int(row["actual"]), int(row["predicted"])] += 1
        checks = {}
        for name, actual, predicted in [("minor_to_destroyed", 1, [3]), ("major_to_minor", 2, [1]),
                ("major_recall", 2, [2]), ("no_damage_to_minor", 0, [1]), ("no_damage_to_damage", 0, [1, 2, 3]),
                ("destroyed_recall", 3, [3]), ("destroyed_to_no_damage_or_minor", 3, [0, 1])]:
            count, support = int(matrix[actual, predicted].sum()), int(matrix[actual].sum())
            checks[name] = {"count": count, "support": support, "rate": count / support if support else None}
        output[event] = {"metrics": metrics_from_confusion(matrix), "checks": checks}
    return output


def paired_errors(a, b):
    if a.keys() != b.keys():
        raise ValueError("Prediction IDs differ")
    counts, groups = {}, {}
    for key in sorted(a):
        x, y = a[key], b[key]
        if x["actual"] != y["actual"]:
            raise ValueError("Labels differ")
        correct_a, correct_b = x["actual"] == x["predicted"], y["actual"] == y["predicted"]
        state = {(True, True): "both_correct", (False, False): "both_wrong",
                 (True, False): "new_error", (False, True): "corrected"}[(correct_a, correct_b)]
        for event in ("pooled", key[0]):
            counts.setdefault(event, dict.fromkeys(("both_correct", "both_wrong", "new_error", "corrected"), 0))[state] += 1
        if state in ("new_error", "corrected"):
            group = "/".join((key[0], state, x["actual"], x["predicted"], y["predicted"]))
            groups.setdefault(group, []).append({"id": list(key), "actual": x["actual"], "a": x["predicted"], "b": y["predicted"]})
    # Predeclared selection: 5 lowest SHA256 IDs per event/state/true/A/B class.
    samples = {group: sorted(values, key=lambda r: hashlib.sha256(json.dumps(r["id"]).encode()).hexdigest())[:5]
               for group, values in sorted(groups.items())}
    return {"counts": counts, "case_selection": "lowest SHA256 ID; <=5 per event/state/true/A/B class", "cases": samples}


def compare_runs(paths, output):
    if output.exists():
        raise FileExistsError(output)
    summaries, statuses, pairs = {}, {}, {}
    for name, path in paths.items():
        if (path / "RUNNING.lock").exists():
            statuses[name] = "running_or_stale_lock"
        elif (path / "summary.json").exists():
            summaries[name] = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            statuses[name] = summaries[name]["status"]
            if (path / "status.json").exists():
                current = json.loads((path / "status.json").read_text(encoding="utf-8"))["status"]
                if current != statuses[name]:
                    statuses[name] = current
        else:
            statuses[name] = "failed" if (path / "failure.json").exists() else "missing"
    for model in ("b1", "b2"):
        an, bn = "a_" + model, "b_" + model
        if statuses[an] != "complete" or statuses[bn] != "complete":
            continue
        a, b = summaries[an], summaries[bn]
        validate_pair(a, b)
        if a["mode"] != {"b1": "post", "b2": "early"}[model]:
            raise ValueError("Model path mapping is incorrect")
        ap, bp = read_predictions(paths[an]), read_predictions(paths[bn])
        if len(ap) != a["val_rows"] or len(bp) != b["val_rows"]:
            raise ValueError("Incomplete validation predictions")
        pairs[model] = {"a": diagnostics(ap), "b": diagnostics(bp), "paired": paired_errors(ap, bp),
                       "B_minus_A": {k: b["best_val"][k] - a["best_val"][k] for k in ("accuracy", "macro_f1", "balanced_accuracy")}}
    result = {"statuses": statuses, "pairs": pairs, "test_evaluated": False,
              "scope": "single-seed exploratory validation; no significance or stability claim",
              "sources": {k: str(v.resolve()) for k, v in paths.items()},
              "failure_records": {k: sorted(p.name for p in v.glob("failure_*.json")) for k, v in paths.items()}}
    if len(pairs) == 2:
        for key in ("manifest_sha256", "split_map_sha256", "data_lock_sha256", "cache_sha256", "code_sha256",
                    "versions", "hardware", "train_indices_sha256", "val_indices_sha256"):
            if summaries["a_b1"][key] != summaries["a_b2"][key]:
                raise ValueError(f"Unmatched B1/B2 provenance: {key}")
        if {k: v for k, v in summaries["a_b1"]["config"].items() if k != "attempt"} != {k: v for k, v in summaries["a_b2"]["config"].items() if k != "attempt"}:
            raise ValueError("B1/B2 configs differ")
        result["B2_minus_B1"] = {condition: {k: summaries[condition + "_b2"]["best_val"][k] - summaries[condition + "_b1"]["best_val"][k]
                                for k in ("accuracy", "macro_f1", "balanced_accuracy")} for condition in ("a", "b")}
        result["gap_change"] = {k: result["B2_minus_B1"]["b"][k] - result["B2_minus_B1"]["a"][k] for k in result["B2_minus_B1"]["a"]}
    output.mkdir(parents=True)
    atomic_json(output / "comparison.json", result)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "status", "seed", "attempt", "best_epoch", "val_macro_f1"])
        for name, status in statuses.items():
            run = summaries.get(name, {})
            writer.writerow([name, status, run.get("config", {}).get("seed"), run.get("config", {}).get("attempt"),
                             run.get("best_epoch"), run.get("best_val", {}).get("macro_f1")])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    ev = sub.add_parser("export")
    ev.add_argument("--run", type=Path, required=True)
    ev.add_argument("--project-root", type=Path, default=ROOT)
    ev.add_argument("--device", default="cuda")
    compare = sub.add_parser("compare")
    for name in ("a-b1", "a-b2", "b-b1", "b-b2"):
        compare.add_argument("--" + name, type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "export":
        export(args.run, args.run / "validation", args.project_root.resolve(), torch.device(args.device))
    else:
        compare_runs({k: getattr(args, k) for k in ("a_b1", "a_b2", "b_b1", "b_b2")}, args.output)


if __name__ == "__main__":
    main()
