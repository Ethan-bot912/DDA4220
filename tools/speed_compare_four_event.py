"""Controlled four-event speed comparison: workers=0 vs workers=4.

Everything except ``--workers`` is identical to the published four-event
protocol (stage_b_damage/train_baseline.py, ResNet-18, random init, seed 42,
batch 32, lr 1e-3, crop 128, context 16, cosine schedule, val-MacroF1 selection).

Stages
  A  workers=0, 2 epochs, both modes   -> baseline timing on this machine
  B  workers=4, 2 epochs, both modes   -> same config, so epoch-level metrics
                                          must match A exactly
  C  workers=4, 20 epochs, both modes  -> full protocol, compared with the
                                          published four-event reference

The stage list, wall-clock times and a status file are written under
``output/speed_four_event`` so the run can be watched while it is executing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "speed_four_event"
MANIFEST = ROOT / "dataset" / "pilot_tier3_4events" / "manifest_classification.csv"

STAGES = [
    ("A_post_e2_w0", "post", 2, 0),
    ("A_early_e2_w0", "early", 2, 0),
    ("B_post_e2_w4", "post", 2, 4),
    ("B_early_e2_w4", "early", 2, 4),
    ("C_post_e20_w4", "post", 20, 4),
    ("C_early_e20_w4", "early", 20, 4),
]


def write_status(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_stage(name: str, mode: str, epochs: int, workers: int) -> dict:
    output = OUT / f"{name}.json"
    command = [
        sys.executable,
        "-m",
        "stage_b_damage.train_baseline",
        "--manifest",
        str(MANIFEST),
        "--project-root",
        str(ROOT),
        "--output",
        str(output),
        "--mode",
        mode,
        "--backbone",
        "resnet18",
        "--epochs",
        str(epochs),
        "--batch-size",
        "32",
        "--lr",
        "0.001",
        "--crop-size",
        "128",
        "--context",
        "16",
        "--workers",
        str(workers),
        "--seed",
        "42",
        "--device",
        "cuda",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT)
    elapsed = time.perf_counter() - started
    result = {
        "stage": name,
        "mode": mode,
        "epochs": epochs,
        "workers": workers,
        "returncode": completed.returncode,
        "wall_seconds": elapsed,
        "output": str(output),
    }
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        result["training_seconds"] = payload.get("training_seconds")
        result["epoch_seconds"] = [row.get("epoch_seconds") for row in payload.get("history", [])]
        result["best_epoch"] = payload.get("best_epoch")
        result["best_macro_f1"] = (payload.get("best_val") or {}).get("macro_f1")
    return result


def compare() -> dict:
    def load(name: str) -> dict:
        return json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))

    result: dict = {}
    for mode, a_name, b_name in (
        ("post", "A_post_e2_w0", "B_post_e2_w4"),
        ("early", "A_early_e2_w0", "B_early_e2_w4"),
    ):
        a, b = load(a_name), load(b_name)
        rows = []
        for left, right in zip(a["history"], b["history"]):
            rows.append(
                {
                    "epoch": left["epoch"],
                    "train_loss_absdiff": abs(left["train_loss"] - right["train_loss"]),
                    "macro_f1_absdiff": abs(left["val"]["macro_f1"] - right["val"]["macro_f1"]),
                    "confusion_equal": left["val"]["confusion"] == right["val"]["confusion"],
                }
            )
        a_epoch = [r["epoch_seconds"] for r in a["history"]]
        b_epoch = [r["epoch_seconds"] for r in b["history"]]
        mean_a = sum(a_epoch) / len(a_epoch)
        mean_b = sum(b_epoch) / len(b_epoch)
        result[mode] = {
            "baseline_epoch_seconds": a_epoch,
            "fast_epoch_seconds": b_epoch,
            "baseline_mean_epoch_seconds": mean_a,
            "fast_mean_epoch_seconds": mean_b,
            "epoch_speedup": mean_a / mean_b if mean_b else None,
            "equivalence": rows,
            "bit_identical": all(
                row["train_loss_absdiff"] == 0.0 and row["macro_f1_absdiff"] == 0.0
                and row["confusion_equal"]
                for row in rows
            ),
        }

    reference = {
        "post": {"best_epoch": 5, "macro_f1": 0.601235705274549},
        "early": {"best_epoch": 9, "macro_f1": 0.6022778098},
    }
    for mode, name in (("post", "C_post_e20_w4"), ("early", "C_early_e20_w4")):
        payload = load(name)
        result[mode + "_full"] = {
            "best_epoch": payload["best_epoch"],
            "best_macro_f1": payload["best_val"]["macro_f1"],
            "accuracy": payload["best_val"]["accuracy"],
            "balanced_accuracy": payload["best_val"]["balanced_accuracy"],
            "training_seconds": payload["training_seconds"],
            "reference": reference[mode],
            "macro_f1_delta_vs_reference": payload["best_val"]["macro_f1"] - reference[mode]["macro_f1"],
        }
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    history = []
    write_status({"state": "running", "stages": STAGES, "completed": history, "started": time.time()})
    for name, mode, epochs, workers in STAGES:
        write_status(
            {
                "state": "running",
                "current": name,
                "stages": STAGES,
                "completed": history,
                "started": time.time(),
            }
        )
        record = run_stage(name, mode, epochs, workers)
        history.append(record)
        print(json.dumps(record), flush=True)
        if record["returncode"] != 0:
            write_status({"state": "failed", "failed_stage": name, "completed": history})
            return record["returncode"]

    summary = {"state": "complete", "completed": history, "comparison": compare()}
    write_status(summary)
    (OUT / "comparison.json").write_text(
        json.dumps(summary["comparison"], indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["comparison"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
