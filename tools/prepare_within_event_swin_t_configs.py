"""Generate the frozen B1/B2-simple/B2-4 configs for the within-event 7-event study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EVENTS = [
    "hurricane-harvey",
    "hurricane-matthew",
    "hurricane-michael",
    "joplin-tornado",
    "lower-puna-volcano",
    "moore-tornado",
    "tuscaloosa-tornado",
]
SEEDS = (42, 2026, 3407)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def apply_split_contract(config: dict, split_report: dict, seed: int) -> None:
    fields = split_report["config_fields"]
    pooled = split_report["pooled"]
    config.update(
        {
            "manifest": "dataset/within_event_7events/manifest_classification.csv",
            "manifest_sha256": split_report["outputs"]["manifest_classification.csv"]["sha256"],
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "train_events": list(EVENTS),
            "val_events": list(EVENTS),
            "test_events": list(EVENTS),
            "expected_train_rows": fields["expected_train_rows"],
            "expected_val_rows": fields["expected_val_rows"],
            "expected_test_rows": fields["expected_test_rows"],
            "expected_train_counts": pooled["train"]["class_counts"],
            "expected_val_counts": pooled["val"]["class_counts"],
            "expected_test_counts": pooled["test"]["class_counts"],
            "seed": int(seed),
            "eval_test": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-report",
        type=Path,
        default=Path("dataset/within_event_7events/split_report.json"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    split_report = read_json((root / args.split_report).resolve())
    if not split_report["acceptance"]["passed"]:
        raise RuntimeError("Within-event split did not pass acceptance checks")
    config_dir = (root / args.config_dir).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)

    b1_base = read_json(config_dir / "swin_t_expanded_b1_seed42_bs16.json")
    simple_base = read_json(config_dir / "swin_t_expanded_b2_early_seed42_bs16.json")
    optimized_base = read_json(config_dir / "swin_t_b2_optimized_v2_seed42.json")
    generated = []

    for seed in SEEDS:
        b1 = json.loads(json.dumps(b1_base))
        apply_split_contract(b1, split_report, seed)
        b1["experiment_id"] = f"within_event_7events_b1_seed{seed}_bs16_v1"
        b1["output_dir"] = f"../runs/within_event_7events/b1/seed{seed}"
        b1_path = config_dir / f"within_event_7events_b1_seed{seed}.json"
        write_json(b1_path, b1)
        generated.append(b1_path)

        simple = json.loads(json.dumps(simple_base))
        apply_split_contract(simple, split_report, seed)
        simple["experiment_id"] = f"within_event_7events_b2_simple_seed{seed}_bs16_v1"
        simple["output_dir"] = f"../runs/within_event_7events/b2_simple/seed{seed}"
        simple["b1_reference"] = {
            "config": f"within_event_7events_b1_seed{seed}.json",
            "checkpoint": f"../runs/within_event_7events/b1/seed{seed}/best.pt",
            "selection_metric": "validation pooled macro_f1",
            "status": "pending_same_seed_b1_run",
        }
        simple_path = config_dir / f"within_event_7events_b2_simple_seed{seed}.json"
        write_json(simple_path, simple)
        generated.append(simple_path)

        optimized = json.loads(json.dumps(optimized_base))
        apply_split_contract(optimized, split_report, seed)
        optimized["experiment_id"] = f"within_event_7events_b2_4_seed{seed}_v1"
        optimized["output_dir"] = f"../runs/within_event_7events/b2_4/seed{seed}"
        optimized["b1_checkpoint"] = (
            f"../runs/within_event_7events/b1/seed{seed}/best.pt"
        )
        optimized["hard_negative_file"] = ""
        optimized["hard_negative_bootstrap"] = False
        optimized["hard_negative_source"] = (
            "deterministic_train_only_manifest_fallback; regenerate from current-train OOF when available"
        )
        optimized["calibration_fraction"] = 0
        optimized_path = config_dir / f"within_event_7events_b2_4_seed{seed}.json"
        write_json(optimized_path, optimized)
        generated.append(optimized_path)

    print(json.dumps([p.relative_to(root).as_posix() for p in generated], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
