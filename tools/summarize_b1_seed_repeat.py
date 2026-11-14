"""Summarize and compare two exported Swin-T B1 validation prediction files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stage_b_damage.swin_t_expanded.metrics import LABELS, metrics_from_predictions


def load_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> dict:
    label_to_id = {name: index for index, name in enumerate(LABELS)}

    def metrics(selected: list[dict[str, str]]) -> dict:
        return metrics_from_predictions(
            [label_to_id[row["actual"]] for row in selected],
            [label_to_id[row["predicted"]] for row in selected],
            [[float(row[f"p_{name}"]) for name in LABELS] for row in selected],
        )

    events = sorted({row["event"] for row in rows})
    return {
        "pooled": metrics(rows),
        "per_event": {
            event: metrics([row for row in rows if row["event"] == event])
            for event in events
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    new = summarize(load_predictions(args.new))
    old = summarize(load_predictions(args.old))
    fields = ("accuracy", "macro_f1", "balanced_accuracy", "macro_ap")
    result = {
        "new": new,
        "old": old,
        "pooled_delta": {
            field: new["pooled"][field] - old["pooled"][field]
            for field in fields
        },
        "event_delta": {
            event: {
                field: new["per_event"][event][field] - old["per_event"][event][field]
                for field in fields
            }
            for event in new["per_event"]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
