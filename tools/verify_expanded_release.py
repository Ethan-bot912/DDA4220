"""Check published expanded records and frozen sources with the standard library.

Does not load weights, read raw images, train, or perform inference. For the
complete ZIP and optional checkpoint loading, use its tools/verify_delivery.py.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_expanded import compare

REPORT = ROOT / "reports/expanded_seed42"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = read(ROOT / "configs/expanded_seed42.json")
    expected_files = read(REPORT / "record_file_hashes.json")
    for name, expected in expected_files.items():
        path = (REPORT / name).resolve()
        require(path.is_relative_to(REPORT.resolve()), "Unsafe record path")
        require(sha(path) == expected, "Modified record: " + name)
    rows = []
    checks = read(REPORT / "checkpoint_verification.json")
    for mode in ("b1", "b2"):
        folder = REPORT / mode
        row = read(folder / "summary.json")
        metadata = read(folder / "run_metadata.json")
        status = read(folder / "status.json")
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()]
        require(row["config"] == config, mode + ": config mismatch")
        require(row["history"] == history, mode + ": history mismatch")
        require([entry["epoch"] for entry in history] == list(range(1, 21)), mode + ": incomplete history")
        require(status["status"] == "complete", mode + ": incomplete status")
        for key in ("code_sha256", "manifest_sha256", "train_indices_sha256", "val_indices_sha256", "versions"):
            require(row[key] == metadata[key], mode + ": metadata mismatch: " + key)
        for name, expected in row["code_sha256"].items():
            require(sha(ROOT / name) == expected, "Modified frozen source (including line endings): " + name)
        require(checks[mode]["last_epoch"] == 20, mode + ": checkpoint verification incomplete")
        require(checks[mode]["best_epoch"] == row["best_epoch"], mode + ": best epoch mismatch")
        require(checks[mode]["checkpoint_loads_ok"] and checks[mode]["history_matches"] and
                checks[mode]["frozen_code_config_manifest_ok"], mode + ": previous checkpoint verification failed")
        rows.append(row)
    actual = compare(*rows)
    published = read(REPORT / "comparison.json")
    for key in ("pooled", "per_event", "manifest_sha256", "config", "resources", "test_evaluated"):
        require(actual[key] == published[key], "Comparison mismatch: " + key)
    with (REPORT / "validation_images.csv").open(encoding="utf-8-sig", newline="") as handle:
        require(len(list(csv.DictReader(handle))) == 1186, "Incorrect validation image count")
    print(json.dumps({"result": "PASS", "completed_epochs": [20, 20],
                      "verified_record_files": len(expected_files),
                      "frozen_source_sha256_matches": True, "test_evaluated": False,
                      "checkpoint_loading_performed_now": False}))


if __name__ == "__main__":
    main()
