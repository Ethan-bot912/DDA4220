"""Contracts for the no-calibration/default hard-negative pipeline."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from run_full_pipeline import prepare_hard_negative_file  # noqa: E402
from split_calibration import make_split  # noqa: E402


class PipelineContracts(unittest.TestCase):
    def _manifest(self, path: Path) -> None:
        fields = ["split", "event", "tile_id", "building_index", "damage"]
        rows = [
            {"split": "train", "event": "e", "tile_id": "t", "building_index": "0", "damage": "minor"},
            {"split": "train", "event": "e", "tile_id": "t", "building_index": "1", "damage": "major"},
            {"split": "val", "event": "v", "tile_id": "t", "building_index": "0", "damage": "minor"},
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_zero_fraction_preserves_train(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pipeline_contract_") as raw:
            root = Path(raw)
            source = root / "source.csv"
            destination = root / "copy.csv"
            self._manifest(source)
            report = make_split(source, destination, 0.0, 42)
            self.assertEqual(report["counts"]["train"], 2)
            self.assertNotIn("calibration", report["counts"])

    def test_hard_negative_filter_excludes_validation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pipeline_contract_") as raw:
            root = Path(raw)
            source = root / "source.csv"
            self._manifest(source)
            input_csv = root / "input.csv"
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["key", "reason"])
                writer.writeheader()
                writer.writerow({"key": "e|t|0", "reason": "error"})
                writer.writerow({"key": "v|t|0", "reason": "leak"})
            output = root / "hard.csv"
            # Use the pipeline helper with an explicit key file.  The source
            # manifest is copied with fraction=0 to establish the train set.
            current = root / "current.csv"
            make_split(source, current, 0.0, 42)
            path, report = prepare_hard_negative_file(
                source_manifest=source,
                current_manifest=current,
                output_dir=root,
                config_dir=root,
                explicit_input=input_csv,
                disabled=False,
                fallback_fraction=0.25,
                seed=42,
            )
            self.assertEqual(path.name, "hard_negatives.csv")
            self.assertEqual(report["filtered"]["selected"], 1)
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["key"] for row in rows], ["e|t|0"])


if __name__ == "__main__":
    unittest.main()
