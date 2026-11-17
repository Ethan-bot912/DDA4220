"""One-epoch CPU smoke test for the optimized training entrypoint."""

from __future__ import annotations

import csv
import gc
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train  # noqa: E402


class TrainSmokeTest(unittest.TestCase):
    def test_one_epoch_cpu(self) -> None:
        with tempfile.TemporaryDirectory(prefix="optimized_b2_train_") as raw:
            root = Path(raw)
            profile = {
                "driver": "GTiff",
                "height": 32,
                "width": 32,
                "count": 3,
                "dtype": "uint8",
                "transform": from_origin(0, 32, 1, 1),
            }
            rows = []
            for index, split in enumerate(("train", "train", "val")):
                path_pre = root / f"pre_{index}.tif"
                path_post = root / f"post_{index}.tif"
                array = np.full((3, 32, 32), 80 + index * 20, dtype=np.uint8)
                for path in (path_pre, path_post):
                    with rasterio.open(path, "w", **profile) as handle:
                        handle.write(array)
                rows.append({
                    "split": split,
                    "event": "synthetic",
                    "tile_id": f"tile_{index}",
                    "building_index": str(index),
                    "damage": ("no_damage" if index == 0 else "minor"),
                    "pre_image": str(path_pre),
                    "post_image": str(path_post),
                    "post_bbox_xmin": "8",
                    "post_bbox_ymin": "8",
                    "post_bbox_xmax": "20",
                    "post_bbox_ymax": "20",
                })
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            config = {
                "experiment_id": "synthetic_smoke",
                "project_root": str(root),
                "manifest": str(manifest),
                "output_dir": str(root / "run"),
                "train_split": "train",
                "val_split": "val",
                "model": "dual_stream_swin_t",
                "pretrained": False,
                "num_classes": 4,
                "crop_size": 224,
                "context": 2,
                "alignment": {"enabled": False},
                "radiometric": {"enabled": False},
                "augmentation": {"enabled": False},
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,
                "backbone_learning_rate": 0.00001,
                "head_learning_rate": 0.0001,
                "weight_decay": 0.01,
                "warmup_epochs": 0,
                "freeze_backbone_epochs": 0,
                "head_dropout": 0.0,
                "ema": False,
                "class_weight_power": 0.0,
                "event_sampler": {"enabled": False},
                "hard_negative_file": "",
                "loss": {"ordinal_weight": 0.1, "damage_weight": 0.1, "distill_weight": 0.0},
                "gate": {"enabled": True, "initial_bias": -3.0},
                "seed": 42,
                "device": "cpu",
                "num_workers": 0,
                "pin_memory": False,
                "amp": False,
                "channels_last": False,
                "deterministic": True,
                "max_train": 0,
                "max_val": 0,
                "early_stopping_patience": 0,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(sys, "argv", ["train.py", "--config", str(config_path)]):
                self.assertEqual(train.main(), 0)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "complete")
            self.assertTrue((root / "run" / "predictions_val.csv").is_file())
            gc.collect()


if __name__ == "__main__":
    unittest.main()
