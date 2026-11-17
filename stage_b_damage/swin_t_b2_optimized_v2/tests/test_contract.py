"""Fast contracts for the optimized B2 modules; no real dataset is accessed."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import OptimizedPairedDataset, warp_integer  # noqa: E402
from model import DualStreamSwinT, load_b1_warm_start  # noqa: E402


class OptimizedB2Contracts(unittest.TestCase):
    def test_integer_warp_does_not_wrap(self) -> None:
        image = torch.zeros(3, 8, 8)
        image[:, 2:4, 2:4] = 1.0
        shifted, mask = warp_integer(image, 2, -1)
        self.assertEqual(tuple(shifted.shape), (3, 8, 8))
        self.assertEqual(tuple(mask.shape), (1, 8, 8))
        self.assertEqual(float(shifted[:, 1:3, 4:6].mean()), 1.0)
        self.assertEqual(float(shifted[:, -1, :].sum()), 0.0)

    def test_dual_stream_forward_contract(self) -> None:
        torch.manual_seed(3)
        model = DualStreamSwinT(pretrained=False)
        output = model(
            torch.randn(1, 3, 224, 224),
            torch.randn(1, 3, 224, 224),
            torch.rand(1, 4),
            torch.ones(1, 1, 224, 224),
        )
        self.assertEqual(tuple(output["logits"].shape), (1, 4))
        self.assertEqual(tuple(output["ordinal"].shape), (1, 3))
        self.assertEqual(tuple(output["damage"].shape), (1,))
        self.assertEqual(tuple(output["gate"].shape), (1, 1))

    def test_b1_classifier_head_warm_start_maps_to_post_head(self) -> None:
        model = DualStreamSwinT(pretrained=False)
        weight = torch.full_like(model.post_head.weight, 0.125)
        bias = torch.full_like(model.post_head.bias, -0.25)
        with tempfile.TemporaryDirectory(prefix="optimized_b2_head_") as raw:
            checkpoint = Path(raw) / "b1.pt"
            torch.save({"model": {"head.weight": weight, "head.bias": bias}}, checkpoint)
            report = load_b1_warm_start(model, checkpoint)
        self.assertTrue(report["head_loaded"])
        self.assertEqual(report["head_loaded_keys"], ["post_head.weight", "post_head.bias"])
        self.assertTrue(torch.equal(model.post_head.weight, weight))
        self.assertTrue(torch.equal(model.post_head.bias, bias))

    def test_dataset_returns_quality_and_pair(self) -> None:
        with tempfile.TemporaryDirectory(prefix="optimized_b2_") as raw:
            root = Path(raw)
            profile = {
                "driver": "GTiff",
                "height": 32,
                "width": 32,
                "count": 3,
                "dtype": "uint8",
                "transform": from_origin(0, 32, 1, 1),
            }
            data = np.zeros((3, 32, 32), dtype=np.uint8)
            data[0] = 40
            data[1] = 100
            data[2] = 180
            paths = []
            for name in ("pre.tif", "post.tif"):
                path = root / name
                with rasterio.open(path, "w", **profile) as handle:
                    handle.write(data)
                paths.append(path)
            manifest = root / "manifest.csv"
            row = {
                "split": "train",
                "event": "synthetic",
                "tile_id": "tile",
                "building_index": "0",
                "damage": "minor",
                "pre_image": str(paths[0]),
                "post_image": str(paths[1]),
                "post_bbox_xmin": "8",
                "post_bbox_ymin": "8",
                "post_bbox_xmax": "20",
                "post_bbox_ymax": "20",
            }
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            dataset = OptimizedPairedDataset(
                manifest,
                "train",
                project_root=root,
                crop_size=16,
                context=2,
                alignment={"enabled": True, "max_shift": 2, "downsample": 2},
                radiometric={"enabled": True},
                augmentation={"enabled": False},
            )
            item = dataset[0]
            self.assertEqual(tuple(item["pre"].shape), (3, 16, 16))
            self.assertEqual(tuple(item["post"].shape), (3, 16, 16))
            self.assertEqual(tuple(item["valid_mask"].shape), (1, 16, 16))
            self.assertEqual(tuple(item["quality"].shape), (4,))
            self.assertEqual(int(item["label"]), 1)
            dataset.close()


if __name__ == "__main__":
    unittest.main()
