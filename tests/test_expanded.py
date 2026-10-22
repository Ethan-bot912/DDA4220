"""Synthetic verification of epoch-boundary resume and expanded data safeguards."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin
from stage_b_damage.dataset import LABEL_TO_ID, XBDPairedDataset
from tools.train_expanded import preflight
from tools.compare_expanded import compare

ROOT = Path(__file__).resolve().parents[1]


class ExpandedTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "output")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(123)
        rows = []
        for split, n in [("train", 8), ("val", 4)]:
            paths = {}
            for phase in ("pre", "post"):
                path = self.root / f"{split}_{phase}.tif"
                with rasterio.open(path, "w", driver="GTiff", height=64, width=64, count=3,
                                   dtype="int16", transform=from_origin(0, 64, 1, 1), nodata=-99) as dst:
                    dst.write(rng.integers(0, 256, (3, 64, 64), dtype=np.int16))
                paths[phase] = str(path)
            for i in range(n):
                rows.append(dict(split=split, event="synthetic-" + split, tile_id="00000001", building_index=str(i),
                                 damage=list(LABEL_TO_ID)[i % 4], pre_image=paths["pre"], post_image=paths["post"],
                                 post_bbox_xmin=8, post_bbox_ymin=8, post_bbox_xmax=48, post_bbox_ymax=48))
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.config = json.loads((ROOT / "configs/expanded_seed42.json").read_text())
        self.config.update(manifest=str(self.manifest), train_events=["synthetic-train"], val_events=["synthetic-val"],
                           expected_train_rows=8, expected_val_rows=4, expected_train_counts={k: 2 for k in LABEL_TO_ID},
                           epochs=2, batch_size=4, crop_size=32, max_open_files=1)
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config))

    def command(self, out, *extra, success=True, mode="post"):
        result = subprocess.run([sys.executable, "-m", "tools.train_expanded", "--config", str(self.config_path),
                                 "--mode", mode, "--output-dir", str(out), "--device", "cuda", *extra],
                                cwd=ROOT, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_resume_is_identical_to_uninterrupted_cuda(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA acceptance required on this machine")
        for mode in ("post", "early"):
            full, resumed = self.root / (mode + "_full"), self.root / (mode + "_resumed")
            self.command(full, mode=mode)
            self.command(resumed, "--stop-after-epoch", "1", mode=mode)
            paused = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(paused["epoch"], 1)
            self.assertEqual(json.loads((resumed / "status.json").read_text())["status"], "paused")
            self.command(resumed, "--resume", mode=mode)
            a = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
            b = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
            for name in a["model"]:
                self.assertTrue(torch.equal(a["model"][name], b["model"][name]), f"{mode}: {name}")
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertTrue(torch.equal(a["rng"]["loader"], b["rng"]["loader"]))
            for x, y in zip(a["history"], b["history"]):
                for key in ("train_loss", "val", "per_event", "lr"):
                    self.assertEqual(x[key], y[key], (mode, key))
            self.assertEqual(a["best_epoch"], b["best_epoch"])
            self.assertEqual(len((resumed / "history.jsonl").read_text().splitlines()), 2)
            self.command(resumed, success=False, mode=mode)
            self.config["batch_size"] = 2
            self.config_path.write_text(json.dumps(self.config))
            result = self.command(resumed, "--resume", success=False, mode=mode)
            self.assertIn("config changed", result.stderr)
            self.config["batch_size"] = 4
            self.config_path.write_text(json.dumps(self.config))

    def test_data_counts_and_bounded_handles(self):
        preflight(self.config, self.root, self.manifest)
        bad = dict(self.config, expected_train_rows=9)
        with self.assertRaises(ValueError):
            preflight(bad, self.root, self.manifest)
        one = XBDPairedDataset(self.manifest, "train", crop_size=32, max_open_files=1)
        many = XBDPairedDataset(self.manifest, "train", crop_size=32, max_open_files=64)
        for index in (0, 3, 6):
            a, b = one[index], many[index]
            self.assertLessEqual(len(one._handles), 1)
            for key in ("pre", "post", "label"):
                self.assertTrue(torch.equal(a[key], b[key]))
        one.close()
        many.close()

    def test_cross_machine_comparison_guards(self):
        import copy
        score = dict(accuracy=0.5, macro_f1=0.4, balanced_accuracy=0.3)
        base = dict(mode="post", status="complete", completed_epochs=20,
                    config=dict(epochs=20, max_train=0, max_val=0), manifest_sha256="x",
                    train_indices_sha256="a", val_indices_sha256="b", code_sha256={}, versions={}, events={},
                    best_val=score, best_per_event={"moore": score}, hardware="GPU A", training_seconds=1,
                    best_epoch=3, test_evaluated=False)
        second = copy.deepcopy(base)
        second.update(mode="early", hardware="GPU B")
        self.assertEqual(compare(base, second)["pooled"]["B2_minus_B1"]["macro_f1"], 0)
        second["manifest_sha256"] = "other"
        with self.assertRaises(ValueError):
            compare(base, second)
        second["manifest_sha256"] = "x"
        second["status"] = "paused"
        with self.assertRaises(ValueError):
            compare(base, second)


if __name__ == "__main__":
    unittest.main()
