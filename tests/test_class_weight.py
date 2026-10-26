"""Controlled synthetic demo: no real training, no real test-image access."""
import csv
import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tests import test_expanded
from stage_b_damage.class_weight import class_weights
from stage_b_damage.dataset import LABEL_TO_ID, XBDPairedDataset
from stage_b_damage.within_event_cache import CachedDataset, verify_cache
from tools.build_within_event_cache import build, encode
from tools.train_expanded import sha256
from tools.train_class_weight import code_fingerprint, evaluate_events
from stage_b_damage.train_baseline import build_model
from evaluation.class_weight import export, compare_runs, validate_pair

ROOT = Path(__file__).resolve().parents[1]


class ClassWeightTest(unittest.TestCase):
    def setUp(self):
        test_expanded.ExpandedTest.setUp(self)
        with self.manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        # Nonuniform class counts, so A and B genuinely use different objectives.
        rows[4]["damage"] = "minor"
        rows[6]["damage"] = "no_damage"
        # Sealed image exists but is deliberately invalid: opening/hashing it is forbidden.
        sealed = self.root / "sealed.tif"
        sealed.write_bytes(b"not an image")
        rows.append(dict(rows[0], split="test", event="synthetic-test", tile_id="sealed",
                         pre_image=str(sealed), post_image=str(sealed)))
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        split_map = self.root / "split_map.csv"
        split_map.write_text("synthetic frozen split metadata")
        sources = sorted({r[k] for r in rows if r["split"] != "test" for k in ("pre_image", "post_image")})
        lock = self.root / "data_lock.json"
        lock.write_text(json.dumps({"files": [{"path": p, "sha256": sha256(p)} for p in sources]}))
        self.config.update(class_weight_mode="inverse_frequency", experiment="baseline", attempt="attempt01",
            workers=2, prefetch_factor=2, persistent_workers=False, pin_memory=True,
            cache_dir=str(self.root / "cache"), cache_dtype="uint8", expected_train_counts=dict(zip(LABEL_TO_ID, (2,3,1,2))),
            expected_test_rows=1, test_events=["synthetic-test"],
            split_map=str(split_map), data_lock=str(lock))
        for key in ("manifest", "split_map", "data_lock"):
            self.config[key + "_sha256"] = sha256(self.config[key])
        self.save()

    def save(self):
        self.config_path.write_text(json.dumps(self.config))

    def command(self, output, *extra, mode="post", success=True):
        result = subprocess.run([sys.executable, "-m", "tools.train_class_weight", "--config", str(self.config_path),
            "--mode", mode, "--output-dir", str(output), "--device", "cuda", *extra], cwd=ROOT, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_loss_math_and_fractional_cache(self):
        counts = dict(zip(LABEL_TO_ID, (2,3,1,2)))
        w = class_weights(counts, "inverse_frequency")
        expected = torch.tensor([1/2,1/3,1,1/2])
        self.assertTrue(torch.equal(w, expected / expected.mean()))
        logits = torch.tensor([[1.,2.,0.,-1.], [2.,0.,1.,-1.], [0.,1.,2.,3.]])
        labels = torch.tensor([0,1,3])
        nll = -logits.log_softmax(1)[torch.arange(3), labels]
        for mode in ("inverse_frequency", "equal"):
            weight = class_weights(counts, mode)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
            self.assertTrue(torch.allclose(loss, (nll * weight[labels]).sum() / weight[labels].sum()))
            numerator = denominator = 0.
            for i in range(3):
                batch_loss = torch.nn.functional.cross_entropy(logits[i:i+1], labels[i:i+1], weight=weight)
                numerator += float(batch_loss) * float(weight[labels[i]])
                denominator += float(weight[labels[i]])
            self.assertAlmostEqual(numerator / denominator, float(loss), places=6)
        with self.assertRaises(ValueError):
            class_weights(counts, "typo")
        fraction = np.array([0.123456], dtype=np.float32)
        with self.assertRaises(ValueError):
            encode(fraction, "uint8")
        self.assertTrue(np.array_equal(encode(fraction, "float32"), fraction))
        self.assertTrue(np.array_equal(encode(np.array([1.], dtype=np.float32), "float32"), [1.]))

    def test_export_deterministic_inference(self):
        build(self.config, self.root)
        digest = verify_cache(self.config, self.root)
        model = build_model("resnet18", "post").cuda()
        ds = CachedDataset(self.manifest, "val", crop_size=32, cache_dir=self.config["cache_dir"], return_meta=True)
        # Also exercise the actual formal worker count, without a formal run.
        four = list(torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4, multiprocessing_context="spawn",
                                               persistent_workers=False, prefetch_factor=2, pin_memory=True))
        zero = list(torch.utils.data.DataLoader(ds, batch_size=4))
        for key in ("pre", "post", "label"):
            self.assertTrue(torch.equal(four[0][key], zero[0][key]))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        metrics, events = evaluate_events(model, torch.utils.data.DataLoader(ds, batch_size=4), torch.device("cuda"), "post")
        ds.close()
        run = self.root / "export_fixture"
        run.mkdir()
        saved = dict(config=self.config, mode="post", initialization_sha256="synthetic", cache_sha256=digest,
                     code_sha256=code_fingerprint(), epoch=1, val=metrics, model=model.state_dict())
        torch.save(saved, run / "best.pt")
        summary = {k: v for k, v in saved.items() if k != "model"}
        summary.update(status="complete", best_epoch=1, best_val=metrics, best_per_event=events)
        (run / "summary.json").write_text(json.dumps(summary))
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        export(run, run / "validation", self.root, torch.device("cuda"))
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_cache_integrity_and_sealed_scope(self):
        original = sha256
        def guarded(path):
            self.assertNotEqual(Path(path).name, "sealed.tif")
            return original(path)
        with patch("stage_b_damage.within_event_cache.sha256", side_effect=guarded):
            build(self.config, self.root)
            verify_cache(self.config, self.root)
        for split in ("train", "val"):
            raw = XBDPairedDataset(self.manifest, split, crop_size=32, context=self.config["context"], return_meta=True)
            cached = CachedDataset(self.manifest, split, crop_size=32, cache_dir=self.config["cache_dir"], return_meta=True)
            for i in range(len(raw)):
                a, b = raw[i], cached[i]
                for key in ("pre", "post", "label"):
                    self.assertTrue(torch.equal(a[key], b[key]))
                self.assertEqual(a["meta"], b["meta"])
            raw.close()
            cached.close()
        with self.assertRaises(ValueError):
            CachedDataset(self.manifest, "test", cache_dir=self.config["cache_dir"])
        with self.assertRaises(FileExistsError):
            build(self.config, self.root)
        with patch("stage_b_damage.within_event_cache.cache_versions", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "versions"):
                verify_cache(self.config, self.root)
        with patch("stage_b_damage.within_event_cache.cache_code", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "cache_code"):
                verify_cache(self.config, self.root)
        changed = dict(self.config, manifest_sha256="wrong")
        with self.assertRaisesRegex(ValueError, "manifest"):
            verify_cache(changed, self.root)
        source = self.root / "train_pre.tif"
        original_bytes = source.read_bytes()
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Source differs"):
            verify_cache(self.config, self.root)
        source.write_bytes(original_bytes)
        float_config = dict(self.config, cache_dtype="float32", cache_dir=str(self.root / "float_cache"))
        build(float_config, self.root)
        verify_cache(float_config, self.root)
        raw = XBDPairedDataset(self.manifest, "train", crop_size=32, context=self.config["context"])
        float_ds = CachedDataset(self.manifest, "train", crop_size=32, cache_dir=float_config["cache_dir"])
        for i in range(len(raw)):
            for key in ("pre", "post"):
                self.assertTrue(torch.equal(raw[i][key], float_ds[i][key]))
        raw.close()
        float_ds.close()
        path = Path(self.config["cache_dir"]) / "train_pre.npy"
        with path.open("r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"x")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_cache(self.config, self.root)

    def test_cuda_pair_resume_and_validation(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA acceptance required")
        build(self.config, self.root)
        paths = {}
        for mode, model in (("post", "b1"), ("early", "b2")):
            for condition, weight in (("a", "inverse_frequency"), ("b", "equal")):
                self.config.update(class_weight_mode=weight, experiment="baseline" if condition == "a" else "class_weight")
                self.save()
                full = self.root / (condition + "_" + model)
                resumed = self.root / (condition + "_" + model + "_resumed")
                print(f"Demo {condition}/{model}: full vs resumed", flush=True)
                self.command(full, mode=mode)
                self.command(resumed, "--stop-after-epoch", "1", mode=mode)
                self.command(resumed, "--resume", mode=mode)
                a, b = (torch.load(p / "last.pt", map_location="cpu", weights_only=True) for p in (full, resumed))
                for name in a["model"]:
                    self.assertTrue(torch.equal(a["model"][name], b["model"][name]), name)
                self.assertEqual(a["scheduler"], b["scheduler"])
                for x, y in zip(a["history"], b["history"]):
                    for key in ("train_loss", "lr", "val", "per_event", "sample_order_sha256"):
                        self.assertEqual(x[key], y[key], key)
                self.assertTrue(torch.equal(a["rng"]["loader"], b["rng"]["loader"]))
                export(full, full / "validation", self.root, torch.device("cuda"))
                paths[condition + "_" + model] = full
                del a, b
        result = compare_runs(paths, self.root / "comparison")
        self.assertEqual(len(result["pairs"]), 2)
        self.assertEqual(sum(result["pairs"]["b1"]["paired"]["counts"]["pooled"].values()), 4)
        summaries = [json.loads((paths[k] / "summary.json").read_text()) for k in ("a_b1", "b_b1")]
        self.assertEqual(summaries[0]["initialization_sha256"], summaries[1]["initialization_sha256"])
        for key in ("cache_sha256", "initialization_sha256", "code_sha256", "manifest_sha256", "mode"):
            changed = copy.deepcopy(summaries[1])
            changed[key] = "changed"
            with self.assertRaises(ValueError):
                validate_pair(summaries[0], changed)
        changed = copy.deepcopy(summaries[1])
        changed["history"][0]["sample_order_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "sample orders"):
            validate_pair(summaries[0], changed)
        # A checkpoint cannot be resumed under B configuration, even if complete.
        result = self.command(paths["a_b2"], "--resume", mode="early", success=False)
        self.assertIn("config changed", result.stderr)
        self.command(paths["b_b2"], mode="early", success=False)
        # Valid recovery keeps the failure audit while returning to complete.
        self.config.update(class_weight_mode="inverse_frequency", experiment="baseline")
        self.save()
        self.command(paths["a_b2"], "--resume", mode="early")
        self.assertEqual(json.loads((paths["a_b2"] / "status.json").read_text())["status"], "complete")
        self.assertEqual(len(list(paths["a_b2"].glob("failure_*.json"))), 1)
        missing = compare_runs({k: self.root / ("missing_" + k) for k in paths}, self.root / "missing_results")
        self.assertEqual(set(missing["statuses"].values()), {"missing"})
        self.assertFalse(missing["pairs"])


if __name__ == "__main__":
    unittest.main()
