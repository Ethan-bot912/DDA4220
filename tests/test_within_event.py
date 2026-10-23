"""Within-event split protocol v3: guards, preflight, split tool and data lock.

Synthetic fixtures live under ignored output/. The real-manifest check only
READS dataset/expanded_7events/manifest_classification.csv when it exists.
Run from repository root: python -m unittest tests.test_within_event -v
"""
import copy
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from stage_b_damage.dataset import LABEL_TO_ID, XBDPairedDataset
from stage_b_damage.train_baseline import audit_splits, choose_indices
import tools.build_within_event_split as split_tool
import tools.run_expanded_session as session
import tools.train_expanded as trainer
from tools.train_expanded import preflight, sha256

ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST = ROOT / "dataset/expanded_7events/manifest_classification.csv"
WITHIN_CONFIG = ROOT / "configs/within_event_matthew_moore_seed42.json"
WITHIN_DIR = ROOT / "dataset/within_event_mm"
HEADER = ["split", "event", "tile_id", "building_index", "damage_raw", "damage", "pre_image", "post_image",
          "pre_label", "post_label", "post_bbox_xmin", "post_bbox_ymin", "post_bbox_xmax", "post_bbox_ymax",
          "post_geometry_json", "pre_geometry_json"]
LABELS = list(LABEL_TO_ID)
RAW = {"no_damage": "no-damage", "minor": "minor-damage", "major": "major-damage", "destroyed": "destroyed"}


def _preflight_before_change(config, root, manifest):
    """Verbatim copy of tools.train_expanded.preflight at commit 1638a3e (reference only)."""
    from tools.train_expanded import audit_splits, Counter, csv, LABEL_TO_ID, sha256
    events = audit_splits(manifest)
    expected = {"train": sorted(config["train_events"]), "val": sorted(config["val_events"])}
    if events != expected:
        raise ValueError(f"Unexpected events/splits: {events}; expected {expected}. Test is sealed.")
    counts = Counter()
    train_counts = Counter()
    seen = set()
    image_paths = set()
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["event"], row["tile_id"], row["building_index"])
            if key in seen:
                raise ValueError(f"Duplicate building key: {key}")
            seen.add(key)
            if row["damage"] not in LABEL_TO_ID:
                raise ValueError("Unexpected damage label")
            counts[row["split"]] += 1
            if row["split"] == "train":
                train_counts[row["damage"]] += 1
            image_paths.update((row["pre_image"], row["post_image"]))
    if counts != {"train": config["expected_train_rows"], "val": config["expected_val_rows"]}:
        raise ValueError(f"Building counts differ: {counts}")
    if train_counts != config["expected_train_counts"]:
        raise ValueError(f"Training class counts differ: {train_counts}")
    missing = [p for p in sorted(image_paths) if not (root / p.replace("\\", "/")).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[:3]}")
    return {"events": events, "rows": dict(counts), "train_counts": dict(train_counts),
            "unique_images": len(image_paths), "manifest_sha256": sha256(manifest)}


def outcome(function, *args):
    try:
        return ("ok", function(*args))
    except Exception as exc:  # compare exception type and exact message
        return (type(exc).__name__, str(exc))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write_tile(directory, event, tile, rng):
    """One pre/post GeoTIFF pair and label JSON pair per tile, paths relative to ROOT."""
    paths = {}
    for phase in ("pre", "post"):
        image = directory / f"{event}_{tile}_{phase}_disaster.tif"
        with rasterio.open(image, "w", driver="GTiff", height=64, width=64, count=3, dtype="int16",
                           transform=from_origin(0, 64, 1, 1), nodata=-99) as dst:
            dst.write(rng.integers(0, 256, (3, 64, 64), dtype=np.int16))
        label = directory / f"{event}_{tile}_{phase}_disaster.json"
        label.write_text(json.dumps({"event": event, "tile": tile, "phase": phase}), encoding="utf-8")
        paths[phase + "_image"], paths[phase + "_label"] = relative(image), relative(label)
    return paths


def write_manifest(path, tiles, rng):
    """tiles: list of (split, event, tile_id, [damage, ...]). Writes LF, UTF-8, csv-quoted geometry."""
    rows = []
    for split, event, tile, damages in tiles:
        paths = write_tile(path.parent, event, tile, rng)
        for i, damage in enumerate(damages):
            x = 4 + i % 8 * 6
            geometry = json.dumps({"type": "Polygon", "coordinates": [[[x, 8], [x + 5, 8], [x + 5, 13], [x, 8]]]})
            rows.append([split, event, tile, str(i), RAW[damage], damage, paths["pre_image"], paths["post_image"],
                         paths["pre_label"], paths["post_label"], x, 8, x + 5, 13, geometry, geometry])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    return rows


def base_config(manifest, **changes):
    config = json.loads((ROOT / "configs/expanded_seed42.json").read_text(encoding="utf-8"))
    config.update(manifest=str(manifest), epochs=2, batch_size=4, crop_size=32, max_open_files=4, **changes)
    return config


def synthetic_source_layout(tiles_per_event=60):
    """Deterministic (no RNG) source manifest: two events of ~12k rows plus one excluded event."""
    layout = []
    for e, event in enumerate(("synthetic-alpha", "synthetic-beta")):
        for k in range(tiles_per_event):
            size = 100 + (k * 37 + e * 11) % 201
            main = (k + e) % 4
            damages = [LABELS[main] if i % 20 < 11 else LABELS[(main + 1 + i % 3) % 4] for i in range(size)]
            layout.append(("val" if e else "train", event, f"{k:08d}", damages))
    layout.append(("train", "synthetic-excluded", "00000000", ["no_damage", "minor"]))
    return layout


class AuditSplitsTest(unittest.TestCase):
    def setUp(self):
        (ROOT / "output").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "output")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def manifest(self, rows):
        path = self.root / f"audit_{len(list(self.root.iterdir()))}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["split", "event", "tile_id", "building_index", "damage"])
            writer.writerows(rows)
        return path

    def test_same_event_different_tiles_pass_in_any_two_splits(self):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            path = self.manifest([[a, "E", "00000001", "0", "minor"], [a, "E", "00000001", "1", "major"],
                                  [b, "E", "00000002", "0", "minor"]])
            self.assertEqual(audit_splits(path), {a: ["E"], b: ["E"]})
        path = self.manifest([["train", "E", "1", "0", "minor"], ["val", "E", "2", "0", "minor"],
                              ["test", "E", "3", "0", "minor"], ["test", "F", "1", "0", "minor"]])
        self.assertEqual(audit_splits(path), {"train": ["E"], "val": ["E"], "test": ["E", "F"]})

    def test_same_tile_in_any_two_splits_fails(self):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"), ("test", "train")):
            path = self.manifest([[a, "E", "00000001", "0", "minor"], [b, "E", "00000001", "1", "major"]])
            with self.assertRaisesRegex(ValueError, r"Data leakage across splits: \('tile', 'E', '00000001'\)"):
                audit_splits(path)

    def test_same_tile_id_in_different_events_is_not_leakage(self):
        path = self.manifest([["train", "E", "00000001", "0", "minor"], ["val", "F", "00000001", "0", "minor"]])
        self.assertEqual(audit_splits(path), {"train": ["E"], "val": ["F"]})

    def test_unexpected_split_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected split"):
            audit_splits(self.manifest([["holdout", "E", "1", "0", "minor"]]))

    def test_trainer_and_frozen_evaluator_share_one_guard(self):
        import evaluation.evaluate_classifier as evaluator
        import stage_b_damage.train_baseline as baseline
        self.assertIs(evaluator.audit_splits, baseline.audit_splits)
        self.assertIs(trainer.audit_splits, baseline.audit_splits)
        names = [n for n in vars(baseline) if "audit" in n.lower()]
        self.assertEqual(names, ["audit_splits"], "only the one guard function may exist")


class PreflightCompatibilityTest(unittest.TestCase):
    """Configs without expected_test_rows/test_events behave exactly as before."""

    def setUp(self):
        (ROOT / "output").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "output")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rng = np.random.default_rng(7)

    def fixture(self, tiles):
        path = self.root / f"manifest_{len(list(self.root.glob('*.csv')))}.csv"
        write_manifest(path, tiles, self.rng)
        return path

    def assertSame(self, config, manifest):
        self.assertEqual(outcome(preflight, config, ROOT, manifest),
                         outcome(_preflight_before_change, config, ROOT, manifest))

    def test_old_configs_match_previous_implementation(self):
        good = self.fixture([("train", "A", "00000001", LABELS * 2), ("val", "B", "00000001", LABELS)])
        config = base_config(good, train_events=["A"], val_events=["B"], expected_train_rows=8, expected_val_rows=4,
                             expected_train_counts={k: 2 for k in LABELS})
        self.assertEqual(outcome(preflight, config, ROOT, good)[0], "ok")
        self.assertNotIn("test", preflight(config, ROOT, good)["rows"])
        variants = [config, dict(config, expected_train_rows=9), dict(config, expected_val_rows=3),
                    dict(config, expected_train_counts={k: 1 for k in LABELS}), dict(config, train_events=["B"]),
                    dict(config, val_events=["A", "B"])]
        for variant in variants:
            self.assertSame(variant, good)
        sealed = self.fixture([("train", "A", "00000001", LABELS * 2), ("val", "B", "00000001", LABELS),
                               ("test", "C", "00000001", LABELS)])
        self.assertSame(config, sealed)
        self.assertIn("Test is sealed", outcome(preflight, config, ROOT, sealed)[1])
        duplicate = self.fixture([("train", "A", "00000001", LABELS * 2), ("val", "B", "00000001", LABELS)])
        text = duplicate.read_text(encoding="utf-8").splitlines(keepends=True)
        duplicate.write_text("".join(text + [text[1]]), encoding="utf-8", newline="")
        self.assertSame(dict(config, expected_train_rows=9), duplicate)
        bad_label = self.fixture([("train", "A", "00000001", LABELS * 2), ("val", "B", "00000001", LABELS)])
        bad_label.write_text(bad_label.read_text(encoding="utf-8").replace(",major,", ",un-classified,", 1),
                             encoding="utf-8", newline="")
        self.assertSame(config, bad_label)
        missing = self.fixture([("train", "A", "00000002", LABELS * 2), ("val", "B", "00000002", LABELS)])
        with missing.open(encoding="utf-8", newline="") as handle:
            (ROOT / next(csv.DictReader(handle))["pre_image"]).unlink()
        self.assertSame(config, missing)

    def test_old_config_still_rejects_event_crossing_splits(self):
        # The relaxed audit no longer raises, but per-split event lists still differ from the config.
        crossing = self.fixture([("train", "A", "00000001", LABELS * 2), ("val", "A", "00000002", LABELS)])
        config = base_config(crossing, train_events=["A"], val_events=["B"], expected_train_rows=8,
                             expected_val_rows=4, expected_train_counts={k: 2 for k in LABELS})
        with self.assertRaisesRegex(ValueError, "Unexpected events/splits"):
            preflight(config, ROOT, crossing)

    def test_test_fields_must_come_together(self):
        good = self.fixture([("train", "A", "00000001", LABELS * 2), ("val", "B", "00000001", LABELS)])
        config = base_config(good, train_events=["A"], val_events=["B"], expected_train_rows=8, expected_val_rows=4,
                             expected_train_counts={k: 2 for k in LABELS})
        for key, value in (("expected_test_rows", 0), ("test_events", ["A"])):
            with self.assertRaisesRegex(ValueError, "configured together"):
                preflight(dict(config, **{key: value}), ROOT, good)


class TestRowsNeverTrainTest(unittest.TestCase):
    """A manifest with test rows passes preflight; training/selection only see train/val."""

    def setUp(self):
        (ROOT / "output").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "output")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.csv"
        write_manifest(self.manifest, [
            ("train", "E1", "00000001", LABELS), ("train", "E2", "00000001", LABELS),
            ("val", "E1", "00000002", LABELS[:2]), ("val", "E2", "00000002", LABELS[2:]),
            ("test", "E1", "00000003", ["destroyed"] * 3), ("test", "E2", "00000003", ["destroyed"] * 2)],
            np.random.default_rng(11))
        self.config = base_config(self.manifest, train_events=["E1", "E2"], val_events=["E1", "E2"],
                                  test_events=["E2", "E1"], expected_train_rows=8, expected_val_rows=4,
                                  expected_test_rows=5, expected_train_counts={k: 2 for k in LABELS})
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def test_preflight_counts_three_splits(self):
        checked = preflight(self.config, ROOT, self.manifest)
        self.assertEqual(checked["rows"], {"train": 8, "val": 4, "test": 5})
        self.assertEqual(checked["events"], {"train": ["E1", "E2"], "val": ["E1", "E2"], "test": ["E1", "E2"]})
        for change in (dict(expected_test_rows=4), dict(test_events=["E1"]), dict(expected_val_rows=9)):
            with self.assertRaises(ValueError):
                preflight(dict(self.config, **change), ROOT, self.manifest)

    def test_train_val_indices_have_expected_lengths(self):
        train = XBDPairedDataset(self.manifest, "train", project_root=ROOT, crop_size=32)
        val = XBDPairedDataset(self.manifest, "val", project_root=ROOT, crop_size=32)
        ti = choose_indices(len(train), self.config["max_train"], self.config["seed"])
        vi = choose_indices(len(val), self.config["max_val"], self.config["seed"] + 1)
        self.assertEqual((len(ti), len(vi)), (self.config["expected_train_rows"], self.config["expected_val_rows"]))
        self.assertEqual({r["split"] for r in train.rows + val.rows}, {"train", "val"})
        self.assertEqual(Counter(train.rows[i]["damage"] for i in ti), self.config["expected_train_counts"])
        train.close()
        val.close()

    def test_training_run_never_constructs_or_reads_test_rows(self):
        constructed, accessed = [], Counter()

        class Spy(XBDPairedDataset):
            def __init__(self, manifest_path, split, **kwargs):
                constructed.append(split)
                super().__init__(manifest_path, split, **kwargs)

            def __getitem__(self, index):
                accessed[self.rows[index]["split"]] += 1
                return super().__getitem__(index)

        out = self.root / "run"
        argv = ["train_expanded", "--config", str(self.config_path), "--mode", "early", "--output-dir", str(out),
                "--device", "cpu", "--stop-after-epoch", "1"]
        self.addCleanup(torch.use_deterministic_algorithms, False)
        with mock.patch.object(trainer, "XBDPairedDataset", Spy), mock.patch.object(sys, "argv", argv):
            self.assertEqual(trainer.main(), 0)
        self.assertEqual(constructed, ["train", "val"])
        self.assertEqual(dict(accessed), {"train": 8, "val": 4})
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["train_rows"], summary["val_rows"]), (8, 4))
        self.assertEqual(summary["class_counts"], self.config["expected_train_counts"])
        self.assertEqual(set(summary["best_per_event"]), {"E1", "E2"})
        self.assertEqual(summary["best_val"]["n"], 4)


class SplitToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "output").mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT / "output")
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "source" / "manifest_classification.csv"
        cls.source.parent.mkdir()
        cls.rows = write_manifest(cls.source, synthetic_source_layout(), np.random.default_rng(5))
        cls.source_sha = sha256(cls.source)
        cls.events = ["synthetic-alpha", "synthetic-beta"]
        cls.out = cls.root / "within"
        cls.build_split(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def build_split(cls, out, *extra, ratios=("0.6", "0.2", "0.2")):
        return split_tool.main(["--manifest", str(cls.source), "--events", *cls.events, "--ratios", *ratios,
                                "--output-dir", str(out), *extra])

    def report(self):
        return json.loads((self.out / split_tool.REPORT).read_text(encoding="utf-8"))

    def test_class_order_matches_dataset(self):
        self.assertEqual(split_tool.CLASSES, tuple(LABEL_TO_ID))

    def test_largest_remainder_targets_match_protocol(self):
        ratios = split_tool.parse_ratios(["0.6", "0.2", "0.2"])
        lr = split_tool.largest_remainder
        matthew = [lr(n, ratios) for n in (4058, 12331, 2717, 3524)]
        moore = [lr(n, ratios) for n in (19453, 886, 449, 1584)]
        self.assertEqual([sum(x) for x in zip(*matthew)], [13578, 4527, 4525])
        self.assertEqual([sum(x) for x in zip(*moore)], [13423, 4475, 4474])
        self.assertEqual(matthew[0], [2435, 812, 811])  # .8/.6/.6: train then val (tie -> val first)
        self.assertEqual(lr(5, ratios), [3, 1, 1])
        with self.assertRaises(ValueError):
            split_tool.parse_ratios(["0.6", "0.2", "0.3"])

    def test_outputs_are_byte_identical_and_frozen(self):
        second = self.root / "second"
        self.assertEqual(self.build_split(second), 0)
        for name in (split_tool.SPLIT_MAP, split_tool.MANIFEST, split_tool.REPORT):
            self.assertEqual((self.out / name).read_bytes(), (second / name).read_bytes(), name)
        self.assertEqual(self.build_split(self.out), 0)  # rerun into the frozen directory: identical, accepted
        with self.assertRaises(FileExistsError):
            self.build_split(second, ratios=("0.5", "0.25", "0.25"))
        self.assertEqual(sha256(self.source), self.source_sha, "source manifest must stay untouched")

    def test_split_quality_within_acceptance_lines(self):
        report = self.report()
        self.assertTrue(report["acceptance"]["passed"], report["acceptance"])
        for event in self.events:
            info = report["per_event"][event]
            self.assertTrue(info["converged"])
            for side in info["splits"].values():
                self.assertLessEqual(abs(side["deviation_pp"]), 1.5)
                for label, value in side["class_deviation"].items():
                    if value["checked"]:
                        self.assertLessEqual(abs(value["relative_deviation"]), 0.25)
        for split in ("val", "test"):
            self.assertGreaterEqual(min(report["pooled"][split]["class_counts"].values()), 500)

    def test_tiles_disjoint_and_split_map_consistent(self):
        split_tool.verify_split_files(self.out)
        owners = {}
        with (self.out / split_tool.MANIFEST).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self.assertEqual(owners.setdefault((row["event"], row["tile_id"]), row["split"]), row["split"])
        sides = {s: {k for k, v in owners.items() if v == s} for s in ("train", "val", "test")}
        self.assertTrue(all(sides.values()))
        self.assertFalse(sides["train"] & sides["val"] or sides["train"] & sides["test"] or sides["val"] & sides["test"])
        audit_splits(self.out / split_tool.MANIFEST)

    def test_manifest_rows_and_columns_match_source(self):
        source_raw = self.source.read_bytes()
        new_raw = (self.out / split_tool.MANIFEST).read_bytes()
        self.assertNotIn(b"\r", new_raw)
        self.assertFalse(new_raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\\", new_raw)
        with self.source.open(encoding="utf-8", newline="") as handle:
            source = [r for r in csv.reader(handle)]
        with (self.out / split_tool.MANIFEST).open(encoding="utf-8", newline="") as handle:
            derived = [r for r in csv.reader(handle)]
        self.assertEqual(derived[0], source[0])
        kept = [r for r in source[1:] if r[1] in self.events]
        self.assertEqual(len(derived) - 1, len(kept))
        self.assertEqual(len(kept), sum(1 for r in self.rows if r[1] in self.events))
        for old, new in zip(kept, derived[1:]):
            self.assertEqual(old[1:], new[1:])
            self.assertIn(new[0], ("train", "val", "test"))
        # Byte level: each record equals the source record with only the split field replaced.
        old_lines = [l for l in source_raw.decode("utf-8").splitlines(keepends=True)[1:]
                     if l.split(",", 2)[1] in self.events]
        new_lines = new_raw.decode("utf-8").splitlines(keepends=True)[1:]
        for old, new in zip(old_lines, new_lines):
            self.assertEqual(old.split(",", 1)[1], new.split(",", 1)[1])

    def test_report_config_fields_match_manifest(self):
        fields = self.report()["config_fields"]
        counts, train = Counter(), Counter()
        with (self.out / split_tool.MANIFEST).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                counts[row["split"]] += 1
                if row["split"] == "train":
                    train[row["damage"]] += 1
        self.assertEqual((fields["expected_train_rows"], fields["expected_val_rows"], fields["expected_test_rows"]),
                         (counts["train"], counts["val"], counts["test"]))
        self.assertEqual(fields["expected_train_counts"], dict(train))

    def test_pooled_minimum_shortfall_exits_without_split_files(self):
        small = self.root / "small" / "manifest.csv"
        small.parent.mkdir()
        write_manifest(small, [("val", "tiny", f"{k:08d}", LABELS * 3) for k in range(10)], np.random.default_rng(3))
        out = self.root / "small_out"
        with self.assertRaisesRegex(split_tool.SplitRejected, "Pooled val/test class minimum"):
            split_tool.main(["--manifest", str(small), "--events", "tiny", "--output-dir", str(out)])
        self.assertFalse((out / split_tool.MANIFEST).exists() or (out / split_tool.SPLIT_MAP).exists())
        self.assertTrue(json.loads((out / split_tool.REPORT).read_text(encoding="utf-8"))["acceptance"]["pooled_shortfalls"])

    def test_data_lock_is_enforced_by_launcher(self):
        lock_dir = self.root / "locked"
        self.assertEqual(self.build_split(lock_dir), 0)
        fields = json.loads((lock_dir / split_tool.REPORT).read_text(encoding="utf-8"))["config_fields"]
        config = base_config(relative(lock_dir / split_tool.MANIFEST), **fields)
        config_path = self.root / "locked_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="")
        self.assertEqual(self.build_split(lock_dir, "--config", str(config_path)), 0)
        lock = json.loads((lock_dir / split_tool.LOCK).read_text(encoding="utf-8"))
        self.assertEqual(list(lock), ["manifest_sha256", "config_sha256", "files", "test_prepared"])
        self.assertIs(lock["test_prepared"], False)
        self.assertEqual(lock["config_sha256"], sha256(config_path))
        self.assertEqual(lock["manifest_sha256"], sha256(lock_dir / split_tool.MANIFEST))
        referenced = set()
        with (lock_dir / split_tool.MANIFEST).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                referenced.update(row[k] for k in split_tool.FILE_COLUMNS)
        self.assertEqual([r["path"] for r in lock["files"]], sorted(referenced))
        self.assertFalse(any("synthetic-excluded" in r["path"] for r in lock["files"]))
        self.assertEqual(json.loads((lock_dir / split_tool.AUDIT).read_text(encoding="utf-8"))["errors"], [])

        busy = self.root / "busy_output"
        busy.mkdir()
        (busy / "keep").write_text("x")
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)

        def launch():
            argv = ["run_expanded_session", "--mode", "post", "--config", str(config_path), "--output-dir", str(busy)]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(session.subprocess, "Popen", side_effect=AssertionError("trainer started")):
                session.main()

        # All lock checks pass; the launcher then stops at the nonempty output guard (no trainer started).
        with self.assertRaises(FileExistsError):
            launch()
        manifest = lock_dir / split_tool.MANIFEST
        label = ROOT / lock["files"][0]["path"]
        image = ROOT / next(r["path"] for r in lock["files"] if r["path"].endswith(".tif"))
        for target, message in ((manifest, "Manifest differs"), (config_path, "Configuration differs"),
                                (label, "data file|Data bytes"), (image, "data file|Data bytes")):
            original = target.read_bytes()
            if target == config_path:  # keep the JSON parseable so the hash check, not the parser, rejects it
                variants = (original.replace(b" ", b"\t", 1), original + b" ")
            else:
                variants = (original[:-2] + bytes([original[-2] ^ 1]) + original[-1:], original + b" ")
            for tampered in variants:
                self.assertNotEqual(tampered, original)
                target.write_bytes(tampered)
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        launch()
                finally:
                    target.write_bytes(original)
        with self.assertRaises(FileExistsError):
            launch()


@unittest.skipUnless(REAL_MANIFEST.is_file(), "expanded manifest not present")
class RealManifestReferenceTest(unittest.TestCase):
    """Read-only check of the v3 reference magnitudes (doc section 3 step 1, rounded as printed)."""

    def test_reference_magnitudes_and_frozen_outputs(self):
        before = sha256(REAL_MANIFEST)
        files, report = split_tool.build_outputs(REAL_MANIFEST, ["hurricane-matthew", "moore-tornado"],
                                                 split_tool.parse_ratios(["0.6", "0.2", "0.2"]))
        self.assertEqual(sha256(REAL_MANIFEST), before)
        self.assertEqual(report["rows"], 45002)
        limits = {"hurricane-matthew": (22630, [13578, 4527, 4525], 0.095, 0.0225),
                  "moore-tornado": (22372, [13423, 4475, 4474], 0.025, 0.0095)}
        for event, (rows, targets, total_pp, class_rel) in limits.items():
            info = report["per_event"][event]
            self.assertEqual(info["rows"], rows)
            self.assertEqual([info["splits"][s]["target_rows"] for s in ("train", "val", "test")], targets)
            for side in info["splits"].values():
                self.assertLessEqual(abs(side["deviation_pp"]), total_pp, event)
                for value in side["class_deviation"].values():
                    if value["checked"]:
                        self.assertLessEqual(abs(value["relative_deviation"]), class_rel, event)
        self.assertTrue(report["acceptance"]["passed"])
        if (WITHIN_DIR / split_tool.MANIFEST).is_file():
            for name, data in files.items():
                self.assertEqual((WITHIN_DIR / name).read_bytes(), data, name)
            split_tool.verify_split_files(WITHIN_DIR)
        if WITHIN_CONFIG.is_file():
            config = json.loads(WITHIN_CONFIG.read_text(encoding="utf-8-sig"))
            for key, value in report["config_fields"].items():
                self.assertEqual(config[key] if "events" not in key else sorted(config[key]), value, key)


@unittest.skipUnless(WITHIN_CONFIG.is_file(), "within-event config not present")
class WithinEventConfigTest(unittest.TestCase):
    def test_config_differs_from_expanded_only_where_allowed(self):
        new = json.loads(WITHIN_CONFIG.read_text(encoding="utf-8-sig"))
        old = json.loads((ROOT / "configs/expanded_seed42.json").read_text(encoding="utf-8-sig"))
        allowed = {"experiment", "manifest", "train_events", "val_events", "expected_train_rows",
                   "expected_val_rows", "expected_train_counts"}
        self.assertEqual(set(new) - set(old), {"expected_test_rows", "test_events"})
        self.assertEqual(set(old) - set(new), set())
        self.assertLessEqual({k for k in old if old[k] != new[k]}, allowed)
        self.assertEqual(list(new)[:len(old)], list(old), "keep the original key order")
        both = ["hurricane-matthew", "moore-tornado"]
        self.assertEqual(new["manifest"], "dataset/within_event_mm/manifest_classification.csv")
        for key in ("train_events", "val_events", "test_events"):
            self.assertEqual(sorted(new[key]), both)
        self.assertEqual(new["sealed_test_events"], old["sealed_test_events"])
        self.assertEqual(new["expected_train_rows"] + new["expected_val_rows"] + new["expected_test_rows"], 45002)
        self.assertEqual(sum(new["expected_train_counts"].values()), new["expected_train_rows"])
        self.assertEqual(WITHIN_CONFIG.read_bytes().count(b"\r"), 0)


if __name__ == "__main__":
    unittest.main()
