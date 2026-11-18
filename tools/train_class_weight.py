"""Paired class-weight experiment with verified train/val crop cache. No held-out test evaluation.

Run as python -m tools.train_class_weight. Session budget is checked at complete
epoch boundaries. The LR schedule always spans config.epochs, across sessions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import rasterio
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from stage_b_damage.dataset import LABEL_TO_ID
from stage_b_damage.within_event_cache import CachedDataset, metadata_identity, verify_cache
from stage_b_damage.class_weight import class_weights, model_hash, epoch_order, order_hash
from stage_b_damage.train_baseline import (
    audit_splits, build_model, choose_indices, make_input, metrics_from_confusion, synchronize,
)

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(".pt.partial")
    torch.save(value, temporary)
    temporary.replace(path)


def code_fingerprint():
    names = ["tools/train_class_weight.py", "tools/train_expanded.py", "stage_b_damage/train_baseline.py",
             "stage_b_damage/dataset.py", "stage_b_damage/within_event_cache.py",
             "stage_b_damage/class_weight.py", "tools/build_within_event_cache.py"]
    return {name: sha256(ROOT / name) for name in names}


def rng_state(generator):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def preflight(config, root, manifest):
    events = audit_splits(manifest)
    expected = {"train": sorted(config["train_events"]), "val": sorted(config["val_events"])}
    # Within-event protocol v3 (change 2): a config declares an internal test split
    # with BOTH fields; then test is counted and its events checked like train/val
    # (counts and integrity only). Configs without them keep the two-split checks.
    declared = [key for key in ("expected_test_rows", "test_events") if key in config]
    if len(declared) == 1:
        raise ValueError("expected_test_rows and test_events must be configured together")
    with_test = len(declared) == 2
    if with_test:
        expected["test"] = sorted(config["test_events"])
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
    expected_rows = {"train": config["expected_train_rows"], "val": config["expected_val_rows"]}
    if with_test:
        expected_rows["test"] = config["expected_test_rows"]
    if counts != expected_rows:
        raise ValueError(f"Building counts differ: {counts}")
    if train_counts != config["expected_train_counts"]:
        raise ValueError(f"Training class counts differ: {train_counts}")
    missing = [p for p in sorted(image_paths) if not (root / p.replace("\\", "/")).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[:3]}")
    return {"events": events, "rows": dict(counts), "train_counts": dict(train_counts),
            "unique_images": len(image_paths), "manifest_sha256": sha256(manifest)}


def evaluate_events(model, loader, device, mode):
    model.eval()
    confusion = torch.zeros(4, 4, dtype=torch.long)
    grouped = {}
    with torch.inference_mode():
        for batch in loader:
            predicted = model(make_input(batch, mode).to(device)).argmax(1).cpu()
            actual = batch["label"]
            confusion += torch.bincount(actual * 4 + predicted, minlength=16).reshape(4, 4)
            for i, event in enumerate(batch["meta"]["event"]):
                grouped.setdefault(event, torch.zeros(4, 4, dtype=torch.long))[actual[i], predicted[i]] += 1
    return metrics_from_confusion(confusion), {e: metrics_from_confusion(c) for e, c in grouped.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mode", choices=["post", "early"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--session-hours", type=float, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stop-after-epoch", type=int, default=0, help="Additional epoch-boundary stop for verification")
    a = p.parse_args()
    if a.session_hours <= 0 or a.stop_after_epoch < 0:
        p.error("Session hours must be positive; stop epoch must be nonnegative")
    c = json.loads(a.config.read_text(encoding="utf-8-sig"))
    if c["workers"] < 0 or c["prefetch_factor"] < 1:
        raise ValueError("Invalid loader settings")
    if c["precision"] != "float32" or c["augmentation"] or c["pretrained"] or c["max_train"] or c["max_val"]:
        raise ValueError("This protocol requires full data, float32, no augmentation/pretraining")
    if c["selection"] != "pooled_validation_macro_f1_strict_improvement" or c["backbone"] != "resnet18":
        raise ValueError("Unsupported protocol")
    if c["persistent_workers"] is not False:
        raise ValueError("Persistent workers are disabled for epoch-boundary reproducibility")
    class_weights(c["expected_train_counts"], c["class_weight_mode"])
    if c["experiment"] != {"inverse_frequency": "baseline", "equal": "class_weight"}[c["class_weight_mode"]]:
        raise ValueError("Experiment identity does not match weight condition")
    root = a.project_root.resolve()
    manifest = root / c["manifest"]
    metadata_identity(c, root)
    checked = preflight(c, root, manifest)
    print(json.dumps({"preflight": checked}), flush=True)
    if a.preflight_only:
        return 0
    checked["cache_sha256"] = verify_cache(c, root)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently run formal training on CPU")
    out = a.output_dir.resolve()
    if not a.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {out}; use --resume for an existing run")
    if a.resume and not (out / "last.pt").is_file():
        raise FileNotFoundError(out / "last.pt")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "RUNNING.lock").exists():
        raise RuntimeError("RUNNING.lock exists; check the recorded PID before removing a stale lock")
    with (out / "RUNNING.lock").open("x") as lock:
        lock.write(str(os.getpid()))
    try:
        return run(a, c, root, manifest, checked, out, device)
    except BaseException as exc:
        failure = {"error": repr(exc), "pid": os.getpid(), "time": time.time()}
        atomic_json(out / f"failure_{time.time_ns()}.json", failure)
        atomic_json(out / "failure.json", failure)
        atomic_json(out / "status.json", {"status": "failed", **failure})
        raise
    finally:
        (out / "RUNNING.lock").unlink(missing_ok=True)


def run(a, c, root, manifest, checked, out, device):
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    # GDAL_CACHEMAX expects bytes through Rasterio, unlike the shell MB option.
    environment = rasterio.Env(GDAL_CACHEMAX=c["gdal_cache_mb"] * 1024 * 1024)
    with environment:
        return train(a, c, root, manifest, checked, out, device)


def train(a, c, root, manifest, checked, out, device):
    ds_args = dict(project_root=root, crop_size=c["crop_size"], context=c["context"], max_open_files=c["max_open_files"], cache_dir=root / c["cache_dir"])
    train_ds = CachedDataset(manifest, "train", **ds_args)
    val_ds = CachedDataset(manifest, "val", return_meta=True, **ds_args)
    ti = choose_indices(len(train_ds), c["max_train"], c["seed"])
    vi = choose_indices(len(val_ds), c["max_val"], c["seed"] + 1)
    index_hashes = {"train_indices_sha256": hashlib.sha256(json.dumps(ti).encode()).hexdigest(),
                    "val_indices_sha256": hashlib.sha256(json.dumps(vi).encode()).hexdigest()}
    generator = torch.Generator().manual_seed(c["seed"])
    loader_args = dict(batch_size=c["batch_size"], num_workers=c["workers"], pin_memory=c["pin_memory"])
    if c["workers"]:
        loader_args.update(prefetch_factor=c["prefetch_factor"], persistent_workers=False, multiprocessing_context="spawn")
    val_generator = torch.Generator()
    vl = DataLoader(Subset(val_ds, vi), shuffle=False, generator=val_generator, **loader_args)
    counts = Counter(train_ds.rows[i]["damage"] for i in ti)
    weights = class_weights(counts, c["class_weight_mode"])
    model = build_model(c["backbone"], a.mode).to(device)
    initialization_sha256 = model_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c["epochs"])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    fingerprint = code_fingerprint()
    versions = {"python": platform.python_version(), "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__), "numpy": str(np.__version__), "rasterio": str(rasterio.__version__)}
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", "unavailable"
    config_args = {**c, "mode": a.mode, "manifest": c["manifest"]}
    provenance = {"config": c, "mode": a.mode, "backbone": c["backbone"], "args": config_args,
                  "manifest_sha256": checked["manifest_sha256"], **index_hashes, "code_sha256": fingerprint,
                  "label_to_id": LABEL_TO_ID, "versions": versions, "hardware": hardware,
                  "initialization_sha256": initialization_sha256, "cache_sha256": checked["cache_sha256"],
                  "class_weights": weights.tolist(), "split_map_sha256": c["split_map_sha256"],
                  "data_lock_sha256": c["data_lock_sha256"],
                  "git_commit": commit, "git_dirty": dirty, "events": checked["events"], "test_evaluated": False}
    history, completed, total_seconds = [], 0, 0.0
    best_f1, best_epoch, best_metrics, best_events = -1.0, 0, None, None
    if a.resume:
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        for key in ("config", "mode", "manifest_sha256", "code_sha256", "train_indices_sha256", "val_indices_sha256", "versions", "hardware", "initialization_sha256", "cache_sha256", "class_weights", "split_map_sha256", "data_lock_sha256"):
            if saved[key] != provenance[key]:
                raise ValueError(f"Cannot resume: {key} changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        completed, history, total_seconds = saved["epoch"], saved["history"], saved["training_seconds"]
        best_f1, best_epoch = saved["best_f1"], saved["best_epoch"]
        best_metrics, best_events = saved["best_val"], saved["best_per_event"]
        restore_rng(saved["rng"], generator)
        del saved
    atomic_json(out / "run_metadata.json", provenance)
    # Reconcile JSONL to the committed last checkpoint after a crash.
    with (out / "history.jsonl").open("w", encoding="utf-8") as f:
        for record in history:
            f.write(json.dumps(record) + "\n")
    synchronize(device)
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def progress(phase, epoch, batch=0):
        atomic_json(out / "status.json", {"status": "running", "phase": phase, "pid": os.getpid(),
                    "epoch_in_progress": epoch, "completed_epochs": completed, "batch": batch,
                    "train_batches": len(tl), "session_elapsed_seconds": time.monotonic() - started,
                    "session_budget_seconds": a.session_hours * 3600, "best_epoch": best_epoch,
                    "best_macro_f1": best_f1, "updated_unix": time.time()})

    for epoch in range(completed + 1, c["epochs"] + 1):
        val_generator.manual_seed(c["seed"] + 100000 + epoch)
        order = epoch_order(len(ti), c["seed"], epoch)
        tl = DataLoader(Subset(train_ds, [ti[i] for i in order]), shuffle=False, generator=generator, **loader_args)
        progress("train", epoch)
        epoch_start = time.monotonic()
        model.train()
        numerator, denominator = 0.0, 0.0
        lr = optimizer.param_groups[0]["lr"]
        for batch_index, batch in enumerate(tl, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(make_input(batch, a.mode).to(device))
            labels = batch["label"].to(device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            weight = float(criterion.weight[labels].sum())
            numerator += float(loss.detach()) * weight
            denominator += weight
            if batch_index == 1 or batch_index % 100 == 0:
                progress("train", epoch, batch_index)
                print(json.dumps({"epoch": epoch, "batch": batch_index, "batches": len(tl),
                                  "elapsed_seconds": time.monotonic() - epoch_start}), flush=True)
        progress("validation", epoch, len(tl))
        metrics, per_event = evaluate_events(model, vl, device, a.mode)
        synchronize(device)
        record = {"epoch": epoch, "train_loss": numerator / denominator, "lr": lr,
                  "val": metrics, "per_event": per_event, "sample_order_sha256": order_hash([ti[i] for i in order]), "epoch_seconds": time.monotonic() - epoch_start}
        scheduler.step()
        history.append(record)
        improved = metrics["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, best_metrics, best_events = metrics["macro_f1"], epoch, metrics, per_event
            atomic_checkpoint(out / "best.pt", {**provenance, "model": model.state_dict(), "epoch": epoch,
                                               "val": metrics, "per_event": per_event})
        total_seconds += record["epoch_seconds"]
        completed = epoch
        atomic_checkpoint(out / "last.pt", {**provenance, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "rng": rng_state(generator), "epoch": completed,
                          "history": history, "training_seconds": total_seconds, "best_f1": best_f1,
                          "best_epoch": best_epoch, "best_val": best_metrics, "best_per_event": best_events})
        with (out / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if time.monotonic() - started >= a.session_hours * 3600 or (a.stop_after_epoch and completed >= a.stop_after_epoch):
            break
    done = completed == c["epochs"]
    result = {**provenance, "status": "complete" if done else "paused", "completed_epochs": completed,
              "best_epoch": best_epoch, "best_val": best_metrics, "best_per_event": best_events,
              "history": history, "train_rows": len(ti), "val_rows": len(vi), "class_counts": dict(counts),
              "class_weights": weights.tolist(), "training_seconds": total_seconds,
              "session_elapsed_seconds": time.monotonic() - started,
              "session_peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    atomic_json(out / "summary.json", result)
    atomic_json(out / "status.json", {"status": result["status"], "completed_epochs": completed,
                "best_epoch": best_epoch, "best_macro_f1": best_f1, "updated_unix": time.time(),
                "session_elapsed_seconds": result["session_elapsed_seconds"], "checkpoint": str(out / "last.pt")})
    train_ds.close()
    val_ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
