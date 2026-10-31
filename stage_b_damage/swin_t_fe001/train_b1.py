"""Train the FE-001 Swin-T post-only B1 model.

The script is intentionally self-contained under ``stage_b_damage/swin_t_fe001/``.  It uses AMP,
pinned host memory, persistent workers, prefetching and non-blocking transfers
when CUDA is available.  Importing this module never starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# Must be set before importing torch for deterministic CUDA GEMM where needed.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import rasterio
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from dataset import LABEL_TO_ID, XBDPostDataset
from metrics import metrics_from_predictions
from modeling import build_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path) -> tuple[dict, Path, Path]:
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config_dir = path.parent
    project_root = (config_dir / config["project_root"]).resolve()
    manifest = (project_root / config["manifest"]).resolve()
    return config, project_root, manifest


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_seed(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(
    dataset: XBDPostDataset,
    config: dict,
    *,
    train: bool,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> DataLoader:
    workers = int(config["num_workers"])
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config["batch_size"]),
        "shuffle": train,
        "num_workers": workers,
        "pin_memory": bool(config["pin_memory"]),
        "drop_last": False,
        "worker_init_fn": worker_seed,
        "generator": generator if generator is not None else torch.Generator().manual_seed(int(seed)),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(config["persistent_workers"])
        kwargs["prefetch_factor"] = int(config["prefetch_factor"])
    return DataLoader(**kwargs)


def make_amp(device: torch.device, config: dict):
    enabled = bool(config["amp"]) and device.type == "cuda"
    dtype = torch.float16 if config.get("amp_dtype", "float16") == "float16" else torch.bfloat16
    if hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda", enabled=enabled)
    else:  # pragma: no cover - compatibility for older torch versions
        scaler = torch.cuda.amp.GradScaler(enabled=enabled)
    return enabled, dtype, scaler


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def prepare_images(images: torch.Tensor, device: torch.device, config: dict) -> torch.Tensor:
    images = images.to(device, non_blocking=bool(config["pin_memory"]))
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    images = (images - mean) / std
    if bool(config["channels_last"]):
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict:
    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images = prepare_images(batch["image"], device, config)
            with autocast_context(device, amp_enabled, amp_dtype):
                logits = model(images)
            probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(batch["label"].tolist())
    return metrics_from_predictions(actual, predicted, np.asarray(probabilities))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Any,
    device: torch.device,
    config: dict,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, int]:
    model.train()
    accumulation = max(1, int(config["gradient_accumulation_steps"]))
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    seen = 0
    for step, batch in enumerate(loader, start=1):
        images = prepare_images(batch["image"], device, config)
        labels = batch["label"].to(device, non_blocking=bool(config["pin_memory"]))
        with autocast_context(device, amp_enabled, amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)
            scaled_loss = loss / accumulation
        if amp_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        should_step = step % accumulation == 0 or step == len(loader)
        if should_step:
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        batch_size = int(labels.numel())
        loss_sum += float(loss.detach()) * batch_size
        seen += batch_size
    return loss_sum / max(seen, 1), seen


def rng_state(loader_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader_generator.get_state(),
    }


def restore_rng(state: dict, loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_generator.set_state(state["loader"])


def validate_dataset_contract(config: dict, manifest: Path, project_root: Path) -> tuple[XBDPostDataset, XBDPostDataset]:
    if sha256_file(manifest) != config["manifest_sha256"]:
        raise RuntimeError("Manifest SHA256 does not match config; stop before training")
    # Validate the complete FE-001 contract first. A bounded pilot may take
    # only the first N rows, which is not expected to contain every event.
    full_train = XBDPostDataset(
        manifest, config["train_split"], project_root=project_root,
        crop_size=config["crop_size"], context=config["context"],
        max_rows=0, max_open_files=config["max_open_files"],
    )
    full_val = XBDPostDataset(
        manifest, config["val_split"], project_root=project_root,
        crop_size=config["crop_size"], context=config["context"],
        max_rows=0, max_open_files=config["max_open_files"],
    )
    if len(full_train) != config["expected_train_rows"]:
        raise RuntimeError(f"Unexpected train rows: {len(full_train)}")
    if len(full_val) != config["expected_val_rows"]:
        raise RuntimeError(f"Unexpected val rows: {len(full_val)}")
    if full_train.class_counts() != config["expected_train_counts"]:
        raise RuntimeError(f"Unexpected train class counts: {full_train.class_counts()}")
    if sorted(full_train.events()) != sorted(config["train_events"]):
        raise RuntimeError(f"Unexpected train events: {full_train.events()}")
    if sorted(full_val.events()) != sorted(config["val_events"]):
        raise RuntimeError(f"Unexpected val events: {full_val.events()}")
    train = full_train if not config["max_train"] else XBDPostDataset(
        manifest, config["train_split"], project_root=project_root,
        crop_size=config["crop_size"], context=config["context"],
        max_rows=config["max_train"], max_open_files=config["max_open_files"],
    )
    val = full_val if not config["max_val"] else XBDPostDataset(
        manifest, config["val_split"], project_root=project_root,
        crop_size=config["crop_size"], context=config["context"],
        max_rows=config["max_val"], max_open_files=config["max_open_files"],
    )
    return train, val


def make_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    history: list[dict],
    best_epoch: int,
    best_val: dict | None,
    config: dict,
    manifest: Path,
    loader_generator: torch.Generator,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "history": history,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "config": config,
        "config_sha256": stable_hash(config),
        "manifest_sha256": sha256_file(manifest),
        "rng": rng_state(loader_generator),
        "label_to_id": LABEL_TO_ID,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("..\..\configs\swin_t_fe001_b1.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config, project_root, manifest = load_config(args.config)
    config = dict(config)
    if args.device is not None:
        config["device"] = args.device
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        config["batch_size"] = args.batch_size
    if args.max_train is not None:
        config["max_train"] = args.max_train
    if args.max_val is not None:
        config["max_val"] = args.max_val
    if args.epochs is not None:
        config["epochs"] = args.epochs
    output_dir = (args.output_dir or (args.config.parent / config["output_dir"])).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is non-empty: {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(config["device"])
    seed_everything(int(config["seed"]), bool(config["deterministic"]))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config["allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(config["allow_tf32"])
    amp_enabled, amp_dtype, scaler = make_amp(device, config)
    train_ds, val_ds = validate_dataset_contract(config, manifest, project_root)
    loader_generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = make_loader(train_ds, config, train=True, generator=loader_generator)
    val_loader = make_loader(val_ds, config, train=False, seed=int(config["seed"]) + 1)

    model, weight_name = build_model(
        num_classes=int(config["num_classes"]), pretrained=bool(config["pretrained"])
    )
    model = model.to(device)
    if bool(config["channels_last"]):
        model = model.to(memory_format=torch.channels_last)
    counts = train_ds.class_counts()
    weights = torch.tensor(
        [1.0 / max(counts[name], 1) for name in LABEL_TO_ID],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))

    history: list[dict] = []
    best_epoch = 0
    best_val: dict | None = None
    start_epoch = 1
    last_path = output_dir / "last.pt"
    if args.resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["manifest_sha256"] != config["manifest_sha256"]:
            raise RuntimeError("Resume manifest hash does not match current config")
        if checkpoint["config_sha256"] != stable_hash(config):
            raise RuntimeError("Resume config differs; use a new output directory")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint["history"]
        best_epoch = checkpoint["best_epoch"]
        best_val = checkpoint["best_val"]
        restore_rng(checkpoint["rng"], loader_generator)
        start_epoch = int(checkpoint["epoch"]) + 1

    provenance = {
        "config": config,
        "config_sha256": stable_hash(config),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "model": "swin_t",
        "pretrained_weights": weight_name,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "class_counts": counts,
        "class_weights": weights.detach().cpu().tolist(),
        "acceleration": {
            "amp": amp_enabled,
            "num_workers": config["num_workers"],
            "pin_memory": config["pin_memory"],
            "persistent_workers": config["persistent_workers"],
            "channels_last": config["channels_last"],
            "compile": config["compile"],
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_loss, seen = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, config, amp_enabled, amp_dtype
        )
        val_metrics = evaluate(model, val_loader, device, config, amp_enabled, amp_dtype)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_rows_seen": seen,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val": val_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        improved = best_val is None or val_metrics["macro_f1"] > best_val["macro_f1"]
        if improved:
            best_epoch = epoch
            best_val = val_metrics
        checkpoint = make_checkpoint(
            model, optimizer, scheduler, scaler, epoch, history, best_epoch, best_val,
            config, manifest, loader_generator,
        )
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = {
        **provenance,
        "status": "complete",
        "best_epoch": best_epoch,
        "best_val": best_val,
        "history": history,
        "training_seconds": time.perf_counter() - started,
        "test_evaluated": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

