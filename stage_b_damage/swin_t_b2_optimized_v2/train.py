"""Train the optimized dual-stream Swin-T B2 model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

try:  # Running as a script from this directory.
    from dataset import ID_TO_LABEL, LABEL_TO_ID, OptimizedPairedDataset
    from losses import OptimizedLoss, class_weights, set_requires_grad
    from model import build_model, normalize_rgb
    from sampler import build_weighted_sampler
except ImportError:  # pragma: no cover - package import path
    from .dataset import ID_TO_LABEL, LABEL_TO_ID, OptimizedPairedDataset
    from .losses import OptimizedLoss, class_weights, set_requires_grad
    from .model import build_model, normalize_rgb
    from .sampler import build_weighted_sampler


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for name, value in ema_state.items():
            source = model_state[name].detach()
            if torch.is_floating_point(value):
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_seed(worker_id: int) -> None:
    worker_seed_value = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed_value)
    np.random.seed(worker_seed_value)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_config(path: Path) -> tuple[dict[str, Any], Path, Path]:
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_config_dir"] = str(path.parent)
    project_root = (path.parent / config["project_root"]).resolve()
    manifest = (project_root / config["manifest"]).resolve()
    return config, project_root, manifest


def make_dataset(config: dict[str, Any], manifest: Path, project_root: Path, split: str, max_rows: int = 0, hard_keys: set[str] | None = None) -> OptimizedPairedDataset:
    return OptimizedPairedDataset(
        manifest,
        split,
        project_root=project_root,
        crop_size=int(config.get("crop_size", 224)),
        context=int(config.get("context", 16)),
        alignment=config.get("alignment", {}),
        radiometric=config.get("radiometric", {}),
        augmentation=config.get("augmentation", {}),
        hard_negative=config.get("hard_negative", {}),
        max_rows=max_rows,
        max_open_files=int(config.get("max_open_files", 64)),
        hard_negative_keys=hard_keys,
    )


def make_loader(
    dataset: OptimizedPairedDataset,
    config: dict[str, Any],
    *,
    train: bool,
    generator: torch.Generator,
    sampler: Any = None,
) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config.get("batch_size", 8)),
        "shuffle": bool(train and sampler is None),
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(config.get("pin_memory", False)),
        "drop_last": False,
        "worker_init_fn": worker_seed,
        "generator": generator,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def make_amp(device: torch.device, config: dict[str, Any]):
    enabled = bool(config.get("amp", True)) and device.type == "cuda"
    dtype = torch.float16 if config.get("amp_dtype", "float16") == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=enabled)
    return enabled, dtype, scaler


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if enabled else torch.autocast(device_type="cpu", enabled=False)


def prepare_batch(batch: dict[str, Any], device: torch.device, config: dict[str, Any]) -> dict[str, Any]:
    non_blocking = bool(config.get("pin_memory", False))
    pre = normalize_rgb(batch["pre"].to(device, non_blocking=non_blocking))
    post = normalize_rgb(batch["post"].to(device, non_blocking=non_blocking))
    valid_mask = batch["valid_mask"].to(device, non_blocking=non_blocking)
    quality = batch["quality"].to(device, non_blocking=non_blocking)
    labels = batch["label"].to(device, non_blocking=non_blocking)
    return {
        "pre": pre,
        "post": post,
        "valid_mask": valid_mask,
        "quality": quality,
        "labels": labels,
        "hard_negative": batch["hard_negative"].to(device, non_blocking=non_blocking),
        "change_score": batch["change_score"].to(device, non_blocking=non_blocking),
    }


def metrics_from_predictions(labels: list[int], predictions: list[int], probabilities: np.ndarray) -> dict[str, Any]:
    matrix = np.zeros((4, 4), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        matrix[int(actual), int(predicted)] += 1
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for index, name in ID_TO_LABEL.items():
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - matrix[index, index])
        fn = float(matrix[index, :].sum() - matrix[index, index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index].sum())}
        f1_values.append(f1)
        recalls.append(recall)
    return {
        "n": int(matrix.sum()),
        "accuracy": float(np.trace(matrix) / max(int(matrix.sum()), 1)),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class": per_class,
        "confusion": matrix.tolist(),
        "macro_ap": float(_macro_ap(labels, probabilities)),
    }


def _macro_ap(labels: list[int], probabilities: np.ndarray) -> float:
    values: list[float] = []
    labels_array = np.asarray(labels, dtype=np.int64)
    for class_index in range(4):
        target = (labels_array == class_index).astype(np.int64)
        positives = int(target.sum())
        if positives == 0:
            values.append(0.0)
            continue
        order = np.argsort(-probabilities[:, class_index], kind="mergesort")
        sorted_target = target[order]
        true_positive = np.cumsum(sorted_target, dtype=np.float64)
        false_positive = np.cumsum(1 - sorted_target, dtype=np.float64)
        precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
        recall = true_positive / positives
        previous = np.concatenate(([0.0], recall[:-1]))
        values.append(float(np.sum((recall - previous) * precision)))
    return float(np.mean(values))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            prepared = prepare_batch(batch, device, config)
            with autocast_context(device, amp_enabled, amp_dtype):
                outputs = model(
                    prepared["pre"],
                    prepared["post"],
                    prepared["quality"],
                    prepared["valid_mask"],
                )
            logits = outputs["logits"].float()
            probs = torch.softmax(logits, dim=1)
            batch_predictions = probs.argmax(dim=1)
            labels.extend(prepared["labels"].cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            probabilities.extend(probs.cpu().numpy())
            meta = batch["meta"]
            for index in range(len(batch_predictions)):
                row = {
                    "key": meta["key"][index],
                    "event": meta["event"][index],
                    "tile_id": meta["tile_id"][index],
                    "building_index": meta["building_index"][index],
                    "label": int(prepared["labels"][index].cpu()),
                    "prediction": int(batch_predictions[index].cpu()),
                    "change_score": float(batch["change_score"][index].item()),
                    "registration_score": float(batch["quality"][index, 0].item()),
                    "gate": float(outputs["gate"][index].detach().cpu()),
                }
                row.update({f"logit_{j}": float(logits[index, j].cpu()) for j in range(4)})
                rows.append(row)
    probability_array = np.asarray(probabilities, dtype=np.float32)
    return metrics_from_predictions(labels, predictions, probability_array), rows


def write_prediction_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: OptimizedLoss,
    scaler: Any,
    device: torch.device,
    config: dict[str, Any],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    ema: ModelEMA | None,
) -> dict[str, float]:
    model.train()
    accumulation = max(1, int(config.get("gradient_accumulation_steps", 1)))
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = {"total": 0.0, "ce": 0.0, "ordinal": 0.0, "damage": 0.0, "distill": 0.0}
    seen = 0
    for step, batch in enumerate(loader, start=1):
        prepared = prepare_batch(batch, device, config)
        with autocast_context(device, amp_enabled, amp_dtype):
            outputs = model(
                prepared["pre"],
                prepared["post"],
                prepared["quality"],
                prepared["valid_mask"],
            )
            loss, components = criterion(
                outputs,
                prepared["labels"],
                hard_negative=prepared["hard_negative"],
                distill_mask=(prepared["labels"] == 0) & (prepared["change_score"].squeeze(1) < 0.12),
            )
            loss_for_backward = loss / accumulation
        if amp_enabled:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()
        if step % accumulation == 0 or step == len(loader):
            if amp_enabled:
                scaler.unscale_(optimizer)
            clip_norm = float(config.get("gradient_clip_norm", 1.0))
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        batch_size = int(prepared["labels"].numel())
        seen += batch_size
        for name in totals:
            totals[name] += float(components.get(name, 0.0)) * batch_size
    return {name: value / max(seen, 1) for name, value in totals.items()}


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    encoder_parameters: list[nn.Parameter] = []
    other_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        (encoder_parameters if name.startswith("encoder.") else other_parameters).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": float(config.get("backbone_learning_rate", 2e-5)), "group_name": "encoder"},
            {"params": other_parameters, "lr": float(config.get("head_learning_rate", 5e-4)), "group_name": "head"},
        ],
        weight_decay=float(config.get("weight_decay", 0.01)),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> Any:
    epochs = int(config.get("epochs", 12))
    warmup_epochs = int(config.get("warmup_epochs", 0))
    if warmup_epochs > 0 and warmup_epochs < epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config_b2_optimized.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, project_root, manifest = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    for name in ("epochs", "batch_size", "max_train", "max_val"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    expected_hash = config.get("manifest_sha256")
    if expected_hash and sha256_file(manifest) != expected_hash:
        raise RuntimeError("Manifest SHA256 does not match config")
    output_dir = (args.output_dir or (args.config.parent / config["output_dir"])).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is non-empty: {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config.get("seed", 42)), bool(config.get("deterministic", True)))
    device = choose_device(str(config.get("device", "auto")))
    amp_enabled, amp_dtype, scaler = make_amp(device, config)

    # The hard-negative file is read before constructing the dataset so its keys
    # are included in every batch and can affect both sampler and loss weight.
    try:
        from sampler import load_hard_negative_keys
    except ImportError:  # pragma: no cover
        from .sampler import load_hard_negative_keys
    hard_file = str(config.get("hard_negative_file", ""))
    hard_path = (args.config.parent / hard_file).resolve() if hard_file else None
    hard_keys = load_hard_negative_keys(hard_path)
    train_ds = make_dataset(config, manifest, project_root, str(config.get("train_split", "train")), int(config.get("max_train", 0)), hard_keys)
    val_ds = make_dataset(config, manifest, project_root, str(config.get("val_split", "val")), int(config.get("max_val", 0)), set())
    loader_generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    sampler, sampler_metadata, _ = build_weighted_sampler(train_ds, config, loader_generator)
    train_loader = make_loader(train_ds, config, train=True, generator=loader_generator, sampler=sampler)
    val_loader = make_loader(val_ds, config, train=False, generator=torch.Generator().manual_seed(int(config.get("seed", 42)) + 1))

    model, model_report = build_model(config)
    model = model.to(device)
    if bool(config.get("channels_last", False)):
        model = model.to(memory_format=torch.channels_last)
    freeze_epochs = int(config.get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0:
        set_requires_grad(model.encoder, False)
    counts = train_ds.class_counts()
    weights = class_weights(counts, list(LABEL_TO_ID.keys()), float(config.get("class_weight_power", 0.7))).to(device)
    loss_config = config.get("loss", {})
    criterion = OptimizedLoss(
        class_weight=weights,
        ordinal_weight=float(loss_config.get("ordinal_weight", 0.2)),
        damage_weight=float(loss_config.get("damage_weight", 0.1)),
        distill_weight=float(loss_config.get("distill_weight", 0.05)),
        focal_gamma=float(loss_config.get("focal_gamma", 0.0)),
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    ema = ModelEMA(model, float(config.get("ema_decay", 0.999))) if bool(config.get("ema", True)) else None
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_val: dict[str, Any] | None = None
    best_epoch = 0
    last_path = output_dir / "last.pt"
    if args.resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if ema is not None and checkpoint.get("ema_model") is not None:
            ema.module.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint.get("history", [])
        best_val = checkpoint.get("best_val")
        best_epoch = int(checkpoint.get("best_epoch", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if start_epoch > freeze_epochs:
            set_requires_grad(model.encoder, True)

    provenance = {
        "experiment_id": config.get("experiment_id"),
        "config": config,
        "config_sha256": stable_hash(config),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "model": "dual_stream_swin_t",
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__},
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "class_counts": counts,
        "class_weights": weights.detach().cpu().tolist(),
        "sampler": sampler_metadata,
        "model_report": model_report,
        "amp": amp_enabled,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    started = time.perf_counter()
    stale = 0
    patience = int(config.get("early_stopping_patience", 0))
    for epoch in range(start_epoch, int(config.get("epochs", 12)) + 1):
        if freeze_epochs and epoch == freeze_epochs + 1:
            set_requires_grad(model.encoder, True)
        train_values = train_epoch(model, train_loader, optimizer, criterion, scaler, device, config, amp_enabled, amp_dtype, ema)
        evaluation_model = ema.module if ema is not None and bool(config.get("eval_use_ema", True)) else model
        val_values, val_rows = evaluate(evaluation_model, val_loader, device, config, amp_enabled, amp_dtype)
        scheduler.step()
        record = {"epoch": epoch, "train": train_values, "val": val_values, "learning_rates": [float(group["lr"]) for group in optimizer.param_groups]}
        history.append(record)
        improved = best_val is None or float(val_values["macro_f1"]) > float(best_val["macro_f1"])
        if improved:
            best_val = val_values
            best_epoch = epoch
            stale = 0
            write_prediction_csv(val_rows, output_dir / "predictions_val.csv")
        else:
            stale += 1
        checkpoint = {
            "model": model.state_dict(),
            "ema_model": ema.module.state_dict() if ema is not None else None,
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
        }
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if patience > 0 and stale >= patience:
            break

    summary = {
        **provenance,
        "status": "complete",
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "training_seconds": time.perf_counter() - started,
        "test_evaluated": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
