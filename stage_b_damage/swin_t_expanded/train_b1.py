"""Train the FE-001 Swin-T post-only B1 model.

The script is intentionally self-contained under ``swin-t/``.  It uses AMP,
pinned host memory, persistent workers, prefetching and non-blocking transfers
when CUDA is available.  Importing this module never starts training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import time
from collections import Counter
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from dataset import LABEL_TO_ID, XBDPostDataset
from metrics import metrics_from_predictions
from modeling import build_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ModelEMA:
    """Exponential moving average of model parameters for stable validation."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state)


class EventWeightedSampler(WeightedRandomSampler):
    """Weighted sampler that records the event mix drawn for each epoch."""

    def __init__(self, weights: torch.Tensor, dataset: XBDPostDataset, **kwargs: Any) -> None:
        super().__init__(weights, **kwargs)
        self.dataset = dataset
        self.last_event_counts: dict[str, int] = {}

    def __iter__(self):
        indices = list(super().__iter__())
        counts = Counter(self.dataset.rows[int(index)]["event"] for index in indices)
        self.last_event_counts = dict(sorted(counts.items()))
        return iter(indices)


def split_model_outputs(outputs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(outputs, dict):
        return outputs["logits"], outputs.get("ordinal")
    if isinstance(outputs, (tuple, list)):
        return outputs[0], outputs[1] if len(outputs) > 1 else None
    return outputs, None


def combined_loss(
    outputs: Any,
    labels: torch.Tensor,
    criterion: nn.Module,
    ordinal_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, ordinal_logits = split_model_outputs(outputs)
    ce_loss = criterion(logits, labels)
    if ordinal_logits is None or ordinal_weight <= 0.0:
        ordinal_loss = logits.new_zeros(())
        return ce_loss, ce_loss, ordinal_loss
    ordinal_targets = torch.stack(
        [(labels >= threshold).to(dtype=ordinal_logits.dtype) for threshold in (1, 2, 3)],
        dim=1,
    )
    ordinal_loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets)
    return ce_loss + float(ordinal_weight) * ordinal_loss, ce_loss, ordinal_loss


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
    sampler: Any | None = None,
) -> DataLoader:
    workers = int(config["num_workers"])
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config["batch_size"]),
        "shuffle": train and sampler is None,
        "num_workers": workers,
        "pin_memory": bool(config["pin_memory"]),
        "drop_last": False,
        "worker_init_fn": worker_seed,
        "generator": generator if generator is not None else torch.Generator().manual_seed(int(seed)),
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
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
                outputs = model(images)
                logits, _ = split_model_outputs(outputs)
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
    ema: ModelEMA | None = None,
    ordinal_weight: float = 0.0,
) -> tuple[float, int, float, float]:
    model.train()
    accumulation = max(1, int(config["gradient_accumulation_steps"]))
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    ce_sum = 0.0
    ordinal_sum = 0.0
    seen = 0
    for step, batch in enumerate(loader, start=1):
        window_start = ((step - 1) // accumulation) * accumulation + 1
        window_end = min(window_start + accumulation - 1, len(loader))
        window_size = window_end - window_start + 1
        images = prepare_images(batch["image"], device, config)
        labels = batch["label"].to(device, non_blocking=bool(config["pin_memory"]))
        with autocast_context(device, amp_enabled, amp_dtype):
            outputs = model(images)
            loss, ce_loss, ordinal_loss = combined_loss(
                outputs, labels, criterion, ordinal_weight
            )
            scaled_loss = loss / window_size
        if amp_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        should_step = step % accumulation == 0 or step == len(loader)
        if should_step:
            if amp_enabled:
                scaler.unscale_(optimizer)
            clip_norm = float(config.get("gradient_clip_norm", 0.0))
            if clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
        batch_size = int(labels.numel())
        loss_sum += float(loss.detach()) * batch_size
        ce_sum += float(ce_loss.detach()) * batch_size
        ordinal_sum += float(ordinal_loss.detach()) * batch_size
        seen += batch_size
    denominator = max(seen, 1)
    return loss_sum / denominator, seen, ce_sum / denominator, ordinal_sum / denominator


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
    # Checkpoints are loaded with ``map_location=device``.  When resuming on
    # CUDA that also moves the CPU RNG tensors to the GPU, while
    # ``torch.set_rng_state`` and ``Generator.set_state`` require CPU
    # ByteTensors.  Normalize the states here so resume works on either
    # device without changing the saved random stream.
    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        torch_state = torch.tensor(torch_state, dtype=torch.uint8)
    torch.set_rng_state(torch_state.detach().cpu())
    if state.get("cuda"):
        cuda_states = []
        for cuda_state in state["cuda"]:
            if not isinstance(cuda_state, torch.Tensor):
                cuda_state = torch.tensor(cuda_state, dtype=torch.uint8)
            cuda_states.append(cuda_state.detach().cpu())
        torch.cuda.set_rng_state_all(cuda_states)
    loader_state = state["loader"]
    if not isinstance(loader_state, torch.Tensor):
        loader_state = torch.tensor(loader_state, dtype=torch.uint8)
    loader_generator.set_state(loader_state.detach().cpu())


def validate_dataset_contract(config: dict, manifest: Path, project_root: Path) -> tuple[XBDPostDataset, XBDPostDataset]:
    if sha256_file(manifest) != config["manifest_sha256"]:
        raise RuntimeError("Manifest SHA256 does not match config; stop before training")
    # Validate the complete FE-001 contract first. A bounded pilot may take
    # only the first N rows, which is not expected to contain every event.
    full_train = XBDPostDataset(
        manifest, config["train_split"], project_root=project_root,
        crop_size=config["crop_size"], context=config["context"],
        max_rows=0, max_open_files=config["max_open_files"],
        augmentation=config.get("augmentation"),
        crop_geometry=config.get("crop_geometry"),
    )
    full_val = XBDPostDataset(
        manifest, config["val_split"], project_root=project_root,
        crop_size=config["crop_size"], context=config["context"],
        crop_geometry=config.get("crop_geometry"),
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
        augmentation=config.get("augmentation"),
        crop_geometry=config.get("crop_geometry"),
    )
    val = full_val if not config["max_val"] else XBDPostDataset(
        manifest, config["val_split"], project_root=project_root,
        crop_size=config["crop_size"], context=config["context"],
        crop_geometry=config.get("crop_geometry"),
        max_rows=config["max_val"], max_open_files=config["max_open_files"],
    )
    return train, val


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if not (name.startswith("head.") or name.startswith("ordinal_head.")):
            parameter.requires_grad_(trainable)


def build_event_sampler(
    dataset: XBDPostDataset,
    config: dict,
    generator: torch.Generator,
) -> tuple[EventWeightedSampler | None, dict[str, Any]]:
    settings = config.get("event_sampler", {})
    if not isinstance(settings, dict) or not bool(settings.get("enabled", False)):
        return None, {"enabled": False}
    event_counts = Counter(row["event"] for row in dataset.rows)
    max_count = max(event_counts.values())
    power = float(settings.get("power", 0.5))
    cap = float(settings.get("max_weight", 2.0))
    if power < 0.0 or cap <= 0.0:
        raise ValueError("event sampler power must be non-negative and max_weight positive")
    raw_weights = {
        event: min((max_count / count) ** power, cap)
        for event, count in event_counts.items()
    }
    weights = torch.tensor(
        [raw_weights[row["event"]] for row in dataset.rows], dtype=torch.double
    )
    # Keep the average sampling weight at one while preserving the cap ratio.
    weights = weights / weights.mean()
    requested_samples = int(settings.get("samples_per_epoch", 0))
    samples_per_epoch = len(dataset) if requested_samples <= 0 else min(requested_samples, len(dataset))
    sampler = EventWeightedSampler(
        weights,
        dataset,
        num_samples=samples_per_epoch,
        replacement=bool(settings.get("replacement", True)),
        generator=generator,
    )
    metadata = {
        "enabled": True,
        "power": power,
        "max_weight": cap,
        "replacement": bool(settings.get("replacement", True)),
        "samples_per_epoch": samples_per_epoch,
        "event_counts": dict(sorted(event_counts.items())),
        "raw_event_weights": dict(sorted(raw_weights.items())),
        "normalized_mean_weight": float(weights.mean()),
    }
    return sampler, metadata


def build_optimizer_and_scheduler(
    model: nn.Module, config: dict
) -> tuple[torch.optim.Optimizer, Any]:
    head_parameters: list[nn.Parameter] = []
    backbone_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.startswith("head.") or name.startswith("ordinal_head."):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    if not head_parameters or not backbone_parameters:
        raise RuntimeError("Expected non-empty Swin backbone and classification head parameter groups")
    base_lr = float(config.get("learning_rate", 1e-4))
    backbone_lr = float(config.get("backbone_learning_rate", base_lr))
    head_lr = float(config.get("head_learning_rate", base_lr))
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr, "group_name": "backbone"},
            {"params": head_parameters, "lr": head_lr, "group_name": "head"},
        ],
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    warmup_epochs = int(config.get("warmup_epochs", 0))
    if warmup_epochs > 0:
        if warmup_epochs >= epochs:
            raise ValueError("warmup_epochs must be smaller than epochs")
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler


def make_checkpoint(
    model: nn.Module,
    ema: ModelEMA | None,
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
        "ema_model": ema.state_dict() if ema is not None else None,
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
    parser.add_argument("--config", type=Path, default=Path("config_fe001_swin_t.json"))
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
    event_sampler, event_sampler_metadata = build_event_sampler(
        train_ds, config, loader_generator
    )
    train_loader = make_loader(
        train_ds, config, train=True, generator=loader_generator, sampler=event_sampler
    )
    val_loader = make_loader(val_ds, config, train=False, seed=int(config["seed"]) + 1)

    model, weight_name = build_model(
        num_classes=int(config["num_classes"]),
        pretrained=bool(config["pretrained"]),
        head_dropout=float(config.get("head_dropout", 0.0)),
        ordinal=float(config.get("ordinal_loss_weight", 0.0)) > 0.0,
    )
    model = model.to(device)
    if bool(config["channels_last"]):
        model = model.to(memory_format=torch.channels_last)
    freeze_backbone_epochs = int(config.get("freeze_backbone_epochs", 0))
    if freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
    counts = train_ds.class_counts()
    class_weight_power = float(config.get("class_weight_power", 1.0))
    if class_weight_power < 0.0:
        raise ValueError("class_weight_power must be non-negative")
    total_rows = float(sum(counts.values()))
    weights = torch.tensor(
        [(total_rows / max(counts[name], 1)) ** class_weight_power for name in LABEL_TO_ID],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights)
    ordinal_loss_weight = float(config.get("ordinal_loss_weight", 0.0))
    if ordinal_loss_weight < 0.0:
        raise ValueError("ordinal_loss_weight must be non-negative")
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    ema = ModelEMA(model, float(config["ema_decay"])) if bool(config.get("ema", False)) else None

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
        if ema is not None and checkpoint.get("ema_model") is not None:
            ema.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint["history"]
        best_epoch = checkpoint["best_epoch"]
        best_val = checkpoint["best_val"]
        restore_rng(checkpoint["rng"], loader_generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        set_backbone_trainable(model, start_epoch > freeze_backbone_epochs)

    provenance = {
        "experiment_id": config.get("experiment_id"),
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
        "class_weight_power": class_weight_power,
        "ordinal_loss_weight": ordinal_loss_weight,
        "event_sampler": event_sampler_metadata,
        "head_dropout": float(config.get("head_dropout", 0.0)),
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "learning_rates": {
            "backbone": float(config.get("backbone_learning_rate", config["learning_rate"])),
            "head": float(config.get("head_learning_rate", config["learning_rate"])),
        },
        "ema": {
            "enabled": bool(config.get("ema", False)),
            "decay": float(config.get("ema_decay", 0.0)),
            "evaluation": bool(config.get("eval_use_ema", False)),
        },
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
    early_stopping_patience = int(config.get("early_stopping_patience", 0))
    stale_epochs = 0
    stopping_reason = "max_epochs"
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        if freeze_backbone_epochs and epoch == freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
        learning_rates_used = {
            str(group.get("group_name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        epoch_started = time.perf_counter()
        train_loss, seen, train_ce_loss, train_ordinal_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            config,
            amp_enabled,
            amp_dtype,
            ema=ema,
            ordinal_weight=ordinal_loss_weight,
        )
        evaluation_model = ema.module if (ema is not None and bool(config.get("eval_use_ema", True))) else model
        val_metrics = evaluate(evaluation_model, val_loader, device, config, amp_enabled, amp_dtype)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce_loss": train_ce_loss,
            "train_ordinal_loss": train_ordinal_loss,
            "train_rows_seen": seen,
            "sampled_event_counts": event_sampler.last_event_counts if event_sampler is not None else None,
            "learning_rates": learning_rates_used,
            "evaluation_checkpoint": "ema" if evaluation_model is not model else "online",
            "val": val_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        improved = best_val is None or val_metrics["macro_f1"] > best_val["macro_f1"]
        if improved:
            best_epoch = epoch
            best_val = val_metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint = make_checkpoint(
            model,
            ema,
            optimizer,
            scheduler,
            scaler,
            epoch,
            history,
            best_epoch,
            best_val,
            config,
            manifest,
            loader_generator,
        )
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
            stopping_reason = "early_stopping"
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "patience": early_stopping_patience,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    summary = {
        **provenance,
        "status": "complete",
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "best_checkpoint_variant": "ema" if ema is not None and bool(config.get("eval_use_ema", True)) else "online",
        "stopping_reason": stopping_reason,
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

