"""Train and validate the mask-aware spatial-difference Swin-T B2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import rasterio
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from stage_b_damage.swin_t_expanded.metrics import metrics_from_predictions
from stage_b_damage.swin_t_b2_maskdiff.dataset import MaskDiffDataset
from stage_b_damage.swin_t_b2_maskdiff.model import build_model, normalize_rgb


ID_TO_LABEL = {0: "no_damage", 1: "minor", 2: "major", 3: "destroyed"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


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
    value = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(value)
    np.random.seed(value)


def load_config(path: Path) -> tuple[dict[str, Any], Path, Path]:
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_config_dir"] = str(path.parent)
    project_root = (path.parent / config["project_root"]).resolve()
    manifest = (project_root / config["manifest"]).resolve()
    return config, project_root, manifest


def make_dataset(
    config: dict[str, Any],
    manifest: Path,
    project_root: Path,
    split: str,
    *,
    max_rows: int = 0,
) -> MaskDiffDataset:
    data = config["preprocessing"]
    return MaskDiffDataset(
        manifest,
        split,
        project_root=project_root,
        crop_size=int(data["crop_size"]),
        context=int(data["context"]),
        alignment=data["alignment"],
        radiometric=data["radiometric"],
        augmentation=data["augmentation"],
        hard_negative={"enabled": False},
        max_rows=max_rows,
        max_open_files=int(config.get("max_open_files", 64)),
        mask_dilate_pixels=int(
            config["model"]["building_mask_head"].get("dilate_pixels", 2)
        ),
    )


def validate_contract(
    config: dict[str, Any],
    manifest: Path,
    project_root: Path,
) -> tuple[MaskDiffDataset, MaskDiffDataset]:
    actual_hash = sha256_file(manifest)
    if actual_hash != config["manifest_sha256"]:
        raise RuntimeError(
            f"Manifest SHA256 mismatch: {actual_hash} != {config['manifest_sha256']}"
        )
    train = make_dataset(config, manifest, project_root, config["train_split"])
    val = make_dataset(config, manifest, project_root, config["val_split"])
    if len(train) != int(config["expected_train_rows"]):
        raise RuntimeError(f"Unexpected train rows: {len(train)}")
    if len(val) != int(config["expected_val_rows"]):
        raise RuntimeError(f"Unexpected val rows: {len(val)}")
    if train.class_counts() != config["expected_train_counts"]:
        raise RuntimeError(f"Unexpected train class counts: {train.class_counts()}")
    if sorted(train.events()) != sorted(config["train_events"]):
        raise RuntimeError(f"Unexpected train events: {train.events()}")
    if sorted(val.events()) != sorted(config["val_events"]):
        raise RuntimeError(f"Unexpected val events: {val.events()}")
    return train, val


def make_loader(
    dataset: MaskDiffDataset,
    config: dict[str, Any],
    *,
    train: bool,
    seed: int,
) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config["batch_size"]),
        "shuffle": bool(train),
        "num_workers": workers,
        "pin_memory": bool(config.get("pin_memory", False)),
        "drop_last": False,
        "worker_init_fn": worker_seed,
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def class_weights(counts: dict[str, int]) -> torch.Tensor:
    total = float(sum(counts.values()))
    values = [
        (total / max(float(counts.get(name, 0)), 1.0))
        for name in ID_TO_LABEL.values()
    ]
    weights = torch.tensor(values, dtype=torch.float32)
    return weights / weights.mean().clamp_min(1e-8)


def dice_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (probability * target).sum()
    denominator = probability.sum() + target.sum()
    return 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)


def focal_mask_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    true_term = target * probability + (1.0 - target) * (1.0 - probability)
    alpha_term = target * alpha + (1.0 - target) * (1.0 - alpha)
    return (alpha_term * (1.0 - true_term).pow(gamma) * bce).mean()


def prepare_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "pre": normalize_rgb(batch["pre"].to(device, non_blocking=True)),
        "post": normalize_rgb(batch["post"].to(device, non_blocking=True)),
        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
        "building_mask": batch["building_mask"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
        "event": list(batch["meta"]["event"]),
        "tile_id": list(batch["meta"]["tile_id"]),
        "building_index": list(batch["meta"]["building_index"]),
    }


def mask_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    prediction = probability >= 0.5
    truth = target >= 0.5
    intersection = float((prediction & truth).sum())
    union = float((prediction | truth).sum())
    dice_denominator = float(prediction.sum() + truth.sum())
    iou = intersection / union if union else 1.0
    dice = (2.0 * intersection / dice_denominator) if dice_denominator else 1.0
    return iou, dice


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    labels_all: list[int] = []
    predictions_all: list[int] = []
    probabilities_all: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    iou_sum = 0.0
    dice_sum = 0.0
    mask_batches = 0
    fallback_count = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = prepare_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    batch["pre"],
                    batch["post"],
                    batch["valid_mask"],
                )
            logits = outputs["logits"].float()
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            labels = batch["label"]
            iou, dice = mask_metrics(
                outputs["mask_probability"].float(),
                batch["building_mask"],
            )
            iou_sum += iou
            dice_sum += dice
            mask_batches += 1
            threshold = float(config["inference"]["minimum_mask_fraction"])
            fallback_count += int((outputs["mask_fraction"] < threshold).sum())
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
            probabilities_all.extend(probabilities.cpu().numpy())
            for index in range(len(labels)):
                row = {
                    "event": batch["event"][index],
                    "tile_id": batch["tile_id"][index],
                    "building_index": batch["building_index"][index],
                    "actual": ID_TO_LABEL[int(labels[index].cpu())],
                    "predicted": ID_TO_LABEL[int(predictions[index].cpu())],
                }
                row.update(
                    {
                        f"p_{name}": float(probabilities[index, class_index].cpu())
                        for class_index, name in ID_TO_LABEL.items()
                    }
                )
                rows.append(row)
    metrics = metrics_from_predictions(
        labels_all,
        predictions_all,
        np.asarray(probabilities_all, dtype=np.float32),
    )
    metrics["mask_iou"] = iou_sum / max(mask_batches, 1)
    metrics["mask_dice"] = dice_sum / max(mask_batches, 1)
    metrics["mask_fallback_fraction"] = fallback_count / max(len(rows), 1)
    return metrics, rows


def rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader_generator.get_state(),
    }


def restore_rng(state: dict[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    loader_generator.set_state(state["loader"].cpu())


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_metrics: dict[str, Any] | None,
    config: dict[str, Any],
    manifest_sha256: str,
    loader_generator: torch.Generator,
) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "history": history,
            "best_epoch": int(best_epoch),
            "best_val": best_metrics,
            "config": config,
            "config_sha256": stable_hash(config),
            "manifest_sha256": manifest_sha256,
            "rng": rng_state(loader_generator),
        },
        temporary,
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
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
            config[name] = int(value)
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))
    full_train, full_val = validate_contract(config, manifest, project_root)

    if args.preflight_only:
        sample = full_train[0]
        if sample["building_mask"].sum() <= 0:
            raise RuntimeError("First sample has an empty target building mask")
        model, model_report = build_model(config)
        model = model.to(device).eval()
        with torch.inference_mode():
            pre = normalize_rgb(sample["pre"].unsqueeze(0).to(device))
            post = normalize_rgb(sample["post"].unsqueeze(0).to(device))
            valid = sample["valid_mask"].unsqueeze(0).to(device)
            outputs = model(pre, post, valid)
        if tuple(outputs["logits"].shape) != (1, int(config["model"]["num_classes"])):
            raise RuntimeError(f"Unexpected logits shape: {tuple(outputs['logits'].shape)}")
        if tuple(outputs["mask_logits"].shape[-2:]) != (
            int(config["preprocessing"]["crop_size"]),
            int(config["preprocessing"]["crop_size"]),
        ):
            raise RuntimeError(f"Unexpected mask shape: {tuple(outputs['mask_logits'].shape)}")
        print(
            json.dumps(
                {
                    "status": "preflight_dataset_ok",
                    "train_rows": len(full_train),
                    "val_rows": len(full_val),
                    "mask_pixels": int(sample["building_mask"].sum()),
                    "logits_shape": list(outputs["logits"].shape),
                    "mask_shape": list(outputs["mask_logits"].shape),
                    "model_report": model_report,
                },
                indent=2,
            )
        )
        return 0

    train_ds = (
        make_dataset(
            config,
            manifest,
            project_root,
            config["train_split"],
            max_rows=int(config.get("max_train", 0)),
        )
        if int(config.get("max_train", 0)) > 0
        else full_train
    )
    val_ds = (
        make_dataset(
            config,
            manifest,
            project_root,
            config["val_split"],
            max_rows=int(config.get("max_val", 0)),
        )
        if int(config.get("max_val", 0)) > 0
        else full_val
    )

    output_dir = (
        args.output_dir or (args.config.parent / config["output_dir"])
    ).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(
        train_ds,
        config,
        train=True,
        seed=int(config["seed"]),
    )
    val_loader = make_loader(
        val_ds,
        config,
        train=False,
        seed=int(config["seed"]) + 1,
    )
    model, model_report = build_model(config)
    model = model.to(device)
    if bool(config.get("channels_last", False)):
        model = model.to(memory_format=torch.channels_last)
    weights = class_weights(train_ds.class_counts()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["epochs"])),
    )
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.float16 if config.get("amp_dtype", "float16") == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    manifest_sha256 = sha256_file(manifest)
    provenance = {
        "experiment_id": config["experiment_id"],
        "config": config,
        "config_sha256": stable_hash(config),
        "manifest_sha256": manifest_sha256,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "class_counts": train_ds.class_counts(),
        "class_weights": weights.detach().cpu().tolist(),
        "model_report": model_report,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    start_epoch = 1
    loader_generator = train_loader.generator
    if loader_generator is None:
        raise RuntimeError("Training loader generator missing")
    if args.resume:
        checkpoint = torch.load(
            output_dir / "last.pt",
            map_location=device,
            weights_only=False,
        )
        if checkpoint["manifest_sha256"] != manifest_sha256:
            raise RuntimeError("Resume manifest hash mismatch")
        if checkpoint["config_sha256"] != stable_hash(config):
            raise RuntimeError("Resume config differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint["history"]
        best_metrics = checkpoint["best_val"]
        best_epoch = int(checkpoint["best_epoch"])
        restore_rng(checkpoint["rng"], loader_generator)
        start_epoch = int(checkpoint["epoch"]) + 1

    accumulation = max(1, int(config.get("gradient_accumulation_steps", 1)))
    start_time = time.perf_counter()
    stale_epochs = 0
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_ce = 0.0
        running_dice = 0.0
        running_focal = 0.0
        seen = 0
        epoch_start = time.perf_counter()
        for step, raw_batch in enumerate(train_loader, start=1):
            batch = prepare_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    batch["pre"],
                    batch["post"],
                    batch["valid_mask"],
                )
                ce = criterion(outputs["logits"].float(), batch["label"])
                mask_probability = torch.sigmoid(outputs["mask_logits"].float())
                dice = dice_loss(mask_probability, batch["building_mask"])
                focal = focal_mask_loss(outputs["mask_logits"].float(), batch["building_mask"])
                loss_weights = config["loss"]
                loss = (
                    float(loss_weights["classification_weight"]) * ce
                    + float(loss_weights["mask_dice_weight"]) * dice
                    + float(loss_weights["mask_focal_weight"]) * focal
                )
            scaler.scale(loss / accumulation).backward()
            running_loss += float(loss.detach()) * len(batch["label"])
            running_ce += float(ce.detach()) * len(batch["label"])
            running_dice += float(dice.detach()) * len(batch["label"])
            running_focal += float(focal.detach()) * len(batch["label"])
            seen += len(batch["label"])
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        val_metrics, val_rows = evaluate(
            model,
            val_loader,
            device,
            config,
            amp_enabled,
            amp_dtype,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "train_ce": running_ce / max(seen, 1),
            "train_mask_dice": running_dice / max(seen, 1),
            "train_mask_focal": running_focal / max(seen, 1),
            "val": val_metrics,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        improved = best_metrics is None or val_metrics["macro_f1"] > best_metrics["macro_f1"]
        if improved:
            best_metrics = val_metrics
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            history=history,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            config=config,
            manifest_sha256=manifest_sha256,
            loader_generator=loader_generator,
        )
        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                history=history,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                config=config,
                manifest_sha256=manifest_sha256,
                loader_generator=loader_generator,
            )
            with (output_dir / "val_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                import csv

                writer = csv.DictWriter(handle, fieldnames=list(val_rows[0]))
                writer.writeheader()
                writer.writerows(val_rows)
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        patience = int(config.get("early_stopping_patience", 0))
        if patience > 0 and stale_epochs >= patience:
            break

    summary = {
        **provenance,
        "status": "complete",
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_val": best_metrics,
        "stopping_reason": "max_epochs",
        "history": history,
        "training_seconds": time.perf_counter() - start_time,
        "test_evaluated": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
