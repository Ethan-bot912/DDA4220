"""Train controlled B1/B2 ResNet-18 classifiers on event-disjoint xBD crops.

SmallCNN remains available explicitly for pipeline smoke tests. Test rows are
never evaluated here; checkpoint selection uses validation Macro F1 only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

# Required by deterministic CUDA matrix multiplication; set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import LABEL_TO_ID, XBDPairedDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("post", "early"), default="post")
    parser.add_argument("--backbone", choices=("resnet18", "smallcnn"), default="resnet18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.crop_size, args.benchmark_steps) <= 0:
        parser.error("epochs, batch-size, crop-size and benchmark-steps must be positive")
    if min(args.context, args.workers, args.max_train, args.max_val, args.benchmark_warmup) < 0:
        parser.error("context, workers, limits and benchmark-warmup must be nonnegative")
    if args.lr <= 0:
        parser.error("lr must be positive")
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end in .json (checkpoint is saved separately as .pt)")
    return args


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def build_model(backbone: str, mode: str) -> nn.Module:
    if backbone == "smallcnn":
        return SmallCNN(3 if mode == "post" else 6)
    from torchvision.models import resnet18

    # Build the same complete three-channel network before expanding B2.
    # No external/pretrained weights: both runs start from shared random weights.
    model = resnet18(weights=None, num_classes=4)
    if mode == "early":
        original = model.conv1
        expanded = nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            expanded.weight.copy_(torch.cat((original.weight, original.weight), dim=1) / 2)
        model.conv1 = expanded
    return model


def audit_splits(manifest: Path) -> dict[str, list[str]]:
    """Fail before training if an event or paired tile crosses data splits."""
    owners: dict[tuple[str, ...], str] = {}
    events: dict[str, set[str]] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unexpected split: {split!r}")
            events.setdefault(split, set()).add(row["event"])
            for key in (("event", row["event"]), ("tile", row["event"], row["tile_id"])):
                if key in owners and owners[key] != split:
                    raise ValueError(f"Data leakage across splits: {key}")
                owners[key] = split
    return {split: sorted(names) for split, names in events.items()}


def choose_indices(length: int, limit: int, seed: int) -> list[int]:
    if not limit or limit >= length:
        return list(range(length))
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator)[:limit].tolist()


def make_input(batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "post":
        return batch["post"]
    return torch.cat((batch["pre"], batch["post"]), dim=1)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, object]:
    per_class = {}
    f1_values = []
    recalls = []
    for index, name in enumerate(LABEL_TO_ID):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        if tp + fn:
            recalls.append(recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": int(tp + fn)}
    total = int(confusion.sum())
    if not total:
        raise ValueError("Cannot evaluate an empty validation set")
    return {
        "n": total,
        "accuracy": float(confusion.diag().sum()) / total,
        "macro_f1": sum(f1_values) / 4,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
) -> dict[str, object]:
    model.eval()
    confusion = torch.zeros((4, 4), dtype=torch.long)
    with torch.inference_mode():
        for batch in loader:
            logits = model(make_input(batch, mode).to(device))
            predicted = logits.argmax(dim=1).cpu()
            actual = batch["label"].cpu()
            confusion += torch.bincount(actual * 4 + predicted, minlength=16).reshape(4, 4)
    return metrics_from_confusion(confusion)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model: nn.Module, device: torch.device, args: argparse.Namespace) -> list[dict]:
    """Resident float32 tensors; forward only, excluding disk I/O and transfer."""
    model.eval()
    records = []
    with torch.inference_mode():
        for batch_size in sorted({1, args.batch_size}):
            x = torch.zeros(batch_size, 3 if args.mode == "post" else 6,
                            args.crop_size, args.crop_size, device=device)
            for _ in range(args.benchmark_warmup):
                model(x)
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for _ in range(args.benchmark_steps):
                model(x)
            synchronize(device)
            seconds = (time.perf_counter() - started) / args.benchmark_steps
            records.append({"batch_size": batch_size, "batch_latency_ms": seconds * 1000,
                            "amortized_ms_per_image": seconds * 1000 / batch_size,
                            "images_per_second": batch_size / seconds,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda" else None})
            del x
    return records


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    import numpy as np
    import rasterio
    import torchvision
    import platform
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    events = audit_splits(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()

    train_full = XBDPairedDataset(
        args.manifest,
        "train",
        project_root=args.project_root,
        crop_size=args.crop_size,
        context=args.context,
        cache_dir=args.cache_dir,
    )
    val_full = XBDPairedDataset(
        args.manifest,
        "val",
        project_root=args.project_root,
        crop_size=args.crop_size,
        context=args.context,
        cache_dir=args.cache_dir,
    )
    train_indices = choose_indices(len(train_full), args.max_train, args.seed)
    val_indices = choose_indices(len(val_full), args.max_val, args.seed + 1)
    index_hashes = {
        "train_indices_sha256": hashlib.sha256(json.dumps(train_indices).encode()).hexdigest(),
        "val_indices_sha256": hashlib.sha256(json.dumps(val_indices).encode()).hexdigest(),
    }
    train = Subset(train_full, train_indices)
    val = Subset(val_full, val_indices)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed)
    )
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    counts = Counter(train_full.rows[index]["damage"] for index in train_indices)
    weights = torch.tensor(
        [1.0 / max(counts.get(name, 0), 1) for name in LABEL_TO_ID], dtype=torch.float32
    )
    weights = weights / weights.mean()
    model = build_model(args.backbone, args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    history: list[dict[str, object]] = []
    best_f1 = -1.0
    best_epoch = 0
    best_metrics = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        synchronize(device)
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(make_input(batch, args.mode).to(device))
            labels = batch["label"].to(device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_weight = float(criterion.weight[labels].sum())
            total_loss += float(loss.detach()) * batch_weight
            total_weight += batch_weight
        metrics = evaluate(model, val_loader, device, args.mode)
        synchronize(device)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_weight, 1e-12),
            "val": metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        scheduler.step()
        history.append(record)
        # Strict improvement: retain earliest checkpoint on a Macro F1 tie.
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_metrics = metrics
            torch.save({"model": model.state_dict(), "mode": args.mode,
                        "backbone": args.backbone, "label_to_id": LABEL_TO_ID,
                        "args": config, "epoch": epoch, "val": metrics,
                        "manifest_sha256": manifest_sha256, **index_hashes},
                       args.output.with_suffix(".pt"))
        print(json.dumps(record), flush=True)

    synchronize(device)
    training_seconds = time.perf_counter() - training_started
    training_peak = (torch.cuda.max_memory_allocated(device)
                     if device.type == "cuda" else None)
    # Release gradients/optimizer state so inference peaks exclude training state.
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler, loss, logits
    checkpoint = torch.load(args.output.with_suffix(".pt"), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    efficiency = benchmark(model, device, args)
    args.output.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "backbone": args.backbone,
                "config": config,
                "device": str(device),
                "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "torchvision": torchvision.__version__, "numpy": np.__version__,
                             "rasterio": rasterio.__version__},
                "events": events,
                "manifest_sha256": manifest_sha256,
                **index_hashes,
                "train_rows": len(train),
                "val_rows": len(val),
                "class_counts": dict(counts),
                "class_weights": weights.tolist(),
                "history": history,
                "selection_metric": "validation macro_f1 (four fixed classes)",
                "best_epoch": best_epoch,
                "best_val": best_metrics,
                "params": sum(p.numel() for p in model.parameters()),
                "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                "training_seconds": training_seconds,
                "training_peak_allocated_bytes": training_peak,
                "inference_benchmark": efficiency,
                "benchmark_scope": "float32 forward only, resident synthetic tensors; excludes I/O and transfers",
                "initialization": ("random, shared 3-channel ResNet; B2 conv1 duplicates B1 weights / 2"
                                   if args.backbone == "resnet18" else "random SmallCNN smoke model"),
                "normalization": "clip to [0,255], divide by 255; no ImageNet normalization",
                "test_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
