"""Small seed experiment: same training run, old loader vs multi-worker loader.

The point is to show that parallelising the *data path* speeds up a real
training run without changing the model, the sampling order or the tensors that
reach the GPU. Both arms:

* build ResNet-18 with the same seed (random init, as in the published runs);
* iterate the same batches in the same order (DataLoader shuffle is driven by a
  seeded generator in the main process);
* train for a fixed number of steps and evaluate on the same validation subset.

Only ``num_workers`` differs, so a speed-up cannot come from a different
experiment.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset

from stage_b_damage.dataset import LABEL_TO_ID, XBDPairedDataset


CLASSES = ("no_damage", "minor", "major", "destroyed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("post", "early"), default="post")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--fast-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_model(in_channels: int) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 4)
    return model


def inputs_of(batch: dict, mode: str) -> torch.Tensor:
    return batch["post"] if mode == "post" else torch.cat((batch["pre"], batch["post"]), dim=1)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, mode: str) -> dict:
    model.eval()
    confusion = torch.zeros((4, 4), dtype=torch.long)
    for batch in loader:
        logits = model(inputs_of(batch, mode).to(device, non_blocking=True))
        for target, prediction in zip(batch["label"].cpu(), logits.argmax(1).cpu()):
            confusion[int(target), int(prediction)] += 1
    per_class = {}
    f1_values = []
    recalls = []
    for index, name in enumerate(CLASSES):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        recalls.append(recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}
    total = int(confusion.sum())
    return {
        "n": total,
        "accuracy": float(confusion.diag().sum()) / total if total else 0.0,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def run_arm(name: str, workers: int, args, train_ds, val_ds, val_indices, weights, device) -> dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(val_ds, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        pin_memory=device.type == "cuda",
    )
    model = build_model(3 if args.mode == "post" else 6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    model.train()
    losses = []
    seen = 0
    started = time.perf_counter()
    iterator = iter(train_loader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        inputs = inputs_of(batch, args.mode).to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        seen += labels.numel()
    train_seconds = time.perf_counter() - started

    eval_started = time.perf_counter()
    metrics = evaluate(model, val_loader, device, args.mode)
    eval_seconds = time.perf_counter() - eval_started
    return {
        "arm": name,
        "workers": workers,
        "steps": args.steps,
        "samples": seen,
        "train_seconds": train_seconds,
        "train_samples_per_second": seen / train_seconds,
        "step_ms": 1000.0 * train_seconds / args.steps,
        "mean_train_loss": float(np.mean(losses)),
        "last_train_loss": losses[-1],
        "eval_seconds": eval_seconds,
        "val": metrics,
    }


def main() -> int:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.project_root = args.project_root.resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_ds = XBDPairedDataset(
        args.manifest,
        "train",
        project_root=args.project_root,
        crop_size=args.crop_size,
        context=args.context,
    )
    val_ds = XBDPairedDataset(
        args.manifest,
        "val",
        project_root=args.project_root,
        crop_size=args.crop_size,
        context=args.context,
    )
    step = max(1, len(val_ds) // args.val_samples)
    val_indices = list(range(0, len(val_ds), step))[: args.val_samples]

    counts = Counter(row["damage"] for row in train_ds.rows)
    weights = torch.tensor(
        [1.0 / max(counts.get(name, 0), 1) for name in LABEL_TO_ID], dtype=torch.float32
    )
    weights = weights / weights.mean()

    slow = run_arm("baseline workers=0", 0, args, train_ds, val_ds, val_indices, weights, device)
    fast = run_arm(
        f"workers={args.fast_workers}", args.fast_workers, args, train_ds, val_ds, val_indices, weights, device
    )

    report = {
        "manifest": str(args.manifest),
        "mode": args.mode,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": platform.python_version(),
        "batch_size": args.batch_size,
        "crop_size": args.crop_size,
        "seed": args.seed,
        "arms": [slow, fast],
        "speedup": slow["train_seconds"] / fast["train_seconds"],
        "metric_delta": {
            "macro_f1": fast["val"]["macro_f1"] - slow["val"]["macro_f1"],
            "accuracy": fast["val"]["accuracy"] - slow["val"]["accuracy"],
            "mean_train_loss": fast["mean_train_loss"] - slow["mean_train_loss"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
