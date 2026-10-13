"""Train a small, reproducible xBD damage-classification baseline.

This is deliberately a lightweight B1/B2 smoke-test model. B1 uses only the
post-disaster crop; B2 concatenates pre- and post-disaster crops as six input
channels. The manifest and event split remain fixed across both modes.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


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


def choose_indices(length: int, limit: int, seed: int) -> list[int]:
    if not limit or limit >= length:
        return list(range(length))
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator)[:limit].tolist()


def make_input(batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "post":
        return batch["post"]
    return torch.cat((batch["pre"], batch["post"]), dim=1)


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
            for target, prediction in zip(actual, predicted):
                confusion[int(target), int(prediction)] += 1
    per_class = {}
    f1_values = []
    for index, name in enumerate(("no_damage", "minor", "major", "destroyed")):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}
    total = int(confusion.sum())
    return {
        "n": total,
        "accuracy": float(confusion.diag().sum()) / total if total else 0.0,
        "macro_f1": sum(f1_values) / 4,
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    train_full = XBDPairedDataset(
        args.manifest,
        "train",
        project_root=args.project_root,
        crop_size=args.crop_size,
        context=args.context,
    )
    val_full = XBDPairedDataset(
        args.manifest,
        "val",
        project_root=args.project_root,
        crop_size=args.crop_size,
        context=args.context,
    )
    train_indices = choose_indices(len(train_full), args.max_train, args.seed)
    val_indices = choose_indices(len(val_full), args.max_val, args.seed + 1)
    train = Subset(train_full, train_indices)
    val = Subset(val_full, val_indices)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    counts = Counter(train_full.rows[index]["damage"] for index in train_indices)
    weights = torch.tensor(
        [1.0 / max(counts.get(name, 0), 1) for name in LABEL_TO_ID], dtype=torch.float32
    )
    weights = weights / weights.mean()
    model = SmallCNN(3 if args.mode == "post" else 6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(make_input(batch, args.mode).to(device))
            labels = batch["label"].to(device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            seen += labels.numel()
        metrics = evaluate(model, val_loader, device, args.mode)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val": metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "mode": args.mode,
            "label_to_id": LABEL_TO_ID,
            "args": vars(args),
            "history": history,
        },
        args.output.with_suffix(".pt"),
    )
    args.output.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "device": str(device),
                "train_rows": len(train),
                "val_rows": len(val),
                "class_counts": dict(counts),
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
