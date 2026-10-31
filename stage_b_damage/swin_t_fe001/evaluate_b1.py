"""Evaluate a saved Swin-T B1 checkpoint on FE-001 val or sealed test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import XBDPostDataset
from metrics import LABELS, metrics_from_predictions
from modeling import build_model
from train_b1 import (
    choose_device,
    load_config,
    make_amp,
    make_loader,
    prepare_images,
    sha256_file,
)


def evaluate_export(model, loader: DataLoader, device: torch.device, config: dict):
    amp_enabled, amp_dtype, _ = make_amp(device, config)
    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    rows: list[dict] = []
    with torch.inference_mode():
        for batch in loader:
            images = prepare_images(batch["image"], device, config)
            if amp_enabled:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(images)
            else:
                logits = model(images)
            probabilities = torch.softmax(logits.float(), dim=1).cpu()
            predictions = probabilities.argmax(dim=1)
            labels = batch["label"]
            actual.extend(labels.tolist())
            predicted.extend(predictions.tolist())
            meta = batch.get("meta", {})
            for index in range(len(labels)):
                row = {
                    "event": meta.get("event", [""])[index],
                    "tile_id": meta.get("tile_id", [""])[index],
                    "building_index": meta.get("building_index", [""])[index],
                    "actual": LABELS[int(labels[index])],
                    "predicted": LABELS[int(predictions[index])],
                }
                for class_index, name in enumerate(LABELS):
                    row[f"p_{name}"] = float(probabilities[index, class_index])
                rows.append(row)
    return metrics_from_predictions(actual, predicted, torch.tensor(
        [
            [row[f"p_{name}"] for name in LABELS]
            for row in rows
        ], dtype=torch.float32
    ).numpy()), rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("..\..\configs\swin_t_fe001_b1.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise RuntimeError("Test evaluation is guarded; pass --allow-test only after protocol freeze")

    config, project_root, manifest = load_config(args.config)
    if sha256_file(manifest) != config["manifest_sha256"]:
        raise RuntimeError("Manifest SHA256 does not match config")
    device = choose_device(args.device or config["device"])
    dataset = XBDPostDataset(
        manifest,
        config["test_split"] if args.split == "test" else config["val_split"],
        project_root=project_root,
        crop_size=config["crop_size"],
        context=config["context"],
        return_meta=True,
        max_open_files=config["max_open_files"],
    )
    loader = make_loader(dataset, config, train=False, seed=int(config["seed"]) + 17)
    model, _ = build_model(num_classes=int(config["num_classes"]), pretrained=False)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    if config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    metrics, rows = evaluate_export(model, loader, device, config)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({
            "split": args.split,
            "checkpoint": str(args.checkpoint.resolve()),
            "manifest_sha256": config["manifest_sha256"],
            "metrics": metrics,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

