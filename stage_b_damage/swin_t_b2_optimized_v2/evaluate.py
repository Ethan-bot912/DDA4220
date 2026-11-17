"""Evaluate an optimized B2 checkpoint and export logits for calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from model import build_model
    from train import choose_device, evaluate, load_config, make_dataset, make_loader, seed_everything, write_prediction_csv
except ImportError:  # pragma: no cover
    from .model import build_model
    from .train import choose_device, evaluate, load_config, make_dataset, make_loader, seed_everything, write_prediction_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config_b2_optimized.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "calibration", "val", "test"), default="val")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise RuntimeError("Test evaluation is sealed; pass --allow-test only after protocol freeze")
    config, project_root, manifest = load_config(args.config)
    if args.manifest is not None:
        manifest = args.manifest.resolve()
    if args.device is not None:
        config["device"] = args.device
    seed_everything(int(config.get("seed", 42)), bool(config.get("deterministic", True)))
    device = choose_device(str(config.get("device", "auto")))
    model, model_report = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("ema_model") or checkpoint.get("model")
    if state is None:
        raise RuntimeError(f"No model state found in {args.checkpoint}")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    # Evaluation and hard-negative mining must be deterministic, including when
    # the requested split is ``train``.
    eval_config = dict(config)
    eval_config["augmentation"] = {"enabled": False}
    dataset = make_dataset(eval_config, manifest, project_root, args.split, args.max_rows, set())
    loader = make_loader(
        dataset,
        eval_config,
        train=False,
        generator=torch.Generator().manual_seed(int(config.get("seed", 42)) + 1),
    )
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.float16 if config.get("amp_dtype", "float16") == "float16" else torch.bfloat16
    metrics, rows = evaluate(model, loader, device, eval_config, amp_enabled, amp_dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_csv(rows, args.output_dir / f"predictions_{args.split}.csv")
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "rows": len(dataset),
        "metrics": metrics,
        "model_report": model_report,
        "test_evaluated": args.split == "test",
    }
    (args.output_dir / f"metrics_{args.split}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
