"""Check proposed crop/batch capacity on this GPU using synthetic tensors only.

This measures software/hardware readiness, not real data throughput or accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage_b_damage.train_baseline import build_model, synchronize
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--crop-size', type=int, default=128)
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'capacity_check.json')
    args = parser.parse_args()
    if args.batch_size < 2 or args.crop_size < 32:
        parser.error('Use batch-size >= 2 and crop-size >= 32 for this training check')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    results = []
    device = torch.device('cuda')
    for mode in ('post', 'early'):
        torch.manual_seed(42)
        model = build_model('resnet18', mode).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x = torch.rand(args.batch_size, 3 if mode == 'post' else 6,
                       args.crop_size, args.crop_size, device=device)
        labels = torch.arange(args.batch_size, device=device) % 4
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), labels)
            loss.backward()
            optimizer.step()
        synchronize(device)
        results.append({'mode': mode, 'batch_size': args.batch_size, 'crop_size': args.crop_size,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'forward_backward_optimizer_verified': True})
        del loss, x, labels, optimizer, model
        torch.cuda.empty_cache()
    report = {'synthetic_only': True, 'gpu': torch.cuda.get_device_name(0), 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
