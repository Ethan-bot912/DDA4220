"""Evaluate a frozen Phase 3 checkpoint; validation by default.

Use --split test --final-test only after the experiment protocol is fixed.
The saved manifest checksum must match. Writes metrics, per-event scores,
predictions and a confusion-matrix PNG; does not change or select weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path.cwd() / 'output' / 'matplotlib-cache'))
from stage_b_damage.dataset import LABEL_TO_ID, XBDPairedDataset
from stage_b_damage.train_baseline import (audit_splits, build_model, choose_indices,
                                          make_input, metrics_from_confusion)
import torch
from torch.utils.data import DataLoader, Subset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--final-test', action='store_true')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.json':
        parser.error('--output must end in .json')
    if args.split == 'test' and not args.final_test:
        parser.error('Held-out test requires --final-test after all model choices are frozen')
    if args.output.exists():
        parser.error('Output exists; choose a new path to preserve evaluation history')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if checkpoint.get('manifest_sha256') != manifest_hash:
        raise ValueError('Manifest differs from training, or checkpoint lacks provenance; retrain with current code')
    if checkpoint['label_to_id'] != LABEL_TO_ID:
        raise ValueError('Checkpoint class mapping differs from the dataset')
    config = checkpoint['args']
    if args.split == 'test' and (config['max_train'] or config['max_val']):
        raise ValueError('A limited smoke/pilot run cannot be used as the final test model')
    events = audit_splits(args.manifest)
    dataset = XBDPairedDataset(args.manifest, args.split, project_root=args.project_root,
                              crop_size=config['crop_size'], context=config['context'], return_meta=True)
    indices = (choose_indices(len(dataset), config['max_val'], config['seed'] + 1)
               if args.split == 'val' else list(range(len(dataset))))
    loader = DataLoader(Subset(dataset, indices), batch_size=config['batch_size'], shuffle=False, num_workers=0)
    model = build_model(checkpoint['backbone'], checkpoint['mode']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    confusion = torch.zeros((4, 4), dtype=torch.long)
    grouped = {}
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            probability = model(make_input(batch, checkpoint['mode']).to(device)).softmax(1).cpu()
            predicted = probability.argmax(1)
            actual = batch['label']
            confusion += torch.bincount(actual * 4 + predicted, minlength=16).reshape(4, 4)
            for i, (target, prediction) in enumerate(zip(actual.tolist(), predicted.tolist())):
                event = batch['meta']['event'][i]
                grouped.setdefault(event, torch.zeros((4, 4), dtype=torch.long))[target, prediction] += 1
                predictions.append({'event': event, 'tile_id': batch['meta']['tile_id'][i],
                                    'building_index': batch['meta']['building_index'][i],
                                    'actual': list(LABEL_TO_ID)[target],
                                    'predicted': list(LABEL_TO_ID)[prediction],
                                    'confidence': float(probability[i, prediction]),
                                    'correct': target == prediction})
    dataset.close()
    metrics = metrics_from_confusion(confusion)
    result = {'mode': checkpoint['mode'], 'backbone': checkpoint['backbone'],
              'split': args.split, 'metrics': metrics, 'config': config,
              'manifest_sha256': manifest_hash, 'events': events,
              'train_indices_sha256': checkpoint['train_indices_sha256'],
              'val_indices_sha256': checkpoint['val_indices_sha256'],
              'evaluated_indices_sha256': hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'checkpoint_epoch': checkpoint['epoch'], 'test_evaluated': args.split == 'test',
              'per_event': {event: metrics_from_confusion(matrix) for event, matrix in grouped.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    with args.output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    plot = ax.imshow(confusion.numpy(), cmap='Blues')
    ax.set(xticks=range(4), yticks=range(4), xticklabels=list(LABEL_TO_ID),
           yticklabels=list(LABEL_TO_ID), xlabel='Predicted label', ylabel='True label',
           title=f"{checkpoint['mode']} / {args.split} / epoch {checkpoint['epoch']}")
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(int(confusion[i, j])), ha='center', va='center',
                    color='white' if confusion[i, j] > confusion.max() / 2 else 'black')
    fig.colorbar(plot, ax=ax)
    fig.savefig(args.output.with_suffix('.png'), dpi=160)
    plt.close(fig)
    print(json.dumps({'split': args.split, 'metrics': metrics}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
