"""Recalculate capacity budgets from measured profiles and explicit workload sizes.

Reads published evidence only. Does not choose events, extract data or train.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'reports/expansion_planning_20261015'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-rows', type=int, required=True)
    parser.add_argument('--val-rows', type=int, required=True)
    parser.add_argument('--test-rows', type=int, default=0,
                        help='Optional later inference workload; no inference is run')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.train_rows, args.val_rows, args.batch_size, args.epochs) <= 0 or args.test_rows < 0:
        parser.error('Training/validation sizes, batch and epochs must be positive; inference size cannot be negative')
    if args.batch_size != 32:
        parser.error('The recorded profile measured batch32; remeasure before using another batch size')
    if args.output.exists():
        raise FileExistsError('Existing budget preserved; choose a new output path')

    budget = load(EVIDENCE / 'resource_budget.json')
    profile = load(EVIDENCE / 'profile.json')
    budget['workload_assumptions'] = {
        'train_rows_per_epoch': args.train_rows,
        'val_rows_per_epoch': args.val_rows,
        'optional_inference_rows': args.test_rows,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'scope': 'Capacity calculation inputs only; does not assign events to datasets',
    }
    budget['calculated_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    budget['measurement_date'] = '2026-10-15'
    budget['free_bytes_now'] = shutil.disk_usage(ROOT).free
    budget['additional_free_space_required_bytes'] = max(
        0, budget['desired_free_space_before_expansion_bytes'] - budget['free_bytes_now'])
    timing = []
    for measured in profile['results']:
        mode = measured['mode']
        train_time = measured['train_batch']['mean_seconds']
        infer_time = measured['eval_batch']['mean_seconds']
        epoch = math.ceil(args.train_rows / 32) * train_time + math.ceil(args.val_rows / 32) * infer_time
        historical = load(ROOT / f'reports/four_event_seed42/{mode}_seed42.json')
        old_estimate = 20 * (math.ceil(17552 / 32) * train_time + math.ceil(22372 / 32) * infer_time)
        calibration = max(1, historical['training_seconds'] / old_estimate)
        base = args.epochs * epoch * calibration
        timing.append({
            'mode': mode,
            'measured_expanded_epoch_seconds': epoch,
            'historical_calibration_multiplier': calibration,
            'baseline_20_epoch_seconds': 20 * epoch * calibration,
            'planning_20_epoch_lower_seconds': 20 * epoch * calibration * 1.2,
            'planning_20_epoch_upper_seconds': 20 * epoch * calibration * 1.8,
            'requested_epochs_lower_seconds': base * 1.2,
            'requested_epochs_upper_seconds': base * 1.8,
            'independent_val_seconds': math.ceil(args.val_rows / 32) * infer_time * calibration,
            'final_test_seconds_if_later_authorized': math.ceil(args.test_rows / 32) * infer_time * calibration,
        })
    budget['timing'] = timing
    budget['two_models_one_seed_lower_hours'] = sum(r['requested_epochs_lower_seconds'] for r in timing) / 3600
    budget['two_models_one_seed_upper_hours'] = sum(r['requested_epochs_upper_seconds'] for r in timing) / 3600
    budget['three_seed_lower_hours'] = budget['two_models_one_seed_lower_hours'] * 3
    budget['three_seed_upper_hours'] = budget['two_models_one_seed_upper_hours'] * 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(budget, indent=2) + '\n')
    print(json.dumps(budget, indent=2))


if __name__ == '__main__':
    main()
