"""Compare matched B1/B2 training summaries OR matched frozen evaluations.

Rejects mismatched data/configurations and reports B2-minus-B1 differences.
Validation selection scores are not held-out test performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def compare(b1: dict, b2: dict) -> dict:
    if (b1['mode'], b2['mode']) != ('post', 'early'):
        raise ValueError('Expected B1=post and B2=early in that order')
    for key in ('backbone', 'manifest_sha256', 'train_indices_sha256', 'val_indices_sha256', 'events'):
        if b1[key] != b2[key]:
            raise ValueError(f'Uncontrolled comparison: {key} differs')
    a = {k: v for k, v in b1['config'].items() if k not in {'output', 'mode'}}
    b = {k: v for k, v in b2['config'].items() if k not in {'output', 'mode'}}
    if a != b:
        raise ValueError('Uncontrolled comparison: training configurations differ')
    is_evaluation = 'metrics' in b1
    if is_evaluation != ('metrics' in b2):
        raise ValueError('Cannot mix a training summary and an evaluation')
    if is_evaluation:
        for key in ('split', 'evaluated_indices_sha256'):
            if b1[key] != b2[key]:
                raise ValueError(f'Uncontrolled evaluation: {key} differs')
        m1, m2 = b1['metrics'], b2['metrics']
        scope = f"frozen checkpoint {b1['split']} evaluation"
    else:
        for key in ('hardware', 'versions', 'device'):
            if b1[key] != b2[key]:
                raise ValueError(f'Resource comparison requires identical {key}')
        m1, m2 = b1['best_val'], b2['best_val']
        scope = 'best validation selection score (not held-out test performance)'
    result = {'scope': scope, 'seed': a['seed'], 'manifest_sha256': b1['manifest_sha256'],
              'B1': m1, 'B2': m2,
              'B2_minus_B1': {key: m2[key] - m1[key] for key in
                              ('accuracy', 'macro_f1', 'balanced_accuracy')}}
    if not is_evaluation:
        result['resources'] = {key: {'B1': b1[key], 'B2': b2[key]} for key in
                               ('params', 'parameter_bytes', 'training_seconds', 'training_peak_allocated_bytes')}
        result['inference'] = {'B1': b1['inference_benchmark'], 'B2': b2['inference_benchmark']}
        result['benchmark_scope'] = b1['benchmark_scope']
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b1', type=Path, nargs='+', required=True)
    parser.add_argument('--b2', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.b1) != len(args.b2):
        parser.error('Provide one B1 and one B2 result per seed, in matching order')
    if args.output.suffix.lower() != '.json':
        parser.error('--output must end in .json')
    pairs = [(json.loads(a.read_text(encoding='utf-8')), json.loads(b.read_text(encoding='utf-8')))
             for a, b in zip(args.b1, args.b2)]
    comparisons = [compare(a, b) for a, b in pairs]
    if len({entry['seed'] for entry in comparisons}) != len(comparisons):
        raise ValueError('Repeated seed; do not count the same run as independent evidence')
    reference = pairs[0][0]
    for first, second in pairs[1:]:
        for key in ('backbone', 'manifest_sha256', 'events'):
            if first[key] != reference[key]:
                raise ValueError(f'Multi-seed aggregation requires the same {key}')
        for row in (first, second):
            current = {k: v for k, v in row['config'].items() if k not in {'output', 'mode', 'seed'}}
            expected = {k: v for k, v in reference['config'].items() if k not in {'output', 'mode', 'seed'}}
            if current != expected:
                raise ValueError('Multi-seed aggregation requires identical budgets and preprocessing')
    if len({entry['scope'] for entry in comparisons}) != 1:
        raise ValueError('Cannot aggregate different evaluation scopes')
    if len(comparisons) == 1:
        result = comparisons[0]
    else:
        aggregate = {}
        for mode in ('B1', 'B2', 'B2_minus_B1'):
            aggregate[mode] = {}
            for metric in ('accuracy', 'macro_f1', 'balanced_accuracy'):
                values = [entry[mode][metric] for entry in comparisons]
                aggregate[mode][metric] = {'mean': statistics.mean(values),
                                           'sample_std': statistics.stdev(values), 'n': len(values)}
        result = {'scope': comparisons[0]['scope'], 'seeds': [c['seed'] for c in comparisons],
                  'per_seed': comparisons, 'aggregate': aggregate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    with args.output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'B1', 'B2', 'B2_minus_B1', 'scope'])
        for entry in comparisons:
            for key, difference in entry['B2_minus_B1'].items():
                writer.writerow([key, entry['B1'][key], entry['B2'][key], difference,
                                 entry['scope'] + f"; seed={entry['seed']}"])
    print(json.dumps(result.get('aggregate', result.get('B2_minus_B1'))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
