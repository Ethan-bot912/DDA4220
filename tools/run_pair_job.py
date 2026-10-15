"""Logged, persistent pilot/formal B1/B2 runner for the download heartbeat."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('pilot', 'formal'), required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    for key in ('TEMP', 'TMP'):
        os.environ[key] = 'D:/DDA4220/tmp'
    os.environ['MPLCONFIGDIR'] = 'D:/DDA4220/tmp/matplotlib'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    state = ROOT / 'output' / 'automation_xbd' / f'{args.phase}_state.json'
    output_root = Path('output/real_pilot' if args.phase == 'pilot' else 'output/resnet18')
    stage = 'starting'

    def record(status, **details):
        data = dict(pid=os.getpid(), phase=args.phase, stage=stage, status=status,
                    updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details)
        temporary = state.with_suffix('.partial')
        temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
        temporary.replace(state)

    try:
        for mode in ('post', 'early'):
            stage = mode
            output = output_root / f'{mode}_seed42.json'
            if output.exists() or output.with_suffix('.pt').exists():
                raise FileExistsError(f'Prior output exists: {output}; inspect instead of overwriting')
            command = [sys.executable, '-m', 'stage_b_damage.train_baseline',
                       '--manifest', 'dataset/pilot_tier3_4events/manifest_classification.csv',
                       '--project-root', str(ROOT), '--output', str(output),
                       '--backbone', 'resnet18', '--mode', mode,
                       '--epochs', '1' if args.phase == 'pilot' else '20',
                       '--batch-size', '8' if args.phase == 'pilot' else '32',
                       '--lr', '0.001', '--crop-size', '128', '--context', '16',
                       '--workers', '0', '--seed', '42', '--device', 'cuda']
            if args.phase == 'pilot':
                command += ['--max-train', '128', '--max-val', '64']
            record('running', command=command)
            print(f'=== {args.phase}: {mode} ===', flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        stage = 'validation_comparison'
        command = [sys.executable, '-m', 'evaluation.compare_pair',
                   '--b1', str(output_root / 'post_seed42.json'),
                   '--b2', str(output_root / 'early_seed42.json'),
                   '--output', str(output_root / 'validation_comparison.json')]
        record('running', command=command)
        subprocess.run(command, cwd=ROOT, check=True)
        record('complete', output_root=str(output_root))
        return 0
    except Exception as exc:
        record('failed', error=repr(exc))
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
