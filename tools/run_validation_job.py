"""Restore both completed formal checkpoints and verify validation results."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    os.environ.update(TEMP='D:/DDA4220/tmp', TMP='D:/DDA4220/tmp',
                      MPLCONFIGDIR='D:/DDA4220/tmp/matplotlib',
                      PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    state = ROOT / 'output/automation_xbd/validation_state.json'
    outputs = ROOT / 'output/resnet18'
    stage = 'starting'

    def record(status, **details):
        data = dict(pid=os.getpid(), stage=stage, status=status,
                    updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details)
        temporary = state.with_suffix('.partial')
        temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
        temporary.replace(state)

    try:
        for mode in ('post', 'early'):
            stage = mode
            training = json.loads((outputs / f'{mode}_seed42.json').read_text(encoding='utf-8'))
            if len(training['history']) != 20 or training['test_evaluated']:
                raise ValueError('Formal training is incomplete or test was evaluated')
            output = outputs / f'{mode}_val.json'
            if any(output.with_suffix(suffix).exists() for suffix in ('.json', '.csv', '.png')):
                raise FileExistsError(f'Prior evaluation artifact exists: {output}')
            command = [sys.executable, '-m', 'evaluation.evaluate_classifier',
                       '--checkpoint', str(outputs / f'{mode}_seed42.pt'),
                       '--manifest', 'dataset/pilot_tier3_4events/manifest_classification.csv',
                       '--project-root', str(ROOT), '--output', str(output), '--device', 'cuda']
            record('running', command=command)
            subprocess.run(command, cwd=ROOT, check=True)
            evaluation = json.loads(output.read_text(encoding='utf-8'))
            if (evaluation['metrics'] != training['best_val'] or
                    evaluation['checkpoint_epoch'] != training['best_epoch'] or
                    evaluation['test_evaluated']):
                raise ValueError(f'{mode}: restored validation differs from training best')
        stage = 'comparison'
        output = outputs / 'frozen_val_comparison.json'
        if output.exists() or output.with_suffix('.csv').exists():
            raise FileExistsError(str(output))
        command = [sys.executable, '-m', 'evaluation.compare_pair',
                   '--b1', str(outputs / 'post_val.json'), '--b2', str(outputs / 'early_val.json'),
                   '--output', str(output)]
        record('running', command=command)
        subprocess.run(command, cwd=ROOT, check=True)
        record('complete', restored_metrics_match=True)
        return 0
    except Exception as exc:
        record('failed', error=repr(exc))
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
