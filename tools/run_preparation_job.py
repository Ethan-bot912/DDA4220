"""Persist progress for the unattended, validated four-event preparation job."""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'output' / 'automation_xbd' / 'preparation_state.json'


def record(stage, status, **details):
    payload = dict(pid=os.getpid(), stage=stage, status=status,
                   updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix('.partial')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(STATE)


def main():
    os.chdir(ROOT)
    for key in ('TEMP', 'TMP'):
        os.environ[key] = 'D:/DDA4220/tmp'
    os.environ['MPLCONFIGDIR'] = 'D:/DDA4220/tmp/matplotlib'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    stage = 'environment'
    try:
        for stage, command in [
            ('environment', [sys.executable, 'tools/check_environment.py']),
            ('checksum_extract_manifest_audit', [sys.executable, 'tools/prepare_pilot.py']),
            ('sample_overlay', [sys.executable, 'tools/render_xview2_sample.py']),
        ]:
            record(stage, 'running', command=command)
            print(f'=== {stage} ===', flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        record('ready_for_visual_audit', 'complete',
               note='View the overlay and inspect audits before starting real_pilot training.')
        return 0
    except Exception as exc:
        record(stage, 'failed', error=repr(exc))
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
