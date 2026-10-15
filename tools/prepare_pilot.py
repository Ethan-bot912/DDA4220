"""Prepare the documented four-event pilot from six complete download parts.

No full extraction and no merged archive. Checksum verification precedes all
extraction. Run --check-only now; run without it after downloads complete.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.extract_xview2_subset import archive_parts

EVENTS = {'train': ['joplin-tornado', 'lower-puna-volcano'],
          'val': ['moore-tornado'], 'test': ['nepal-flooding']}


def run(script: str, *arguments: str) -> None:
    subprocess.run([sys.executable, str(ROOT / 'tools' / script), *arguments], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-dir', type=Path, default=Path('D:/DDA4220 Dataset'))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    expected = [args.parts_dir / f'xview2_geotiff.tgz.part-a{s}' for s in 'abcdef']
    status = [{'name': p.name, 'complete_name_present': p.is_file(),
               'bytes': p.stat().st_size if p.is_file() else 0} for p in expected]
    print(json.dumps({'parts': status, 'events': EVENTS,
                      'check_only': args.check_only}, indent=2), flush=True)
    if args.check_only:
        return 0 if all(p.is_file() and p.stat().st_size > 0 for p in expected) else 2
    archive_parts(args.parts_dir)
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError('Need at least 20 GiB free on the project drive for the pilot and outputs')
    output = ROOT / 'dataset' / 'pilot_tier3_4events'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; inspect prior extraction before rerunning')
    event_args = [value for names in EVENTS.values() for name in names for value in ['--event', name]]
    run('extract_xview2_subset.py', '--parts-dir', str(args.parts_dir), '--output', str(output),
        '--split', 'tier3', *event_args)
    extraction = json.loads((output / 'extraction_summary.json').read_text())
    if any(count == 0 for count in extraction['counts_by_split_event'].values()):
        raise RuntimeError('An expected event has no extracted files')
    manifest = output / 'manifest_classification.csv'
    split_args = [value for split, names in EVENTS.items() for name in names
                  for value in [f'--{split}-event', name]]
    run('build_xview2_manifest.py', '--data-root', str(output), '--output', str(manifest),
        '--drop-unknown', *split_args)
    audit = json.loads((output / 'audit.json').read_text())
    if audit['missing_components'] or not audit['building_rows']:
        raise RuntimeError('Manifest is empty or contains incomplete tile pairs; inspect audit.json')
    run('audit_xview2_manifest.py', '--manifest', str(manifest),
        '--output', str(output / 'geotiff_audit.json'), '--project-root', str(ROOT))
    print('Pilot ready. Inspect audit/sample overlays, then run tools/run_controlled_pair.ps1.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
