"""Small synthetic tests for archive preparation and comparison safeguards."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from evaluation.compare_pair import compare
from tools.extract_xview2_subset import ConcatenatedParts, archive_parts, verify_parts

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ['joplin-tornado', 'lower-puna-volcano', 'moore-tornado', 'nepal-flooding']


class PreparationTest(unittest.TestCase):
    def setUp(self):
        (ROOT / 'output').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'output')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def command(self, script, *args, success=True):
        result = subprocess.run([sys.executable, str(ROOT / 'tools' / script), *map(str, args)],
                                cwd=ROOT, text=True, capture_output=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_windows_manifest_paths_work_on_posix(self):
        # Exercise a POSIX filesystem interpretation even when tests run on Windows.
        from pathlib import PurePosixPath
        from unittest.mock import patch
        from stage_b_damage.dataset import XBDPairedDataset
        dataset = object.__new__(XBDPairedDataset)
        dataset.project_root = PurePosixPath('/team/repository')
        with patch('stage_b_damage.dataset.Path', PurePosixPath):
            self.assertEqual(dataset._path(r'dataset\pilot\images\tile.tif'),
                             PurePosixPath('/team/repository/dataset/pilot/images/tile.tif'))
            self.assertEqual(dataset._path('/shared/tile.tif'), PurePosixPath('/shared/tile.tif'))

    def create_parts(self):
        with MemoryFile() as image:
            with image.open(driver='GTiff', height=64, width=64, count=3, dtype='int16',
                            crs='EPSG:4326', transform=from_origin(0, 64, 1, 1), nodata=-99) as dst:
                dst.write(np.full((3, 64, 64), 100, dtype=np.int16))
            image_bytes = image.read()
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode='w:gz') as stream:
            for event in EVENTS:
                for phase in ['pre', 'post']:
                    features = [{'wkt': 'POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))',
                                 'properties': {'feature_type': 'building', 'subtype': label}}
                                for label in ['no-damage', 'minor-damage', 'major-damage',
                                              'destroyed', 'un-classified']]
                    for kind, ext, blob in [('images', 'tif', image_bytes),
                                            ('labels', 'json', json.dumps({'features': {'xy': features}}).encode())]:
                        name = f'geotiffs/tier3/{kind}/{event}_00000001_{phase}_disaster.{ext}'
                        member = tarfile.TarInfo(name)
                        member.size = len(blob)
                        stream.addfile(member, io.BytesIO(blob))
            blob = b'not selected'
            member = tarfile.TarInfo('geotiffs/train/images/other-event_00000001_post_disaster.tif')
            member.size = len(blob)
            stream.addfile(member, io.BytesIO(blob))
        content = archive.getvalue()
        parts_dir = self.directory / 'parts'
        parts_dir.mkdir()
        for i, letter in enumerate('abcdef'):
            (parts_dir / f'xview2_geotiff.tgz.part-a{letter}').write_bytes(
                content[len(content)*i//6:len(content)*(i+1)//6])
        return parts_dir, content

    def test_streamed_data_pipeline(self):
        parts_dir, content = self.create_parts()
        parts = archive_parts(parts_dir)
        with io.BufferedReader(ConcatenatedParts(parts), buffer_size=17) as reader:
            self.assertEqual(reader.read(), content)
        expected = hashlib.sha1(content).hexdigest()
        self.assertEqual(verify_parts(parts, expected), expected)
        output = self.directory / 'dataset' / 'pilot'
        self.command('extract_xview2_subset.py', '--parts-dir', parts_dir, '--expected-sha1', expected,
                     '--output', output, '--split', 'tier3',
                     *[v for event in EVENTS for v in ['--event', event]])
        summary = json.loads((output / 'extraction_summary.json').read_text())
        self.assertEqual(summary['files_written'], 16)
        self.assertEqual(summary['verified_sha1'], expected)
        self.assertEqual(len(list(output.rglob('*.tif'))), 8)
        manifest = output / 'manifest_classification.csv'
        self.command('build_xview2_manifest.py', '--data-root', output, '--output', manifest,
                     '--train-event', EVENTS[0], '--train-event', EVENTS[1],
                     '--val-event', EVENTS[2], '--test-event', EVENTS[3], '--drop-unknown')
        audit = json.loads((output / 'audit.json').read_text())
        self.assertEqual(audit['building_rows'], 16)
        self.assertEqual(audit['dropped_unknown'], 4)
        self.assertEqual(audit['missing_components'], [])
        self.command('audit_xview2_manifest.py', '--manifest', manifest, '--project-root', self.directory,
                     '--output', output / 'geotiff_audit.json')
        image_audit = json.loads((output / 'geotiff_audit.json').read_text())
        self.assertEqual(image_audit['errors'], [])
        self.assertEqual(image_audit['readable_images'], 8)
        self.command('render_xview2_sample.py', '--data-root', output,
                     '--output', output / 'synthetic_overlay.png')
        self.assertTrue((output / 'synthetic_overlay.png').is_file())
        with manifest.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
            self.assertEqual({row['split'] for row in rows}, {'train', 'val', 'test'})
        rows[0]['post_bbox_xmin'] = ''
        with manifest.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.command('audit_xview2_manifest.py', '--manifest', manifest, '--project-root', self.directory,
                     '--output', output / 'invalid_audit.json', success=False)
        self.assertEqual(json.loads((output / 'invalid_audit.json').read_text())['errors'][0]['type'], 'invalid_bbox')

    def test_bad_checksum_and_missing_parts_fail_before_extraction(self):
        parts_dir, content = self.create_parts()
        output = self.directory / 'must_not_exist'
        self.command('extract_xview2_subset.py', '--parts-dir', parts_dir, '--expected-sha1', '0'*40,
                     '--output', output, '--split', 'tier3', '--event', EVENTS[0], success=False)
        self.assertFalse(output.exists())
        (parts_dir / 'xview2_geotiff.tgz.part-aa').unlink()
        with self.assertRaises(FileNotFoundError):
            archive_parts(parts_dir)

    def test_multiple_partitions_and_duplicate_tiles(self):
        parts_dir, content = self.create_parts()
        output = self.directory / 'dataset' / 'multi'
        self.command('extract_xview2_subset.py', '--parts-dir', parts_dir,
                     '--expected-sha1', hashlib.sha1(content).hexdigest(), '--output', output,
                     '--split', 'tier3', *[v for event in EVENTS for v in ['--event', event]])
        for kind in ('images', 'labels'):
            target = output / 'tier1' / kind
            target.mkdir(parents=True)
            for source in (output / 'tier3' / kind).glob(EVENTS[0] + '*'):
                source.rename(target / source.name)
        arguments = ['--data-root', output, '--output', output / 'manifest.csv',
                     '--archive-split', 'tier3', '--archive-split', 'tier1',
                     '--train-event', EVENTS[0], '--train-event', EVENTS[1],
                     '--val-event', EVENTS[2], '--test-event', EVENTS[3], '--drop-unknown']
        self.command('build_xview2_manifest.py', *arguments)
        audit = json.loads((output / 'audit.json').read_text())
        self.assertEqual(audit['building_rows'], 16)
        source = next((output / 'tier1' / 'images').glob('*.tif'))
        (output / 'tier3' / 'images' / source.name).write_bytes(source.read_bytes())
        result = self.command('build_xview2_manifest.py', *arguments, success=False)
        self.assertIn('Duplicate event/tile/phase', result.stderr)

    def test_comparison_rejects_uncontrolled_or_mixed_results(self):
        score = dict(accuracy=0.5, macro_f1=0.4, balanced_accuracy=0.3)
        b1 = dict(mode='post', backbone='resnet18', manifest_sha256='a', train_indices_sha256='b',
                  val_indices_sha256='c', events={}, config=dict(seed=42, mode='post', output='b1'),
                  split='val', evaluated_indices_sha256='d', metrics=score)
        b2 = copy.deepcopy(b1)
        b2['mode'] = b2['config']['mode'] = 'early'
        b2['config']['output'] = 'b2'
        b2['metrics']['macro_f1'] = 0.6
        self.assertAlmostEqual(compare(b1, b2)['B2_minus_B1']['macro_f1'], 0.2)
        for key, value in [('manifest_sha256', 'other'), ('split', 'test')]:
            bad = copy.deepcopy(b2)
            bad[key] = value
            with self.assertRaises(ValueError):
                compare(b1, bad)
        bad = copy.deepcopy(b2)
        bad['config']['seed'] = 43
        with self.assertRaises(ValueError):
            compare(b1, bad)

    def test_multi_seed_summary_and_duplicate_rejection(self):
        inputs = {'post': [], 'early': []}
        for seed in (42, 43):
            for mode, f1 in [('post', 0.4), ('early', 0.6)]:
                path = self.directory / f'{mode}{seed}.json'
                payload = dict(mode=mode, backbone='resnet18', manifest_sha256='a',
                               train_indices_sha256='b', val_indices_sha256='c', events={},
                               config=dict(seed=seed, mode=mode, output=str(path)),
                               split='val', evaluated_indices_sha256='d',
                               metrics=dict(accuracy=0.5, macro_f1=f1, balanced_accuracy=0.3))
                path.write_text(json.dumps(payload))
                inputs[mode].append(str(path))
        output = self.directory / 'summary.json'
        command = [sys.executable, '-m', 'evaluation.compare_pair', '--b1', *inputs['post'],
                   '--b2', *inputs['early'], '--output', str(output)]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        aggregate = json.loads(output.read_text())['aggregate']['B2_minus_B1']['macro_f1']
        self.assertAlmostEqual(aggregate['mean'], 0.2)
        self.assertEqual(aggregate['sample_std'], 0)
        self.assertEqual(aggregate['n'], 2)
        command = [sys.executable, '-m', 'evaluation.compare_pair', '--b1', inputs['post'][0], inputs['post'][0],
                   '--b2', inputs['early'][0], inputs['early'][0], '--output', str(output)]
        self.assertNotEqual(subprocess.run(command, cwd=ROOT, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
