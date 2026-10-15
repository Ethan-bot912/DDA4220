"""Verify metrics, pairing, initialization and CUDA training on SYNTHETIC data.

Run from the repository root: .venv/Scripts/python.exe tools/verify_phase3.py
Generated fixtures/checkpoints stay under ignored output/synthetic_phase3.
These tests produce no real xBD experimental results.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from stage_b_damage.dataset import LABEL_TO_ID, XBDPairedDataset
from stage_b_damage.train_baseline import audit_splits, build_model, metrics_from_confusion


def main() -> None:
    torch.set_num_threads(2)
    fixture = ROOT / 'output' / 'synthetic_phase3'
    fixture.mkdir(parents=True, exist_ok=True)
    confusion = torch.tensor([[3, 1, 0, 0], [0, 2, 2, 0], [0, 0, 4, 0], [1, 0, 0, 3]])
    metrics = metrics_from_confusion(confusion)
    assert abs(metrics['accuracy'] - 0.75) < 1e-12
    assert abs(metrics['balanced_accuracy'] - 0.75) < 1e-12
    assert abs(metrics['macro_f1'] - (0.75 + 4/7 + 0.8 + 6/7) / 4) < 1e-12
    absent = metrics_from_confusion(torch.diag(torch.tensor([3, 0, 0, 0])))
    assert absent['macro_f1'] == 0.25 and absent['balanced_accuracy'] == 1

    torch.manual_seed(42)
    post = build_model('resnet18', 'post').eval()
    torch.manual_seed(42)
    early = build_model('resnet18', 'early').eval()
    for name, value in post.state_dict().items():
        if name != 'conv1.weight':
            assert torch.equal(value, early.state_dict()[name]), name
    x = torch.rand(2, 3, 64, 64)
    with torch.inference_mode():
        torch.testing.assert_close(post(x), early(torch.cat([x, x], dim=1)), rtol=1e-4, atol=1e-5)
    del post, early

    rows = []
    rng = np.random.default_rng(42)
    for split, event, count in [('train', 'synthetic-train', 8), ('val', 'synthetic-val', 4)]:
        pre = rng.integers(0, 256, (3, 128, 128), dtype=np.int16)
        pre[:, :4, :4] = -99
        post = np.clip(pre.astype(np.int32) + 10, -99, 255).astype(np.int16)
        paths = {}
        for phase, array in [('pre', pre), ('post', post)]:
            path = fixture / f'{event}_{phase}.tif'
            with rasterio.open(path, 'w', driver='GTiff', height=128, width=128,
                               count=3, dtype='int16', nodata=-99, crs='EPSG:4326',
                               transform=from_origin(0, 128, 1, 1)) as dst:
                dst.write(array)
            paths[phase] = str(path)
        for i in range(count):
            rows.append(dict(split=split, event=event, tile_id='00000001', building_index=str(i),
                             damage=list(LABEL_TO_ID)[i % 4], pre_image=paths['pre'],
                             post_image=paths['post'], post_bbox_xmin=16 + (i % 4) * 20,
                             post_bbox_ymin=32, post_bbox_xmax=32 + (i % 4) * 20,
                             post_bbox_ymax=48))
    manifest = fixture / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dataset = XBDPairedDataset(manifest, 'train', crop_size=64, context=8)
    item = dataset[0]
    assert item['pre'].shape == item['post'].shape == (3, 64, 64)
    assert 0 <= float(item['pre'].min()) <= float(item['pre'].max()) <= 1
    assert int(item['label']) == 0
    torch.testing.assert_close(item['post'] - item['pre'],
                               torch.full_like(item['pre'], 10/255), atol=0.04, rtol=0)
    dataset.close()
    assert audit_splits(manifest)['val'] == ['synthetic-val']
    leakage = fixture / 'leakage.csv'
    bad_rows = [dict(rows[0]), dict(rows[-1])]
    bad_rows[-1]['event'] = bad_rows[0]['event']
    with leakage.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(bad_rows)
    try:
        audit_splits(leakage)
    except ValueError:
        pass
    else:
        raise AssertionError('Cross-event leakage was not rejected')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the GPU acceptance check')
    outputs = []
    evaluation_outputs = []
    evaluation_run = str(time.time_ns())
    for mode in ('post', 'early'):
        output = fixture / f'{mode}.json'
        subprocess.run([sys.executable, '-m', 'stage_b_damage.train_baseline',
                        '--manifest', str(manifest), '--output', str(output),
                        '--mode', mode, '--epochs', '2', '--batch-size', '4',
                        '--crop-size', '64', '--context', '8', '--device', 'cuda',
                        '--benchmark-steps', '5', '--benchmark-warmup', '2'],
                       cwd=ROOT, check=True)
        result = json.loads(output.read_text(encoding='utf-8'))
        checkpoint = torch.load(output.with_suffix('.pt'), map_location='cpu', weights_only=True)
        best = max(result['history'], key=lambda row: row['val']['macro_f1'])
        assert result['best_epoch'] == checkpoint['epoch'] == best['epoch']
        assert checkpoint['val'] == result['best_val'] == best['val']
        assert len(result['history']) == 2 and result['history'][1]['lr'] < result['history'][0]['lr']
        assert result['test_evaluated'] is False
        assert result['training_peak_allocated_bytes'] > 0
        evaluation_output = fixture / f'{mode}_val_{evaluation_run}.json'
        subprocess.run([sys.executable, '-m', 'evaluation.evaluate_classifier',
                        '--checkpoint', str(output.with_suffix('.pt')), '--manifest', str(manifest),
                        '--output', str(evaluation_output), '--device', 'cuda'], cwd=ROOT, check=True)
        frozen = json.loads(evaluation_output.read_text(encoding='utf-8'))
        assert frozen['metrics'] == result['best_val']
        assert frozen['per_event']['synthetic-val'] == frozen['metrics']
        assert evaluation_output.with_suffix('.png').is_file()
        assert evaluation_output.with_suffix('.csv').is_file()
        evaluation_outputs.append(evaluation_output)
        outputs.append(result)
    for key in ('manifest_sha256', 'train_indices_sha256', 'val_indices_sha256', 'class_counts', 'events'):
        assert outputs[0][key] == outputs[1][key], key
    subprocess.run([sys.executable, '-m', 'evaluation.compare_pair',
                    '--b1', str(fixture / 'post.json'), '--b2', str(fixture / 'early.json'),
                    '--output', str(fixture / 'validation_comparison.json')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'evaluation.compare_pair',
                    '--b1', str(evaluation_outputs[0]), '--b2', str(evaluation_outputs[1]),
                    '--output', str(fixture / 'frozen_comparison.json')], cwd=ROOT, check=True)
    protected = subprocess.run([sys.executable, '-m', 'evaluation.evaluate_classifier',
                                '--checkpoint', str(fixture / 'post.pt'), '--manifest', str(manifest),
                                '--output', str(fixture / 'test_not_authorized.json'), '--split', 'test'],
                               cwd=ROOT, capture_output=True)
    assert protected.returncode != 0
    assert not (fixture / 'test_not_authorized.json').exists()
    summary = {'synthetic_only': True, 'checks': 'metrics, shared initialization, crop pairing, leakage guard, CUDA B1/B2 training, checkpoint selection, scheduler, resource metrics, frozen validation evaluation, predictions, confusion figures, matched comparison, test-set protection',
               'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
               'params': {r['mode']: r['params'] for r in outputs}}
    (fixture / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
