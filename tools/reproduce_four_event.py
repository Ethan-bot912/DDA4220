"""Verify published four-event assets and optionally evaluate best weights, never train."""
from __future__ import annotations
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--evaluate',action='store_true',help='Run val inference with both best checkpoints; never trains')
 p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
 p.add_argument('--output-root',type=Path,default=ROOT/'output/reproduced_four_event_seed42')
 a=p.parse_args();metadata=ROOT/'reproduction/four_event_seed42/BUNDLE.json'
 if not metadata.exists():p.error('Download the reproduction ZIP linked in REPRODUCE_AND_DIAGNOSE.md and extract into the repository root first')
 bundle=json.loads(metadata.read_text(encoding='utf-8'))
 for entry in bundle['files']:
  f=ROOT/entry['path']
  if not f.is_file():raise FileNotFoundError(f)
  if hashlib.sha256(f.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError(f'File checksum differs: {f}')
 missing=[]
 with (ROOT/'reports/four_event_seed42/validation_images.csv').open(encoding='utf-8',newline='') as h:
  for row in csv.DictReader(h):
   f=ROOT/row['relative_path']
   if not f.is_file() or f.stat().st_size!=int(row['bytes']):missing.append(row['relative_path'])
 if missing:raise FileNotFoundError(f'{len(missing)} validation images missing or wrong size; use tools/extract_validation_images.py. First: {missing[:3]}')
 print('Bundle checksums and all 440 validation image sizes verified. No training is performed.')
 if not a.evaluate:return
 if a.output_root.exists() and any(a.output_root.iterdir()):raise FileExistsError('Use a new output directory; existing results are preserved')
 a.output_root.mkdir(parents=True,exist_ok=True)
 for mode in ['post','early']:
  subprocess.run([sys.executable,'-m','evaluation.evaluate_classifier','--checkpoint',f'output/resnet18/{mode}_seed42.pt','--manifest','dataset/pilot_tier3_4events/manifest_classification.csv','--project-root',str(ROOT),'--output',str(a.output_root/f'{mode}_val.json'),'--device',a.device],cwd=ROOT,check=True)
 subprocess.run([sys.executable,'-m','evaluation.compare_pair','--b1',str(a.output_root/'post_val.json'),'--b2',str(a.output_root/'early_val.json'),'--output',str(a.output_root/'comparison.json')],cwd=ROOT,check=True)
 comparisons={}
 for mode in ['post','early']:
  expected=json.loads((ROOT/f'output/resnet18/{mode}_val.json').read_text())['metrics']
  actual=json.loads((a.output_root/f'{mode}_val.json').read_text())['metrics']
  comparisons[mode]={'same_confusion':actual['confusion']==expected['confusion'],'deltas':{k:actual[k]-expected[k] for k in ['accuracy','macro_f1','balanced_accuracy']}}
 (a.output_root/'reference_check.json').write_text(json.dumps(comparisons,indent=2))
 print(json.dumps(comparisons,indent=2))
 print('Different hardware may cause numerical differences; investigate any discrepancy before changing code or retraining. Test remains sealed.')
if __name__=='__main__':main()
