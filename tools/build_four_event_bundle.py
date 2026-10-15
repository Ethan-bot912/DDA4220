"""Package existing real experiment artifacts, with no images or retraining."""
from pathlib import Path
import csv,hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/four_event_seed42'
TAG='four-event-seed42-20261015'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 REPORT.mkdir(exist_ok=True)
 manifest=ROOT/'dataset/pilot_tier3_4events/manifest_classification.csv'
 with manifest.open(encoding='utf-8',newline='') as h:rows=list(csv.DictReader(h))
 val=[r for r in rows if r['split']=='val'];paths=sorted({r[k].replace('\\','/') for r in val for k in ['pre_image','post_image']})
 assert len(val)==22372 and len(paths)==440
 with (REPORT/'validation_images.csv').open('w',newline='',encoding='utf-8') as h:
  w=csv.DictWriter(h,fieldnames=['relative_path','archive_member','bytes','sha256']);w.writeheader()
  for p in paths:
   f=ROOT/p;w.writerow(dict(relative_path=p,archive_member='geotiffs/tier3/images/'+f.name,bytes=f.stat().st_size,sha256=sha(f)))
 files=[manifest]+[p for p in sorted((ROOT/'output/resnet18').iterdir()) if p.suffix in ['.pt','.json','.csv']]
 records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
 summary={}
 for mode in ['post','early']:
  d=json.loads((ROOT/f'output/resnet18/{mode}_seed42.json').read_text())
  summary[mode]={k:d[k] for k in ['best_epoch','best_val','params','training_seconds','training_peak_allocated_bytes','inference_benchmark','hardware','versions','manifest_sha256']}
  for suffix in ['_seed42.json','_val.json']:
   shutil.copy2(ROOT/f'output/resnet18/{mode}{suffix}',REPORT/f'{mode}{suffix}')
 for name in ['frozen_val_comparison.csv','frozen_val_comparison.json']:
  shutil.copy2(ROOT/'output/resnet18'/name,REPORT/name)
 (REPORT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 metadata=dict(experiment='real four-event ResNet18 B1/B2 seed42',release_tag=TAG,manifest_sha256=sha(manifest),files=records,contains_images=False,test_evaluated=False,checkpoint_role='Best validation weights only, not resumable optimizer checkpoints',required_validation_images=440,required_validation_buildings=22372)
 stage=ROOT/'reproduction/four_event_seed42';stage.mkdir(parents=True,exist_ok=True)
 (stage/'BUNDLE.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
 sums='\n'.join(f'{r["sha256"]}  {r["path"]}' for r in records)+'\n'
 (stage/'SHA256SUMS.txt').write_text(sums,encoding='utf-8')
 (REPORT/'artifact_checksums.txt').write_text(sums,encoding='utf-8')
 dest=ROOT/'output/publication_20261015';dest.mkdir(exist_ok=True)
 package=dest/'four-event-seed42-reproduction.zip'
 with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files+[stage/'BUNDLE.json',stage/'SHA256SUMS.txt']:z.write(p,p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(package) as z:
  assert not any(n.lower().endswith(('.tif','.tiff','.jpg','.jpeg')) for n in z.namelist())
  for r in records:
   with z.open(r['path']) as f:
    h=hashlib.sha256()
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==r['sha256'],r['path']
 (dest/(package.name+'.sha256')).write_text(f'{sha(package)}  {package.name}\n',encoding='utf-8')
 print(json.dumps(dict(package=str(package),bytes=package.stat().st_size,sha256=sha(package),members=len(records)+2,validation_images=len(paths)),indent=2))
if __name__=='__main__':main()
