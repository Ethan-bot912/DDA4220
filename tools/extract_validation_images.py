"""Verify or extract only the 440 published val images; no training/test evaluation."""
from __future__ import annotations
import argparse,csv,hashlib,io,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.extract_xview2_subset import archive_parts,verify_parts,ConcatenatedParts,OFFICIAL_SHA1
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--parts-dir',type=Path)
 p.add_argument('--extract',action='store_true',help='Extract only missing listed images after official archive verification')
 a=p.parse_args()
 with (ROOT/'reports/four_event_seed42/validation_images.csv').open(encoding='utf-8',newline='') as h:rows=list(csv.DictReader(h))
 pending={}
 for r in rows:
  dest=ROOT/r['relative_path']
  if dest.exists():
   if dest.stat().st_size!=int(r['bytes']) or digest(dest)!=r['sha256']:raise ValueError(f'Existing image differs; preserved without overwrite: {dest}')
  else:pending[r['archive_member']]=r
 if not pending:print('All 440 image SHA256 checks match');return
 if not a.extract:raise FileNotFoundError(f'{len(pending)} images missing; provide --extract --parts-dir PATH to six official parts')
 if a.parts_dir is None:p.error('--extract requires --parts-dir')
 parts=archive_parts(a.parts_dir);verify_parts(parts,OFFICIAL_SHA1)
 with io.BufferedReader(ConcatenatedParts(parts)) as source,tarfile.open(fileobj=source,mode='r|gz') as stream:
  for member in stream:
   if member.name not in pending:continue
   r=pending[member.name];dest=ROOT/r['relative_path'];dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():raise FileExistsError(dest)
   temp=dest.with_suffix('.tif.partial');sha=hashlib.sha256()
   with stream.extractfile(member) as src,temp.open('xb') as target:
    for b in iter(lambda:src.read(1024*1024),b''):target.write(b);sha.update(b)
   if temp.stat().st_size!=int(r['bytes']) or sha.hexdigest()!=r['sha256']:raise ValueError(f'Extracted image differs; partial file retained: {temp}')
   temp.replace(dest);del pending[member.name]
   if not pending:break
 if pending:raise ValueError(f'Archive did not contain {len(pending)} listed images')
 print('All 440 validation images present and verified; no train/test images extracted')
if __name__=='__main__':main()
