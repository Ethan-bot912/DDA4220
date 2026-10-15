"""Bounded disposable speed/memory experiment; never saves trained weights."""
from __future__ import annotations
import csv,ctypes,json,os,random,sys,time
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader
from stage_b_damage.dataset import XBDPairedDataset,LABEL_TO_ID
from stage_b_damage.train_baseline import build_model,make_input,synchronize
from tools.build_xview2_manifest import load_buildings,bbox_for,normalized_damage,MANIFEST_FIELDS

OUT=ROOT/'output/expansion_planning_20261015'
SAMPLE=ROOT/'dataset/capacity_sample_20261015'
def ram():
 class MEMORYSTATUSEX(ctypes.Structure):
  _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('totalPhys',ctypes.c_ulonglong),('availPhys',ctypes.c_ulonglong),('totalPage',ctypes.c_ulonglong),('availPage',ctypes.c_ulonglong),('totalVirtual',ctypes.c_ulonglong),('availVirtual',ctypes.c_ulonglong),('availExtended',ctypes.c_ulonglong)]
 s=MEMORYSTATUSEX();s.length=ctypes.sizeof(s);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s));return dict(total=s.totalPhys,available=s.availPhys,load_percent=s.load)

def main():
 if (OUT/'profile.json').exists():raise FileExistsError('Existing profile preserved')
 plan=json.loads((OUT/'split_proposal.json').read_text())
 allowed=set(plan['train_events'])
 rows=[];tile_events=Counter()
 for path in sorted(SAMPLE.glob('*/labels/*post_disaster.json')):
  event,tile=path.name.rsplit('_',3)[:2]
  if event not in allowed:continue
  pre=path.with_name(path.name.replace('_post_disaster','_pre_disaster'))
  postimage=path.parent.parent/'images'/path.name.replace('.json','.tif')
  preimage=postimage.with_name(postimage.name.replace('_post_disaster','_pre_disaster'))
  if not all(x.is_file() for x in [pre,postimage,preimage]):continue
  tile_events[event]+=1
  for i,feature in enumerate(load_buildings(path)):
   raw,damage=normalized_damage(feature);box=bbox_for(feature)
   if damage not in LABEL_TO_ID or box is None or box[0]>=box[2] or box[1]>=box[3]:continue
   r={k:'' for k in MANIFEST_FIELDS};r.update(split='train',event=event,tile_id=tile,building_index=str(i),damage_raw=raw,damage=damage,pre_image=str(preimage.relative_to(ROOT)),post_image=str(postimage.relative_to(ROOT)),pre_label=str(pre.relative_to(ROOT)),post_label=str(path.relative_to(ROOT)))
   r.update({f'post_bbox_{k}':v for k,v in zip(['xmin','ymin','xmax','ymax'],box)});rows.append(r)
 if len(rows)<1024:raise ValueError(f'Insufficient expanded train-event samples: {len(rows)}')
 # Bounded, mixed-event sample; use same fixed records for both modes.
 random.Random(42).shuffle(rows);rows=rows[:4096]
 manifest=OUT/'capacity_manifest.csv'
 with manifest.open('w',newline='',encoding='utf-8') as h:
  w=csv.DictWriter(h,fieldnames=MANIFEST_FIELDS);w.writeheader();w.writerows(rows)
 report=dict(scope='Disposable capacity benchmark only; no checkpoint, model selection, final training or held-out evaluation',batch_size=32,crop_size=128,seed=42,train_rows_available=len(rows),sample_events=dict(Counter(r['event'] for r in rows)),sample_tiles_by_event=dict(tile_events),gpu=torch.cuda.get_device_name(0),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,ram_before=ram(),results=[])
 def save(status,**extra):
  (OUT/'profile_state.json').write_text(json.dumps(dict(status=status,pid=os.getpid(),**extra),indent=2))
 save('running')
 for mode in ['post','early']:
  random.seed(42);np.random.seed(42);torch.manual_seed(42)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
  ds=XBDPairedDataset(manifest,'train',project_root=ROOT,crop_size=128,context=16)
  loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)
  model=build_model('resnet18',mode).cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=.001)
  counts=Counter(r['damage'] for r in rows);weights=torch.tensor([1/max(counts[k],1) for k in LABEL_TO_ID],device='cuda');weights/=weights.mean()
  lossfn=torch.nn.CrossEntropyLoss(weight=weights)
  io_times=[];compute_times=[];batch_times=[];ram_min=ram()['available']
  it=iter(loader);torch.cuda.reset_peak_memory_stats()
  for step in range(36):
   t=time.perf_counter();batch=next(it);t1=time.perf_counter()
   opt.zero_grad(set_to_none=True);x=make_input(batch,mode).cuda();y=batch['label'].cuda();loss=lossfn(model(x),y);loss.backward();opt.step();synchronize(torch.device('cuda'));t2=time.perf_counter()
   ram_min=min(ram_min,ram()['available'])
   if step>=4:io_times.append(t1-t);compute_times.append(t2-t1);batch_times.append(t2-t)
  train_peak=torch.cuda.max_memory_allocated();train_reserved=torch.cuda.max_memory_reserved()
  opt.zero_grad(set_to_none=True)
  del x,y,loss,batch,opt
  model.eval();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  vit=iter(DataLoader(ds,batch_size=32,shuffle=False,num_workers=0));eval_io=[];eval_compute=[];eval_times=[]
  with torch.inference_mode():
   for step in range(36):
    t=time.perf_counter();batch=next(vit);t1=time.perf_counter();x=make_input(batch,mode).cuda();logits=model(x);pred=logits.argmax(1).cpu();synchronize(torch.device('cuda'));t2=time.perf_counter()
    if step>=4:eval_io.append(t1-t);eval_compute.append(t2-t1);eval_times.append(t2-t)
  # No accuracy or F1 is computed; labels only exercise normal training loss.
  def stats(v):return dict(mean_seconds=float(np.mean(v)),median_seconds=float(np.median(v)),p10_seconds=float(np.quantile(v,.1)),p90_seconds=float(np.quantile(v,.9)),batches=len(v),samples=len(v)*32)
  record=dict(mode=mode,train_batch=stats(batch_times),train_io=stats(io_times),train_compute=stats(compute_times),eval_batch=stats(eval_times),eval_io=stats(eval_io),eval_compute=stats(eval_compute),train_peak_allocated_bytes=train_peak,train_peak_reserved_bytes=train_reserved,eval_peak_allocated_bytes=torch.cuda.max_memory_allocated(),system_available_ram_min_bytes=ram_min)
  report['results'].append(record);print(json.dumps(record),flush=True)
  save('running',completed_modes=[r['mode'] for r in report['results']])
  ds.close();del model,ds,loader,it,vit,x,logits,pred,batch;torch.cuda.empty_cache()
 report['ram_after']=ram();report['test_evaluated']=False
 (OUT/'profile.json').write_text(json.dumps(report,indent=2),encoding='utf-8');save('complete')
 print('Profile complete; no model weights saved',flush=True)

if __name__=='__main__':main()
