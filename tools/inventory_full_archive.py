"""Read-only archive census plus bounded sample extraction for resource planning."""
from __future__ import annotations
import csv,hashlib,io,json,math,os,re,shutil,sys,tarfile,time,traceback
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.extract_xview2_subset import ConcatenatedParts,archive_parts,OFFICIAL_SHA1
from tools.build_xview2_manifest import choose_features,is_building,normalized_damage,bbox_for
OUT=ROOT/'output'/'expansion_planning_20261015'
SAMPLE=ROOT/'dataset'/'capacity_sample_20261015'
PAT=re.compile(r'^geotiffs/([^/]+)/(images|labels)/(.+)_([0-9]{8})_(pre|post)_disaster\.(tif|json)$')

class HashReader(io.RawIOBase):
    def __init__(self,raw): self.raw=raw;self.digest=hashlib.sha1();self.n=0
    def readable(self):return True
    def readinto(self,b):
        n=self.raw.readinto(b)
        if n:self.digest.update(memoryview(b)[:n]);self.n+=n
        return n
    def close(self):self.raw.close();super().close()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'inventory.json').exists():raise FileExistsError('Inventory already completed')
    if SAMPLE.exists() and any(SAMPLE.iterdir()):raise FileExistsError('Existing sample must not be overwritten')
    parts=archive_parts(Path('D:/DDA4220 Dataset'))
    total=sum(p.stat().st_size for p in parts);reader=HashReader(ConcatenatedParts(parts))
    started=time.perf_counter();last=0;members=0;bytes_total=0;sample_bytes=0
    records={};chosen=defaultdict(set);events=defaultdict(lambda:{'files':0,'bytes':0,'partitions':Counter(),'metadata_types':Counter()})
    unrecognized=[];duplicates=[];errors=[];partitions=Counter()
    def state(status,**kw):
        data=dict(pid=os.getpid(),status=status,elapsed_seconds=time.perf_counter()-started,compressed_bytes_read=reader.n,compressed_total_bytes=total,members_seen=members,uncompressed_regular_bytes=bytes_total,sample_bytes=sample_bytes,events_seen=len(events),**kw)
        tmp=OUT/'inventory_state.partial';tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(OUT/'inventory_state.json')
    state('running')
    try:
      with io.BufferedReader(reader,8*1024*1024) as src:
       with tarfile.open(fileobj=src,mode='r|gz',bufsize=1024*1024) as archive:
        for member in archive:
            members+=1
            if not member.isfile():continue
            bytes_total+=member.size
            m=PAT.match(member.name)
            if not m:
                unrecognized.append({'name':member.name,'size':member.size});continue
            partition,kind,event,tile,phase,ext=m.groups();partitions[partition]+=1
            ev=events[event];ev['files']+=1;ev['bytes']+=member.size;ev['partitions'][partition]+=1
            key=(partition,event,tile)
            rec=records.setdefault(key,dict(partition=partition,event=event,tile_id=tile,components={},class_counts=Counter(),unknown=0,invalid_bbox=0,post_buildings=0))
            component=kind+'_'+phase
            if component in rec['components']:duplicates.append(member.name)
            rec['components'][component]=member.size
            samplekey=(partition,tile)
            if kind=='images' and samplekey not in chosen[event] and len(chosen[event])<2:chosen[event].add(samplekey)
            keep=samplekey in chosen[event]
            # Hard bounded extraction protects the currently limited free space.
            if keep and sample_bytes+member.size>2*1024**3:raise RuntimeError('Sample extraction exceeded 2 GiB limit')
            payload=None
            if kind=='labels':
                payload=archive.extractfile(member).read()
                rec[phase+'_label_sha256']=hashlib.sha256(payload).hexdigest()
                try:
                    label=json.loads(payload)
                    features=[f for f in choose_features(label) if is_building(f)]
                    rec[phase+'_building_count']=len(features)
                    metadata=label.get('metadata',{})
                    if phase=='post':
                        rec['disaster_type']=metadata.get('disaster_type','unknown')
                        ev['metadata_types'][str(rec['disaster_type'])]+=1
                        for feature in features:
                            rec['post_buildings']+=1;raw,damage=normalized_damage(feature)
                            if damage=='unknown':rec['unknown']+=1;continue
                            box=bbox_for(feature)
                            if box is None or not all(math.isfinite(v) for v in box) or box[0]>=box[2] or box[1]>=box[3]:rec['invalid_bbox']+=1;continue
                            rec['class_counts'][damage]+=1
                except Exception as e:errors.append({'member':member.name,'error':repr(e)})
            if keep:
                destination=SAMPLE/partition/kind/f'{event}_{tile}_{phase}_disaster.{ext}'
                destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists():raise FileExistsError(destination)
                with destination.open('wb') as target:
                    if payload is not None:target.write(payload)
                    else:shutil.copyfileobj(archive.extractfile(member),target,8*1024*1024)
                sample_bytes+=member.size
            now=time.perf_counter()
            if now-last>20:
                state('running',last_member=member.name)
                print(f'elapsed_min={(now-started)/60:.1f} compressed_GiB={reader.n/2**30:.2f}/{total/2**30:.2f} events={len(events)} members={members}',flush=True);last=now
       # Read all remaining compressed bytes so the full official hash is checked.
       while src.read(8*1024*1024):pass
      actual=reader.digest.hexdigest()
      if actual!=OFFICIAL_SHA1:raise ValueError(f'Official SHA1 mismatch: {actual}')
      required={'images_pre','images_post','labels_pre','labels_post'}
      missing=[];by_event={};by_partition={}
      for key,rec in records.items():
        rec['complete_pair']=required.issubset(rec['components'])
        rec['valid_rows']=sum(rec['class_counts'].values()) if rec['complete_pair'] else 0
        if not rec['complete_pair']:missing.append({'key':list(key),'missing':sorted(required-set(rec['components']))})
        for group,groupkey in [(by_event,rec['event']),(by_partition,rec['partition'])]:
            g=group.setdefault(groupkey,dict(tiles=0,complete_pairs=0,valid_buildings=0,unknown=0,invalid_bbox=0,class_counts=Counter(),bytes=0))
            g['tiles']+=1;g['complete_pairs']+=int(rec['complete_pair']);g['valid_buildings']+=rec['valid_rows'];g['unknown']+=rec['unknown'];g['invalid_bbox']+=rec['invalid_bbox'];g['bytes']+=sum(rec['components'].values())
            if rec['complete_pair']:g['class_counts'].update(rec['class_counts'])
      for ev,meta in events.items():by_event[ev].update(metadata_types=dict(meta['metadata_types']),partitions=dict(meta['partitions']))
      seen={};tile_overlap=[]
      for rec in records.values():
        k=(rec['event'],rec['tile_id'])
        if k in seen:tile_overlap.append({'event':k[0],'tile_id':k[1],'partitions':[seen[k],rec['partition']]})
        seen[k]=rec['partition']
      report=dict(status='complete',official_sha1=actual,compressed_bytes=total,uncompressed_regular_bytes=bytes_total,members_seen=members,elapsed_seconds=time.perf_counter()-started,sample_bytes=sample_bytes,by_event=by_event,by_archive_partition=by_partition,missing_components=missing,duplicate_member_names=duplicates,cross_archive_partition_tile_overlap=tile_overlap,label_parse_errors=errors,unrecognized_regular_files=unrecognized,sampled_tiles={e:[list(x) for x in sorted(keys)] for e,keys in chosen.items()},scope='Archive and label census; full image decode audit not performed; no training or test evaluation')
      (OUT/'inventory.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
      (OUT/'tile_inventory.json').write_text(json.dumps(list(records.values()),indent=2),encoding='utf-8')
      fields=['event','tiles','complete_pairs','valid_buildings','no_damage','minor','major','destroyed','unknown','invalid_bbox','bytes','partitions','metadata_types']
      with (OUT/'event_inventory.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for event,d in sorted(by_event.items()):
            row={k:d[k] for k in ['tiles','complete_pairs','valid_buildings','unknown','invalid_bbox','bytes']};row.update(event=event,partitions=json.dumps(d['partitions']),metadata_types=json.dumps(d['metadata_types']))
            row.update({k:d['class_counts'].get(k,0) for k in ['no_damage','minor','major','destroyed']});writer.writerow(row)
      state('complete',verified_sha1=actual)
      print(json.dumps({k:v for k,v in report.items() if k not in ['unrecognized_regular_files','sampled_tiles']},indent=2),flush=True)
    except BaseException:
      state('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
