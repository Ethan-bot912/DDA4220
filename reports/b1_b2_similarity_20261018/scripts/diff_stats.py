import os,sys,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
sys.path.insert(0,'.'); os.makedirs('output/diagnosis',exist_ok=True)
import torch;torch.set_num_threads(1)
from stage_b_damage.dataset import XBDPairedDataset
M='dataset/pilot_tier3_4events/manifest_classification.csv'
def shift(a,b):
    a=a-a.mean();b=b-b.mean()
    F=np.fft.fft2(a)*np.conj(np.fft.fft2(b));F/=np.abs(F)+1e-8
    r=np.fft.ifft2(F).real;i=np.unravel_index(r.argmax(),r.shape)
    dy,dx=[(v if v<64 else v-128) for v in i];return np.hypot(dy,dx)
out=[]
for split,n in [('val',None),('train',5000)]:
    ds=XBDPairedDataset(M,split,project_root='.',return_meta=True)
    idx=range(len(ds)) if n is None else np.random.default_rng(0).choice(len(ds),n,replace=False)
    for k,i in enumerate(idx):
        it=ds[int(i)];p=it['pre'].numpy();q=it['post'].numpy();mt=it['meta']
        out.append(dict(split=split,event=mt['event'],tile_id=mt['tile_id'],building_index=int(mt['building_index']),damage=mt['damage'],
            absdiff=float(np.abs(q-p).mean()),bright=float(q.mean()-p.mean()),shift=float(shift(p.mean(0),q.mean(0)))))
        if k%3000==0:print(split,k,flush=True)
pd.DataFrame(out).to_csv('output/diagnosis/diff_stats.csv',index=False);print('done')
