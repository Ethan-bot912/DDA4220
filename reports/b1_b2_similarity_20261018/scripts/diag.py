import os, sys, json, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'.'); os.makedirs('output/diagnosis',exist_ok=True)
from stage_b_damage.dataset import XBDPairedDataset, LABEL_TO_ID
from stage_b_damage.train_baseline import build_model, metrics_from_confusion
torch.set_num_threads(2)
M='dataset/pilot_tier3_4events/manifest_classification.csv'
ds=XBDPairedDataset(M,'val',project_root='.',return_meta=True)
rng=np.random.default_rng(0)
labels=np.array([LABEL_TO_ID[r['damage']] for r in ds.rows])
idx=[]
for c,n in zip(range(4),[800,600,449,800]):
    ii=np.where(labels==c)[0]; idx+=list(rng.choice(ii,min(n,len(ii)),replace=False))
idx=np.array(sorted(idx)); print(len(idx),np.bincount(labels[idx]),flush=True)
pre=np.zeros((len(idx),3,128,128),np.float16); post=pre.copy()
for k,i in enumerate(idx):
    it=ds[int(i)]; pre[k]=it['pre'].numpy(); post[k]=it['post'].numpy()
    if k%500==0: print('load',k,flush=True)
y=labels[idx]
np.savez_compressed('output/diagnosis/subset_cache.npz',pre=pre,post=post,y=y,idx=idx)
def load(mode):
    ck=torch.load(f'output/resnet18/{mode}_seed42.pt',map_location='cpu',weights_only=True)
    m=build_model('resnet18','post' if mode=='post' else 'early'); m.load_state_dict(ck['model']); return m.eval()
def predict(m,x):
    out=[]
    with torch.no_grad():
        for s in range(0,len(x),128): out.append(m(torch.from_numpy(x[s:s+128].astype(np.float32))).softmax(1))
    return torch.cat(out).numpy()
def score(p):
    c=torch.zeros(4,4,dtype=torch.long)
    for a,b in zip(y,p.argmax(1)): c[a,b]+=1
    r=metrics_from_confusion(c); return dict(acc=round(r['accuracy'],4),macroF1=round(r['macro_f1'],4),balAcc=round(r['balanced_accuracy'],4))
b1=load('post'); b2=load('early')
res={}
P={}
P['B1 post only']=predict(b1,post)
P['B2 normal (pre,post)']=predict(b2,np.concatenate([pre,post],1))
P['B2 pre=zeros']=predict(b2,np.concatenate([np.zeros_like(pre),post],1))
P['B2 pre=post (no change)']=predict(b2,np.concatenate([post,post],1))
perm=rng.permutation(len(idx))
P['B2 pre=other building']=predict(b2,np.concatenate([pre[perm],post],1))
base=P['B2 normal (pre,post)'].argmax(1)
for k,p in P.items():
    res[k]=score(p); res[k]['pred_changed_vs_B2_normal_%']=round(float((p.argmax(1)!=base).mean()*100,),2)
res['B1 vs B2 pred differ %']=round(float((P['B1 post only'].argmax(1)!=base).mean()*100),2)
# input signal
diff=np.abs(post.astype(np.float32)-pre.astype(np.float32)).mean((1,2,3))
res['mean|post-pre| by true class']={n:round(float(diff[y==c].mean()),4) for n,c in LABEL_TO_ID.items()}
from sklearn.metrics import roc_auc_score
m=(y==0)|(y==3); res['AUC no_damage vs destroyed using only mean|post-pre|']=round(float(roc_auc_score(y[m]==3,diff[m])),4)
m=(y==0)|(y==1)|(y==2); 
json.dump(res,open('output/diagnosis/diag.json','w'),indent=1,ensure_ascii=False); print(json.dumps(res,indent=1,ensure_ascii=False))
