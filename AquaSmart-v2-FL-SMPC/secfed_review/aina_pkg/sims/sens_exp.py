import sys,json;sys.path.insert(0,'.')
from rcore import *
ponds=load_ponds()
CFG={'base':{},'uniform':{'wmode':'u'},'T10':{'T':10},'T40':{'T':40},'E1':{'epochs':1},'E10':{'epochs':10},'lr0.02':{'lr':0.02},'lr0.1':{'lr':0.1},'uniform_T40':{'wmode':'u','T':40}}
res={c:[] for c in CFG}; cen=[]; mean=[]
for seed in range(5):
    names,S=make_split(ponds,seed)
    Xc=np.vstack([S[n][0][0] for n in names]); yc=np.concatenate([S[n][0][1] for n in names])
    ac=min(ALPHAS,key=lambda a:valmse({n:ridge(Xc,yc,a) for n in names},S,names)); wc=ridge(Xc,yc,ac)
    cen.append(score({n:wc for n in names},S,names)['micro'])
    g=yc.mean(); mean.append(score({n:np.r_[np.zeros(D),g] for n in names},S,names)['micro'])
    for c,kw in CFG.items():
        runs={a:fedavg(S,names,a,seed,**kw) for a in ALPHAS}
        af=min(ALPHAS,key=lambda a:valmse({n:runs[a]['w'] for n in names},S,names))
        res[c].append(score({n:runs[af]['w'] for n in names},S,names)['micro'])
    print(seed,{c:round(v[-1],3) for c,v in res.items()},round(cen[-1],3),flush=True)
json.dump({'cfg':res,'central':cen,'mean':mean},open('sens.json','w'))
