import sys,json;sys.path.insert(0,'.')
import rcore
from rcore import *
ponds=load_ponds(); allnames=list(ponds); res=[]
for seed in range(3):
    row={}
    for h in allnames:
        oth={n:ponds[n] for n in allnames if n!=h}; names,S=make_split(oth,seed); mu,sd=rcore.SC
        d=ponds[h]; step=max(1,len(d)//3000); Xh,yh=XY(d.iloc[::step]); Xh=(Xh-mu)/sd
        wl={}
        for n in names:
            b=min(ALPHAS,key=lambda a:float(np.mean((pred(ridge(*S[n][0],a),S[n][1][0])-S[n][1][1])**2))); wl[n]=ridge(*S[n][0],b)
        Xc=np.vstack([S[n][0][0] for n in names]); yc=np.concatenate([S[n][0][1] for n in names])
        ac=min(ALPHAS,key=lambda a:valmse({n:ridge(Xc,yc,a) for n in names},S,names)); wc=ridge(Xc,yc,ac)
        runs={a:fedavg(S,names,a,seed)['w'] for a in ALPHAS}
        af=min(ALPHAS,key=lambda a:valmse({n:runs[a] for n in names},S,names)); wf=runs[af]
        r=lambda w:float(np.sqrt(np.mean((pred(w,Xh)-yh)**2)))
        bor=[r(wl[n]) for n in names]
        row[h]=dict(fedavg=r(wf),central=r(wc),borrowed_mean=float(np.mean(bor)),borrowed_best=float(np.min(bor)),borrowed_worst=float(np.max(bor)),mean=r(np.r_[np.zeros(D),yc.mean()]))
    res.append(row); print('seed',seed,flush=True)
json.dump(res,open('lopo.json','w'))
