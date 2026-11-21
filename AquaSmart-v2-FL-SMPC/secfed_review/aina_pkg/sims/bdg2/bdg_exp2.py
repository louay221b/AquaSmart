import sys,json,time;sys.path.insert(0,'/home/claude/work/bdg2')
import bcore,rcore
from rcore import *
sites=bcore.load_sites(); D=bcore.D; out={'scarce':{},'loso':[]}
def fit(S,names,seed):
    yc=np.concatenate([S[n][0][1] for n in names]); Xc=np.vstack([S[n][0][0] for n in names]); r={}
    r['mean']=score({n:np.r_[np.zeros(D),yc.mean()] for n in names},S,names)['micro']
    wl={n:ridge(*S[n][0],min(ALPHAS,key=lambda a:float(np.mean((pred(ridge(*S[n][0],a),S[n][1][0])-S[n][1][1])**2)))) for n in names}
    r['local']=score(wl,S,names)['micro']
    ac=min(ALPHAS,key=lambda a:valmse({n:ridge(Xc,yc,a) for n in names},S,names)); r['central']=score({n:ridge(Xc,yc,ac) for n in names},S,names)['micro']
    runs={a:fedavg(S,names,a,seed)['w'] for a in ALPHAS}
    af=min(ALPHAS,key=lambda a:valmse({n:runs[a] for n in names},S,names)); r['fedavg']=score({n:runs[af] for n in names},S,names)['micro']
    return r,wl,runs[af]
for cap in (30,100,300,1000,3000):
    rows=[]
    for seed in range(5):
        names,S=bcore.make_split(sites,seed,cap=cap); rows.append(fit(S,names,seed)[0])
    out['scarce'][cap]={k:[r[k] for r in rows] for k in rows[0]}
    print(cap,{k:round(float(np.mean(v)),3) for k,v in out['scarce'][cap].items()},flush=True)
for seed in range(3):
    row={}
    for h in sites:
        oth={n:sites[n] for n in sites if n!=h}; names,S=bcore.make_split(oth,seed)
        # held-out site: all its data in same standardisation is approximated by re-split with itself included
        allS=bcore.make_split(sites,seed)[1]
        r,wl,wf=fit(S,names,seed)
        Xh,yh=allS[h][2]   # test part (standardised with all sites; close to pooled scaler)
        e=lambda w:float(np.sqrt(np.mean((pred(w,Xh)-yh)**2)))
        bor=[e(wl[n]) for n in names]
        row[h]=dict(fedavg=e(wf),mean=e(np.r_[np.zeros(D),np.concatenate([S[n][0][1] for n in names]).mean()]),borrowed_mean=float(np.mean(bor)),borrowed_best=min(bor),borrowed_worst=max(bor))
    out['loso'].append(row); print('loso',seed,flush=True)
json.dump(out,open('bdg2.json','w'))
