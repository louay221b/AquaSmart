"""Exploratory nested temporal tuning of proximal ridge around a FedAvg model.
No outer test features/targets are used for fitting, tuning, or scoring.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'correction_v1'))
sys.path.insert(0, str(ROOT/'sims'))
from protocol import FE, TARGET, load_ponds, chronological_parts
import rcore as r


def inner_boundary(train):
    sizes = train.groupby('block', sort=False).size().to_numpy()
    if len(sizes) < 2:
        raise ValueError('Outer train needs at least two blocks for inner tuning')
    ends = np.cumsum(sizes)[:-1]
    cut = int(ends[np.argmin(np.abs(ends - .7*len(train)))])
    a,b = train.iloc[:cut], train.iloc[cut:]
    assert a.t.max() < b.t.min() and not set(a.interval)&set(b.interval)
    return a,b


def partitions(ponds):
    inner, outer, counts = {}, {}, {}
    for n,df in sorted(ponds.items()):
        tr,va,_ = chronological_parts(df)
        a,b = inner_boundary(tr)
        inner[n]=(a,b); outer[n]=(tr,va)
        counts[n]={'inner_train_blocks':int(a.block.nunique()),
                   'inner_val_blocks':int(b.block.nunique()),
                   'inner_train_runs':int(a.interval.nunique()),
                   'inner_val_runs':int(b.interval.nunique()),
                   'outer_train_runs':int(tr.interval.nunique()),
                   'outer_val_runs':int(va.interval.nunique()),
                   'inner_train_end':str(a.t.max()),'inner_val_start':str(b.t.min()),
                   'inner_val_end':str(b.t.max()),'outer_val_start':str(va.t.min())}
    return inner,outer,counts


def prepare_two(parts, seed, cap, mode):
    rng=np.random.default_rng(seed); raw={}
    for n,ab in sorted(parts.items()):
        tr,va=ab
        if mode=='intervals':
            tr,va=[p.groupby('interval',sort=False).agg({**{c:'mean' for c in FE},TARGET:'first'}).reset_index(drop=True) for p in (tr,va)]
        if len(tr)>cap: tr=tr.iloc[np.sort(rng.choice(len(tr),cap,replace=False))]
        raw[n]=(tr,va)
    fallback=pd.concat([tr.nitrate for tr,_ in raw.values()]).median()
    if not np.isfinite(fallback): raise ValueError('Nitrate unavailable in fitting partition')
    result={}
    for n,(tr,va) in raw.items():
        med=tr.nitrate.median() if tr.nitrate.notna().any() else fallback
        result[n]=tuple((q[FE].assign(nitrate=q.nitrate.fillna(med)).to_numpy(float),np.log1p(q[TARGET].to_numpy(float))) for q in (tr,va))
    x=np.vstack([a[0][0] for a in result.values()]); mu=x.mean(0); sd=x.std(0)+1e-9
    return {n:tuple(((x-mu)/sd,y) for x,y in a) for n,a in result.items()}


def proximal(X,y,w,alpha,lam):
    """MSE + alpha||slopes||² + lambda||theta-w||²; None = frozen global."""
    if lam is None: return np.array(w,copy=True)
    if alpha<=0 or lam<0: raise ValueError('alpha positive, lambda nonnegative')
    Z=np.c_[X,np.ones(len(X))]; p=Z.shape[1]; R=np.eye(p); R[-1,-1]=0
    return np.linalg.solve(Z.T@Z/len(X)+alpha*R+lam*np.eye(p),Z.T@y/len(X)+lam*w)


def metrics(models,S):
    e={n:r.pred(models[n],v[1][0])-v[1][1] for n,v in S.items()}
    z=np.concatenate(list(e.values())); y=np.concatenate([v[1][1] for v in S.values()]); ss=float(np.sum((y-y.mean())**2))
    per={n:float(np.sqrt(np.mean(a*a))) for n,a in e.items()}
    return dict(micro=float(np.sqrt(np.mean(z*z))),macro=float(np.mean(list(per.values()))),r2=float(1-np.sum(z*z)/ss) if ss else None,per_pond=per)


def global_config(S,seed,selection,quick):
    alphas=(.1,) if quick else tuple(float(a) for a in r.ALPHAS)
    rounds=(20,) if quick else (20,40,80)
    lrs=(.05,) if quick else (.05,.1)
    epochs=(1,) if quick else (1,5)
    best=None
    for a in alphas:
        for lr in lrs:
            for ep in epochs:
                f=r.fedavg(S,list(S),a,seed,T=max(rounds),lr=lr,epochs=ep,track=True)
                for T in rounds:
                    w=f['hist'][T-1]; score=metrics(dict.fromkeys(S,w),S)[selection]
                    key=(score,T*ep,T,ep,lr,a)
                    if best is None or key<best[0]: best=(key,dict(alpha=a,lr=lr,epochs=ep,rounds=T),w.copy())
    return best[1],best[2]


def choose_local(S,w,grid,counts,selection,min_blocks):
    candidates=[]
    for a,l in grid:
        models={n:proximal(*v[0],w,a,l) for n,v in S.items()}
        m=metrics(models,S)
        candidates.append((a,l,m))
    # Ties prefer earlier grid entries; pooled selection includes all ponds.
    common=min(candidates,key=lambda x:x[2][selection])[:2]
    adaptive={n:(min(candidates,key=lambda x:x[2]['per_pond'][n])[:2]
                 if min(counts[n]['inner_train_blocks'],counts[n]['inner_val_blocks'])>=min_blocks else common) for n in S}
    return common,adaptive


def experiment(inner,outer,counts,seed,selection='macro',quick=False,min_blocks=5):
    c,iw=global_config(inner,seed,selection,quick)
    aa=(.01,.1,1.) if quick else tuple(float(a) for a in r.ALPHAS)
    ll=(0.,.1,1.,None) if quick else (0.,.001,.01,.1,1.,10.,100.,None)
    grid=[(a,l) for a in aa for l in ll if l is not None]+[(aa[0],None)]
    common,ad=choose_local(inner,iw,grid,counts,selection,min_blocks)
    lc,la=choose_local(inner,iw,[(a,0.) for a in aa],counts,selection,min_blocks)
    X=np.vstack([v[0][0] for v in inner.values()]); y=np.concatenate([v[0][1] for v in inner.values()])
    ca=min(aa,key=lambda a:metrics(dict.fromkeys(inner,r.ridge(X,y,a)),inner)[selection])
    # All configuration choices are complete before accessing the outer validation arrays.
    args=dict(T=c['rounds'],lr=c['lr'],epochs=c['epochs'])
    plain=r.fedavg(outer,list(outer),c['alpha'],seed,**args)
    masked=r.fedavg(outer,list(outer),c['alpha'],seed,mode='masked',**args)
    w=plain['w']; sw=masked['w']
    def fitted(cfg,g):return {n:proximal(*v[0],g,*cfg[n]) for n,v in outer.items()}
    X=np.vstack([v[0][0] for v in outer.values()]); y=np.concatenate([v[0][1] for v in outer.values()])
    models={'global_mean':dict.fromkeys(outer,np.r_[np.zeros(X.shape[1]),y.mean()]),
            'local_mean':{n:np.r_[np.zeros(X.shape[1]),v[0][1].mean()] for n,v in outer.items()},
            'local_ridge_common':fitted(dict.fromkeys(outer,lc),w),
            'local_ridge_adaptive':fitted(la,w),
            'central_ridge':dict.fromkeys(outer,r.ridge(X,y,ca)),
            'fedavg':dict.fromkeys(outer,w),'secfed_gaussian':dict.fromkeys(outer,sw),
            'proximal_common':fitted(dict.fromkeys(outer,common),w),
            'proximal_adaptive':fitted(ad,w),
            'proximal_adaptive_secfed_gaussian':fitted(ad,sw)}
    diff=max(float(np.max(np.abs(models['proximal_adaptive'][n]-models['proximal_adaptive_secfed_gaussian'][n]))) for n in outer)
    if diff>1e-8 or np.max(np.abs(w-sw))>1e-8: raise AssertionError('Mask equivalence failure')
    return {'seed':seed,'global_config':c,'central_alpha':ca,'proximal_common':common,'proximal_adaptive':ad,'local_common':lc,'local_adaptive':la,
            'eligible_local_selection':[n for n in outer if min(counts[n]['inner_train_blocks'],counts[n]['inner_val_blocks'])>=min_blocks],
            'outer_validation':{k:metrics(v,outer) for k,v in models.items()},
            'max_global_diff':float(np.max(np.abs(w-sw))),'max_personal_diff':diff,
            'theoretical_cost':{'fedavg_bytes':plain['bytes'],'secfed_gaussian_bytes':masked['bytes'],'personalization_extra_model_upload_bytes':0}}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dataset-dir',type=Path,required=True); p.add_argument('--mode',choices=['rows','intervals'],required=True); p.add_argument('--seeds',default='0,1,2,3,4'); p.add_argument('--cap',type=int,default=3000); p.add_argument('--selection',choices=['micro','macro'],default='macro'); p.add_argument('--quick',action='store_true'); p.add_argument('--output-dir',type=Path,default=Path('adaptive_results')); a=p.parse_args()
    seeds=[int(x) for x in a.seeds.split(',')];
    if a.cap<1 or min(seeds)<0: p.error('invalid cap or seeds')
    ponds,clean=load_ponds(a.dataset_dir); ip,op,counts=partitions(ponds)
    result={'protocol':'adaptive-proximal-v3-nested-exploratory','mode':a.mode,'selection':a.selection,'cap':a.cap,'quick':a.quick,'seeds':seeds,'min_inner_blocks':5,'cleaning':clean,'counts':counts,'runs':[], 'notes':['Outer validation previously examined; not independent test','Inner tuning only; preprocessing fitted separately on inner and outer training','Statistical Gaussian masking only, pooled preprocessing not private','Small ponds use common hyperparameters; threshold selected as exploratory design','null lambda denotes frozen global model']}
    a.output_dir.mkdir(parents=True,exist_ok=True); out=a.output_dir/f'{a.mode}_{a.selection}{"_quick" if a.quick else ""}.json'
    for seed in seeds:
        rec=experiment(prepare_two(ip,seed,a.cap,a.mode),prepare_two(op,seed,a.cap,a.mode),counts,seed,a.selection,a.quick)
        result['runs'].append(rec)
        print('Seed',seed,{k:{s:round(v[s],4) for s in ('micro','macro')} for k,v in rec['outer_validation'].items()},flush=True)
        print('  Global:',rec['global_config'],'Common:',rec['proximal_common'],'Eligible:',rec['eligible_local_selection'],'Max diff:',rec['max_personal_diff'],flush=True)
        out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved:',out)

if __name__=='__main__': main()
