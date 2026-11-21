"""Calendar-aligned retrospective evaluation; no test-based hyperparameter search."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'correction_v4'))
import run_frozen as f
a=f.a; r=f.r
BASE='synchronized-v5-retrospective'
# Calendar choices fixed before inspecting new results; recorded wall time, not UTC.
CUT1=pd.Timestamp('2021-07-20'); CUT2=pd.Timestamp('2021-08-10')
MIN_FIT=5; MIN_VAL=2; MIN_TEST=1; MIN_PERSONAL=5
CAP=3000; SEEDS=[0,1,2,3,4]
NOTES=['Calendar-aligned retrospective estimate, not independently collected prospective data',
       'Blocks are weight-run/timestamp proxies inferred retrospectively; crossing blocks discarded',
       'Same-recorded-time interval sensor averages estimate contemporaneous weight, not a future-horizon forecast',
       'Cohort requires training and validation coverage plus at least one test block; coverage selection is retrospective',
       'Selection macro on inner validation only; refit on fit+inner validation complete blocks; test never used for tuning',
       'Pooled preprocessing and Gaussian masking simulation, not end-to-end cryptographic privacy',
       'Boundaries in recorded wall time; timezone and label acquisition times not independently verified']

def split_one(d,c1=CUT1,c2=CUT2):
    if not c1<c2: raise ValueError('Calendar boundaries must increase')
    bounds=d.groupby('block',sort=False).t.agg(['min','max'])
    masks=[bounds['max']<c1,(bounds['min']>=c1)&(bounds['max']<c2),bounds['min']>=c2]
    ids=[bounds.index[m] for m in masks]
    parts=[d[d.block.isin(i)].copy() for i in ids]
    discarded=d[~d.block.isin(set().union(*[set(i) for i in ids]))]
    # Block construction also prevents one interval from spanning two blocks.
    for i in range(3):
        for j in range(i+1,3):
            if len(parts[i]) and len(parts[j]):
                assert parts[i].t.max()<parts[j].t.min()
                assert not set(parts[i].interval)&set(parts[j].interval)
    return parts,discarded

def details(d):
    return {'rows':len(d),'blocks':int(d.block.nunique()),'intervals':int(d.interval.nunique()),
            'bounds':[str(d.t.min()),str(d.t.max())] if len(d) else None}

def partition(ponds,c1=CUT1,c2=CUT2):
    eligible={}; report={}; counts={}
    for n,d in sorted(ponds.items()):
        (tr,va,te),discarded=split_one(d,c1,c2)
        s=[details(p) for p in (tr,va,te)]
        reason=[name for item,min_n,name in zip(s,(MIN_FIT,MIN_VAL,MIN_TEST),('fit coverage','inner validation coverage','test coverage')) if item['blocks']<min_n]
        report[n]={'fit':s[0],'inner_validation':s[1],'test':s[2],'discarded':details(discarded),'included':not reason,'exclusion_reasons':reason}
        if reason:continue
        eligible[n]=(tr,va,te)
        counts[n]={'inner_train_blocks':s[0]['blocks'],'inner_val_blocks':s[1]['blocks']}
    if eligible:
        latest_fit=max(t.t.max() for t,_,_ in eligible.values()); earliest_val=min(v.t.min() for _,v,_ in eligible.values())
        latest_refit=max(v.t.max() for _,v,_ in eligible.values()); earliest_test=min(t.t.min() for _,_,t in eligible.values())
        assert latest_fit<earliest_val and latest_refit<earliest_test
        audit={'latest_fit':str(latest_fit),'earliest_inner_validation':str(earliest_val),'latest_refit':str(latest_refit),'earliest_test':str(earliest_test),'calendar_safe':True}
    else:audit={'calendar_safe':False}
    return eligible,counts,{'cut1':str(c1),'cut2':str(c2),'sites':report,'eligible':list(eligible),'chronology':audit}

def code_hashes():
    return {**f.code_hashes(),'correction_v5/run_synchronized.py':f.sha(Path(__file__))}

def select(S,counts,seed,quick=False):
    c,w=a.global_config(S,seed,'macro',quick)
    aa=(.01,.1,1.) if quick else tuple(float(x) for x in r.ALPHAS)
    ll=(0.,.1,1.) if quick else (0.,.001,.01,.1,1.,10.,100.)
    grid=[(alpha,lam) for alpha in aa for lam in ll]+[(aa[0],None)]
    common,ad=a.choose_local(S,w,grid,counts,'macro',MIN_PERSONAL)
    lc,la=a.choose_local(S,w,[(x,0.) for x in aa],counts,'macro',MIN_PERSONAL)
    X=np.vstack([v[0][0] for v in S.values()]);y=np.concatenate([v[0][1] for v in S.values()])
    ca=min(aa,key=lambda x:a.metrics(dict.fromkeys(S,r.ridge(X,y,x)),S)['macro'])
    rec={'seed':seed,'global_config':c,'central_alpha':ca,'proximal_common':common,'proximal_adaptive':ad,'local_common':lc,'local_adaptive':la}
    # Full grid uses exactly V4 binary and SGD ablation rules; quick is smoke-only.
    rec.update(f.select_ablations(S,rec,counts,MIN_PERSONAL))
    return rec

def freeze(parts,counts,audit,clean,mode,seeds=SEEDS,quick=False):
    if len(parts)<2:raise ValueError('Fewer than two eligible sites: no federated comparison')
    runs=[]
    for seed in seeds:
        S=a.prepare_two({n:(v[0],v[1]) for n,v in parts.items()},seed,CAP,mode)
        rec=select(S,counts,seed,quick);runs.append(rec)
        print('Freeze',mode,seed,rec['global_config'],flush=True)
    return {'protocol':BASE,'mode':mode,'cap':CAP,'seeds':seeds,'quick':quick,'selection':'macro','data':clean,'audit':audit,'counts':counts,'code_sha256':code_hashes(),'runs':runs,'notes':NOTES}

def evaluate(parts,manifest):
    runs=[]
    for rec in manifest['runs']:
        seed=rec['seed']
        pairs={n:(pd.concat([v[0],v[1]],ignore_index=True),v[2]) for n,v in parts.items()}
        S=a.prepare_two(pairs,seed,manifest['cap'],manifest['mode'])
        models,diff,cost=f.fit_models({n:(v[0],v[0]) for n,v in S.items()},rec)
        scores={k:a.metrics(m,S) for k,m in models.items()}
        runs.append({'seed':seed,'test':scores,'max_param_diff':diff,'theoretical_cost':cost,'test_units':{n:len(v[1][1]) for n,v in S.items()}})
        print('Test',manifest['mode'],seed,{k:{x:round(v[x],4) for x in ('micro','macro')} for k,v in scores.items()},flush=True)
    summary={}
    for k in runs[0]['test']:
        summary[k]={}
        for metric in ('micro','macro','r2'):
            vals=[u['test'][k][metric] for u in runs]
            summary[k][metric]={'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.} if all(v is not None for v in vals) else None
    return {'protocol':BASE,'mode':manifest['mode'],'runs':runs,'summary':summary,'audit':manifest['audit'],'notes':NOTES}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--phase',choices=['audit','freeze','test'],required=True);p.add_argument('--mode',choices=['intervals','rows'],default='intervals');p.add_argument('--output-dir',type=Path,default=Path('synchronized_results'));p.add_argument('--manifest-sha256');p.add_argument('--accept-retrospective-test',action='store_true');args=p.parse_args()
    ponds,clean=f.load_ponds(args.dataset_dir)
    original=json.loads(f.REFERENCE.read_text(encoding='utf-8'))['intervals']
    if clean!=original['cleaning']:raise ValueError('CSV fingerprints or cleaning differ from V3; stop.')
    parts,counts,audit=partition(ponds);out=args.output_dir
    if args.phase=='audit':
        for n,v in audit['sites'].items():
            print(n,'blocks:',[v[s]['blocks'] for s in ('fit','inner_validation','test')],'intervals:',[v[s]['intervals'] for s in ('fit','inner_validation','test')],'discarded rows:',v['discarded']['rows'],'included:',v['included'],v['exclusion_reasons'])
        print('Eligible:',audit['eligible']);print('Chronology:',audit['chronology'])
        path=out/'calendar_audit.json';f.dump(path,{'protocol':BASE,'data':clean,'audit':audit,'notes':NOTES});print('Saved:',path);return
    mp=out/f'{args.mode}_manifest.json';tp=out/f'{args.mode}_test.json'
    if args.phase=='freeze':
        if mp.exists():p.error('Manifest exists; do not overwrite')
        m=freeze(parts,counts,audit,clean,args.mode);f.dump(mp,m);print('Saved:',mp);print('SHA256:',f.sha(mp));return
    if not args.accept_retrospective_test or not mp.exists() or tp.exists():p.error('Require existing manifest, explicit retrospective consent, unused output')
    if args.manifest_sha256!=f.sha(mp):p.error('Supply exact recorded manifest SHA256')
    m=json.loads(mp.read_text(encoding='utf-8'))
    if m['protocol']!=BASE or m['mode']!=args.mode or m['data']!=clean or m['audit']!=audit or m['counts']!=counts or m['code_sha256']!=code_hashes() or m['quick']:raise ValueError('Manifest incompatible')
    d=evaluate(parts,m);d['manifest_sha256']=f.sha(mp);f.dump(tp,d);print('Saved:',tp)
if __name__=='__main__':main()
