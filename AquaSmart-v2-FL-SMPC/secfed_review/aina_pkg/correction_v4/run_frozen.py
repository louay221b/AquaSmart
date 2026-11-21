"""Frozen retrospective evaluation of V3 and two ablations. No test-based tuning."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'correction_v3'))
import run_adaptive as a
r=a.r
from protocol import chronological_parts, load_ponds
sys.path.insert(0,str(ROOT/'correction_v2'))
from run_personalized import adapt
HERE=Path(__file__).resolve().parent
REFERENCE=HERE/'v3_reference.json'
BASE='frozen-v4-retrospective'

def dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,allow_nan=False)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def code_hashes():
    files=['correction_v1/protocol.py','correction_v3/run_adaptive.py','correction_v2/run_personalized.py','correction_v1/run_corrected.py','sims/rcore.py','correction_v4/run_frozen.py','correction_v4/v3_reference.json']
    return {f:sha(ROOT/f) for f in files}

def verify_data(ref,clean,counts):
    if ref['cleaning']!=clean or ref['counts']!=counts:
        raise ValueError('Dataset fingerprints, cleaning or temporal partition differ from V3; stop and inspect, do not substitute results.')

def date_audit(ponds):
    parts={n:chronological_parts(d) for n,d in ponds.items()}
    latest=max(tr.t.max() for tr,_,_ in parts.values())
    out={}
    for n,(tr,va,te) in parts.items():
        out[n]={'bounds':[[str(p.t.min()),str(p.t.max())] for p in (tr,va,te)],
                'weight_runs':[int(p.interval.nunique()) for p in (tr,va,te)],
                'other_or_same_train_after_test_start':[k for k,(t,_,_) in parts.items() if t.t.max()>=te.t.min()],
                'other_or_same_train_after_val_start':[k for k,(t,_,_) in parts.items() if t.t.max()>=va.t.min()]}
    return {'latest_training_end':str(latest),'all_sites_time_safe_for_test':all(not v['other_or_same_train_after_test_start'] for v in out.values()),'sites':out}

def global_fit(S,c,seed,mode='plain'):
    return r.fedavg(S,list(S),c['alpha'],seed,T=c['rounds'],lr=c['lr'],epochs=c['epochs'],mode=mode)

def select_ablations(S,rec,counts,min_blocks):
    """Same inner split and frozen global config; no external validation/test."""
    c=rec['global_config']; seed=rec['seed']; w=global_fit(S,c,seed)['w']
    # Same alpha grid and fallback rule as V3, only remove finite positive lambda.
    aa=tuple(float(x) for x in r.ALPHAS)
    common, binary=a.choose_local(S,w,[(x,0.) for x in aa]+[(aa[0],None)],counts,'macro',min_blocks)
    best=None
    for ep in (0,1,3,5,10,20):
        for lr in ((.005,) if ep==0 else (.005,.01,.05)):
            m=a.metrics(adapt(w,S,c['alpha'],seed,ep,lr),S)
            key=(m['macro'],ep,lr)
            if best is None or key<best[0]: best=(key,{'epochs':ep,'lr':lr,'alpha':c['alpha']})
    return {'binary_common':common,'binary_adaptive':binary,'sgd_config':best[1]}

def fit_models(S,rec):
    """Training only. The second partition is not read by fitting or configuration."""
    c=rec['global_config']; seed=rec['seed']; p=global_fit(S,c,seed); q=global_fit(S,c,seed,'masked'); w,sw=p['w'],q['w']
    def fit(cfg,g): return {n:a.proximal(*v[0],g,*cfg[n]) for n,v in S.items()}
    def common(cfg): return dict.fromkeys(S,cfg)
    X=np.vstack([v[0][0] for v in S.values()]); y=np.concatenate([v[0][1] for v in S.values()]); pc=rec['sgd_config']
    m={'global_mean':common(np.r_[np.zeros(X.shape[1]),y.mean()]),'local_mean':{n:np.r_[np.zeros(X.shape[1]),v[0][1].mean()] for n,v in S.items()},
       'local_ridge_common':fit(common(rec['local_common']),w),'local_ridge_adaptive':fit(rec['local_adaptive'],w),
       'central_ridge':common(r.ridge(X,y,rec['central_alpha'])),'fedavg':common(w),'secfed_gaussian':common(sw),
       'proximal_common':fit(common(rec['proximal_common']),w),'proximal_adaptive':fit(rec['proximal_adaptive'],w),
       'proximal_adaptive_secfed_gaussian':fit(rec['proximal_adaptive'],sw),
       'binary_local_global':fit(rec['binary_adaptive'],w),'binary_local_global_secfed_gaussian':fit(rec['binary_adaptive'],sw),
       'personalized_sgd':adapt(w,S,pc['alpha'],seed,pc['epochs'],pc['lr']),
       'personalized_sgd_secfed_gaussian':adapt(sw,S,pc['alpha'],seed,pc['epochs'],pc['lr'])}
    diffs={'global':float(np.max(np.abs(w-sw)))}
    for k,sk in [('proximal_adaptive','proximal_adaptive_secfed_gaussian'),('binary_local_global','binary_local_global_secfed_gaussian'),('personalized_sgd','personalized_sgd_secfed_gaussian')]:
        diffs[k]=max(float(np.max(np.abs(m[k][n]-m[sk][n]))) for n in S)
    if max(diffs.values())>1e-8: raise AssertionError('Mask equivalence failed')
    return m,diffs,{'fedavg_bytes':p['bytes'],'secfed_gaussian_bytes':q['bytes'],'extra_personal_model_upload_bytes':0}

def reproduce(scores,ref,tol=1e-7):
    worst=0.
    for k,v in ref.items():
        for metric in ('micro','macro'):
            worst=max(worst,abs(scores[k][metric]-v[metric]))
        for n,value in v['per_pond'].items(): worst=max(worst,abs(scores[k]['per_pond'][n]-value))
    if worst>tol: raise ValueError(f'V3 reproduction failed: max RMSE difference {worst:.3g}; stop before test.')
    return worst

def freeze(ponds,clean,ref,mode):
    inner,outer,counts=a.partitions(ponds); verify_data(ref,clean,counts)
    records=[]
    for old in ref['runs']:
        seed=old['seed']; S=a.prepare_two(inner,seed,ref['cap'],mode)
        rec={k:v for k,v in old.items() if k!='outer_validation'}
        rec.update(select_ablations(S,rec,counts,ref['min_inner_blocks']))
        V=a.prepare_two(outer,seed,ref['cap'],mode); models,diffs,cost=fit_models(V,rec)
        met={k:a.metrics(m,V) for k,m in models.items()}
        rec['v3_reproduction_max_rmse_diff']=reproduce(met,old['outer_validation']); records.append(rec)
        print(f"Freeze {mode} seed {seed}: V3 max RMSE diff={rec['v3_reproduction_max_rmse_diff']:.3g}; binary and SGD inner-only selected",flush=True)
    return {'protocol':BASE,'mode':mode,'cap':ref['cap'],'seeds':ref['seeds'],'selection':'macro','min_inner_blocks':ref['min_inner_blocks'],'code_sha256':code_hashes(),'data':clean,'counts':counts,'chronology':date_audit(ponds),'runs':records,
            'notes':['Primary intervals; rows sensitivity; original manuscript inspected test previously','Retrospective pond-relative split; may use other ponds future data; not prospective','No outer validation/test tuning; V3 settings reused; binary and SGD picked on inner train/val','No train+validation refit; frozen original 60% training partitions','Gaussian masking simulation, pooled preprocessing not private']}

def evaluate(ponds,manifest):
    # fit with training only by passing identical training pair, then score test once.
    parts={n:chronological_parts(d) for n,d in sorted(ponds.items())}; mode=manifest['mode']; cap=manifest['cap']; runs=[]
    for rec in manifest['runs']:
        seed=rec['seed']; S=a.prepare_two({n:(p[0],p[2]) for n,p in parts.items()},seed,cap,mode)
        train={n:(v[0],v[0]) for n,v in S.items()}
        models,diffs,cost=fit_models(train,rec)
        scores={k:a.metrics(m,S) for k,m in models.items()}
        runs.append({'seed':seed,'test':scores,'max_param_diff':diffs,'theoretical_cost':cost,'test_units':{n:len(v[1][1]) for n,v in S.items()}})
        print(f'Test {mode} seed {seed}: '+str({k:{x:round(v[x],4) for x in ('micro','macro')} for k,v in scores.items()}),flush=True)
    summary={k:{metric:{'mean':float(np.mean([x['test'][k][metric] for x in runs])),'std':float(np.std([x['test'][k][metric] for x in runs],ddof=1)) if len(runs)>1 else 0.} for metric in ('micro','macro')} for k in runs[0]['test']}
    return {'protocol':BASE,'mode':mode,'runs':runs,'summary':summary,'notes':manifest['notes'],'chronology':manifest['chronology']}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dataset-dir',required=True,type=Path); p.add_argument('--mode',required=True,choices=['intervals','rows']); p.add_argument('--phase',required=True,choices=['freeze','test']); p.add_argument('--output-dir',type=Path,default=Path('frozen_results')); p.add_argument('--accept-retrospective-test',action='store_true'); args=p.parse_args()
    ref=json.loads(REFERENCE.read_text(encoding='utf-8'))[args.mode]; dest=args.output_dir; mp=dest/f'{args.mode}_manifest.json'; testpath=dest/f'{args.mode}_test.json'
    if args.phase=='freeze' and mp.exists(): p.error('Manifest already exists: preserve it; do not overwrite')
    if args.phase=='test' and (not args.accept_retrospective_test or not mp.exists() or testpath.exists()): p.error('Need existing frozen manifest, explicit --accept-retrospective-test and unused test output path')
    ponds,clean=load_ponds(args.dataset_dir); _,_,counts=a.partitions(ponds); verify_data(ref,clean,counts)
    if args.phase=='freeze':
        m=freeze(ponds,clean,ref,args.mode); dump(mp,m); print('Saved:',mp); print('SHA256:',sha(mp)); print('All sites chronologically safe for prospective test:',m['chronology']['all_sites_time_safe_for_test'])
    else:
        m=json.loads(mp.read_text(encoding='utf-8'))
        if m['mode']!=args.mode or m['code_sha256']!=code_hashes() or m['data']!=clean or m['counts']!=counts: raise ValueError('Frozen manifest incompatible with current code or data')
        result=evaluate(ponds,m); result['manifest_sha256']=sha(mp); dump(testpath,result); print('Saved:',testpath)
if __name__=='__main__': main()
