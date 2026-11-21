"""BDG2 V6B: frozen calendar comparison, bounded sampling, full-test linear sufficient statistics."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import argparse, json, sys
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import pandas as pd
from audit_bdg2 import WEATHER, PARTS, timestamps, period, digest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'correction_v5'))
import run_synchronized as v5
f=v5.f; r=f.r
CAP=3000; VAL_CAP=10000; SEEDS=[0,1,2,3,4]; D=7
PROTO='bdg2-v6b-frozen-calendar'
NOTES=['Positive finite raw electricity only; contemporaneous estimation, not future forecasting.', 'Recorded local timestamps and two-day embargo; timezone alignment not independently verified.', 'Uniform sample of building-hours per site: fit/refit cap 3000, validation cap 10000 (fixed across seeds); all eligible test hours scored.', 'Refit sample drawn from combined fit and validation; macro tuning; pooled training scaler, not private.', 'Gaussian masking simulation; equal sampled site weights when cap reached; not cryptographic privacy.', 'Same fixed calendar across seeds; no independent temporal replications. BDG2 domain previously examined in original manuscript.']

@contextmanager
def dimension():
    old=r.P
    r.P=D+1
    try: yield
    finally:r.P=old

def code_hashes():
    return {**v5.code_hashes(),**{f'correction_v6/{n}':f.sha(Path(__file__).parent/n) for n in ('audit_bdg2.py','run_bdg2.py')}}

def validate_audit(directory,audit):
    if audit['protocol']!='bdg2-v6-audit':raise ValueError('Wrong audit protocol')
    for name,info in audit['data'].items():
        p=directory/(name+'.csv')
        if p.stat().st_size!=info['bytes'] or digest(p)!=info['sha256']:raise ValueError('Data differs from audit: '+name)
    names=audit['eligible_for_training']
    if len(names)<2 or len(set(names))!=len(names):raise ValueError('Invalid audit cohort')
    return names

def indices(n,cap,seed):
    return np.arange(n) if n<=cap else np.sort(np.random.default_rng(seed).choice(n,cap,replace=False))

def collect(directory,audit,seeds,phase,chunk_size=168,cap=CAP,val_cap=VAL_CAP):
    """Sample by rank among valid building-hours, independent of values and chunk size.
    Test metrics use exact raw second moments over ALL test rows, no test sampling.
    """
    if phase not in ('freeze','test'):raise ValueError('phase')
    names=validate_audit(directory,audit)
    md=pd.read_csv(directory/'metadata.csv'); md['sqm']=pd.to_numeric(md.sqm,errors='coerce')
    headers=pd.read_csv(directory/'electricity.csv',nrows=0).columns
    md=md[md.site_id.astype(str).isin(names)&np.isfinite(md.sqm)&(md.sqm>0)&md.building_id.isin(headers)]
    groups={n:md[md.site_id.astype(str)==n] for n in names}
    wx=pd.read_csv(directory/'weather.csv',usecols=['timestamp','site_id']+WEATHER);wx.timestamp=timestamps(wx.timestamp)
    for c in WEATHER:wx[c]=pd.to_numeric(wx[c],errors='coerce')
    wx=wx.set_index(['site_id','timestamp'])[WEATHER]
    counts={n:{p:0 for p in PARTS} for n in names}; offsets={n:0 for n in names}
    picks={}; buffers={}; moments={n:np.zeros((D+2,D+2)) for n in names}
    for i,n in enumerate(names):
        q=audit['sites'][n]['partitions']; nf=q['fit']['valid_rows']; nv=q['validation']['valid_rows']
        for s in seeds:
            key=(s,n,'train');picks[key]=indices(nf if phase=='freeze' else nf+nv,cap,100000+s*1000+i);buffers[key]=[]
        if phase=='freeze':
            key=(-1,n,'val');picks[key]=indices(nv,val_cap,900000+i);buffers[key]=[]
    def save(key,base,rows):
        ids=picks[key]; l,h=np.searchsorted(ids,[base,base+len(rows)]); j=ids[l:h]-base
        if len(j):buffers[key].append(rows[j])
    previous=None
    for chunk in pd.read_csv(directory/'electricity.csv',usecols=['timestamp']+list(md.building_id),chunksize=chunk_size):
        t=timestamps(chunk.timestamp)
        if not t.is_monotonic_increasing or t.duplicated().any() or (previous is not None and t.iloc[0]<=previous):raise ValueError('Electricity timestamps must be strictly increasing')
        previous=t.iloc[-1]; labels=period(t); hour=t.dt.hour.to_numpy(); sine=np.sin(2*np.pi*hour/24); cosine=np.cos(2*np.pi*hour/24); weekend=(t.dt.dayofweek.to_numpy()>=5).astype(float)
        for n,g in groups.items():
            vals=chunk[list(g.building_id)].apply(pd.to_numeric,errors='coerce').to_numpy(float)
            w=wx.reindex(pd.MultiIndex.from_arrays([[n]*len(t),t],names=['site_id','timestamp'])).to_numpy(float)
            valid=np.isfinite(vals)&(vals>0)&np.isfinite(w).all(axis=1)[:,None]
            areas=np.log(g.sqm.to_numpy(float))
            # Strict chronological source order ensures chunk-independent sampling.
            for part in PARTS:
                ii,jj=np.nonzero(valid & (labels==part)[:,None]);base=counts[n][part];counts[n][part]+=len(ii)
                if not len(ii) or (phase=='freeze' and part=='test'):continue
                rows=np.column_stack((w[ii],sine[ii],cosine[ii],weekend[ii],areas[jj],np.log1p(vals[ii,jj])))
                if part=='test':
                    z=np.column_stack((rows[:,:D],np.ones(len(rows)),rows[:,D])); moments[n]+=z.T@z
                elif phase=='freeze':
                    if part=='fit':
                        for s in seeds:save((s,n,'train'),base,rows)
                    else:save((-1,n,'val'),base,rows)
                else:
                    for s in seeds:save((s,n,'train'),offsets[n],rows)
                    offsets[n]+=len(rows)
    for n in names:
        for p in PARTS:
            if counts[n][p]!=audit['sites'][n]['partitions'][p]['valid_rows']:raise ValueError(f'Audit count mismatch {n}/{p}')
    out={k:np.concatenate(v) for k,v in buffers.items()}
    if any(len(out[k])!=len(picks[k]) for k in out):raise ValueError('Sampling count mismatch')
    return out,moments,counts

def prepared(samples,names,seed,phase):
    x=np.vstack([samples[seed,n,'train'][:,:D] for n in names]); mu=x.mean(0); sd=x.std(0); sd=np.where(sd<1e-8,1.,sd)
    def xy(z):return ((z[:,:D]-mu)/sd,z[:,D])
    S={n:(xy(samples[seed,n,'train']),xy(samples[-1,n,'val']) if phase=='freeze' else xy(samples[seed,n,'train'])) for n in names}
    return S,mu,sd

def moment_scores(models,moments,mu,sd):
    """For raw vector [x, 1, y], error = z @ [raw slopes, intercept, -1]."""
    per={}; sse=0.; yy=0.; sy=0.; count=0.
    for n,w in models.items():
        M=moments[n]; N=M[D,D]
        if not N:continue
        slopes=w[:-1]/sd; v=np.r_[slopes,w[-1]-mu@slopes,-1.]
        e=float(v@M@v); tol=1e-9*max(1.,float(np.abs(v)@np.abs(M)@np.abs(v)))
        if e < -tol:raise ValueError('Numerical SSE failure')
        e=max(0.,e);per[n]=float(np.sqrt(e/N));sse+=e;count+=N;yy+=M[-1,-1];sy+=M[D,-1]
    if not count:raise ValueError('No test data')
    sst=yy-sy*sy/count
    return {'micro':float(np.sqrt(sse/count)),'macro':float(np.mean(list(per.values()))),'r2':float(1-sse/sst) if sst>1e-10*max(1,yy) else None,'per_site':per,'n':int(count)}

def freeze(directory,audit,seeds,quick=False):
    names=audit['eligible_for_training']; samples,_,counts=collect(directory,audit,seeds,'freeze'); runs=[]
    blocks={n:{'inner_train_blocks':audit['sites'][n]['partitions']['fit']['days'],'inner_val_blocks':audit['sites'][n]['partitions']['validation']['days']} for n in names}
    with dimension():
        for seed in seeds:
            S,mu,sd=prepared(samples,names,seed,'freeze');rec=v5.select(S,blocks,seed,quick);rec['sample_counts']={n:[len(v[0][1]),len(v[1][1])] for n,v in S.items()};runs.append(rec)
            print('Freeze BDG2',seed,rec['global_config'],flush=True)
    return {'protocol':PROTO,'audit':audit,'code_sha256':code_hashes(),'seeds':seeds,'quick':quick,'cap':CAP,'validation_cap':VAL_CAP,'runs':runs,'notes':NOTES}

def evaluate(directory,manifest):
    audit=manifest['audit'];names=audit['eligible_for_training'];samples,moments,counts=collect(directory,audit,manifest['seeds'],'test');runs=[]
    with dimension():
        for rec in manifest['runs']:
            S,mu,sd=prepared(samples,names,rec['seed'],'test');models,diffs,cost=f.fit_models(S,rec)
            scores={k:moment_scores(m,moments,mu,sd) for k,m in models.items()}
            runs.append({'seed':rec['seed'],'test':scores,'max_param_diff':diffs,'theoretical_cost':cost,'test_units':{n:counts[n]['test'] for n in names}})
            print('Test BDG2',rec['seed'],{k:{m:round(v[m],4) for m in ('micro','macro')} for k,v in scores.items()},flush=True)
    summary={k:{m:{'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.} if all(v is not None for v in vals) else None for m in ('micro','macro','r2') for vals in [[u['test'][k][m] for u in runs]]} for k in runs[0]['test']}
    return {'protocol':PROTO,'runs':runs,'summary':summary,'notes':NOTES,'audit':audit}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--phase',choices=['freeze','test'],required=True);p.add_argument('--output-dir',type=Path,default=Path('bdg2_results'));p.add_argument('--audit',type=Path,default=Path('bdg2_results/calendar_audit.json'));p.add_argument('--manifest-sha256');p.add_argument('--accept-retrospective-test',action='store_true');p.add_argument('--quick',action='store_true');args=p.parse_args()
    audit=json.loads(args.audit.read_text(encoding='utf-8')); suffix='_quick' if args.quick else '';mp=args.output_dir/f'bdg2_manifest{suffix}.json';tp=args.output_dir/'bdg2_test.json'
    if args.phase=='freeze':
        if mp.exists():p.error('Manifest exists; preserve it')
        m=freeze(args.dataset_dir,audit,[42] if args.quick else SEEDS,args.quick);f.dump(mp,m);print('Saved:',mp);print('SHA256:',f.sha(mp));return
    if args.quick or not args.accept_retrospective_test or not mp.exists() or tp.exists():p.error('Require full frozen manifest, consent and unused test output')
    if args.manifest_sha256!=f.sha(mp):p.error('Manifest SHA256 mismatch')
    m=json.loads(mp.read_text(encoding='utf-8'))
    if m['protocol']!=PROTO or m['audit']!=audit or m['code_sha256']!=code_hashes() or m['quick'] or m['seeds']!=SEEDS:raise ValueError('Manifest incompatible')
    result=evaluate(args.dataset_dir,m);result['manifest_sha256']=f.sha(mp);f.dump(tp,result);print('Saved:',tp)
if __name__=='__main__':main()
