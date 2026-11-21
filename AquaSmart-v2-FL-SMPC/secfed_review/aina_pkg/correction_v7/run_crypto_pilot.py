"""V7A: exercise unchanged mTLS SecAgg on frozen, real weighted local model contributions.
Plaintext replay supplies a checkpoint; this is NOT end-to-end secure FL training.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE/'vendor'))
from fl_smpc.tls_secagg_v45 import tls_secure_sum_round
from fl_smpc.secure_aggregation import SCALE
sys.path.insert(0,str(ROOT/'correction_v6'))
import run_bdg2 as b
v=b.v5; f=b.f; r=b.r
MANIFESTS={
 'aqua':('synchronized_results/intervals_manifest.json','dc210c55643a97061046d4eb5cd8b16828db3b86279d04b22f429cd6432ba1bc'),
 'bdg2':('bdg2_results/bdg2_manifest.json','b201559475f3b5cf48efbc9833abba4ca146e4a8f6818770cc20b41bbed24689')}

def read_manifest(domain,root=ROOT):
    rel,want=MANIFESTS[domain]; p=root/rel
    if f.sha(p)!=want: raise ValueError('Manifest differs from previously frozen SHA256; stop')
    m=json.loads(p.read_text(encoding='utf-8'))
    expected=v.code_hashes() if domain=='aqua' else b.code_hashes()
    if m['code_sha256']!=expected or m['quick'] or m['seeds']!=[0,1,2,3,4]:
        raise ValueError('Frozen code or experiment mismatch; stop')
    return m,want

def prepare(domain,directory,m,seed):
    if domain=='aqua':
        ponds,clean=f.load_ponds(directory)
        parts,counts,audit=v.partition(ponds)
        if clean!=m['data'] or counts!=m['counts'] or audit!=m['audit'] or m['mode']!='intervals':
            raise ValueError('Aquaculture data or partition changed')
        pairs={n:(pd.concat([p[0],p[1]],ignore_index=True),)*2 for n,p in parts.items()}
        return v.a.prepare_two(pairs,seed,m['cap'],'intervals')
    # Collect exact V6 refit sample. Existing loader also accumulates test moments;
    # they are discarded here, never used for model fitting or score computation.
    samples,_,_=b.collect(directory,m['audit'],[seed],'test',cap=m['cap'],val_cap=m['validation_cap'])
    S,_,_=b.prepared(samples,m['audit']['eligible_for_training'],seed,'test')
    return S

def checkpoint(S,rec,which):
    """Replicate r.fedavg plaintext in identical order and RNG, return weighted inputs."""
    c=rec['global_config']; seed=rec['seed']; names=list(S)
    if len(names)<2: raise ValueError('Need at least two sites')
    sizes={n:len(S[n][0][1]) for n in names}; total=sum(sizes.values())
    weights={n:sizes[n]/total for n in names}; d=S[names[0]][0][0].shape[1]+1
    w=np.zeros(d); rng=np.random.default_rng(seed)
    num=1 if which=='first' else c['rounds']
    for _ in range(num):
        updates={n:weights[n]*r.local_train(*S[n][0],w,c['alpha'],rng,epochs=c['epochs'],lr=c['lr']) for n in names}
        w=np.array(list(updates.values())).sum(0)
    baseline=r.fedavg(S,names,c['alpha'],seed,T=num,epochs=c['epochs'],lr=c['lr'])['w']
    diff=float(np.max(np.abs(w-baseline)))
    if diff>1e-10: raise ValueError(f'Plaintext replay failed ({diff})')
    return updates,weights,diff,num

def run_round(updates,weights,missing=(),timeout=30.):
    survivors=[n for n in updates if n not in missing]
    if len(survivors)<2: raise ValueError('Need >=2 survivors')
    mass=sum(weights[n] for n in survivors)
    start=time.perf_counter(); expected=np.array([updates[n] for n in survivors]).sum(0)
    sum_time=time.perf_counter()-start
    got,receipt,metrics=tls_secure_sum_round(updates,missing=missing,timeout_sec=timeout)
    delta=float(np.max(np.abs(got-expected)))
    bound=len(survivors)/(2*SCALE)+1e-10
    if delta>bound: raise AssertionError(f'Quantization error {delta} exceeds bound {bound}')
    normalized=float(np.max(np.abs(got/mass-expected/mass)))
    # No individual plaintext update or secret is written to disk.
    return {'missing':list(missing),'survivors':survivors,'survivor_weight_mass':mass,
            'max_sum_error':delta,'sum_error_bound':bound,'max_renormalized_model_error':normalized,
            'plaintext_sum_wall_sec':sum_time,'tls_metrics':metrics}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain',choices=MANIFESTS,required=True);p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=range(5),default=0);p.add_argument('--checkpoint',choices=['first','last'],default='first')
    p.add_argument('--dropout',action='store_true',help='Also test one pre-upload dropout: alphabetically first site')
    p.add_argument('--timeout',type=float,default=30.);p.add_argument('--output-dir',type=Path,default=ROOT/'crypto_pilot_results');args=p.parse_args()
    out=args.output_dir/f'{args.domain}_{args.seed}_{args.checkpoint}{"_dropout" if args.dropout else ""}.json'
    if out.exists():p.error('Output exists; preserve previous run')
    m,h=read_manifest(args.domain); rec=next(x for x in m['runs'] if x['seed']==args.seed)
    S=prepare(args.domain,args.dataset_dir,m,args.seed)
    # r.P is a legacy module global, changed only in this bounded scope for BDG2.
    old=r.P;r.P=next(iter(S.values()))[0][0].shape[1]+1
    try: updates,weights,diff,num=checkpoint(S,rec,args.checkpoint)
    finally:r.P=old
    print('Plain replay',args.domain,'round',num,'clients',len(updates),'max diff',diff,flush=True)
    records=[run_round(updates,weights,timeout=args.timeout)]
    if args.dropout:records.append(run_round(updates,weights,missing=[sorted(updates)[0]],timeout=args.timeout))
    result={'protocol':'secfed-v7a-crypto-checkpoint-pilot','domain':args.domain,'seed':args.seed,'checkpoint':num,
            'manifest_sha256':h,'frozen_global_config':rec['global_config'],'dimension':len(next(iter(updates.values()))),
            'plaintext_replay_error':diff,'code_sha256':{str(x.relative_to(ROOT)).replace(os.sep,'/'):f.sha(x) for x in [Path(__file__),*sorted((HERE/'vendor/fl_smpc').glob('*.py'))]},
            'rounds':records,'notes':['Frozen plaintext replay, cryptographic transport only for selected checkpoint; not full secure trajectory.',
            'Controller sees all local data and updates; loopback workers use ephemeral test PKI. Honest noncolluding server and clients assumed.',
            'Dropout compares same precomputed surviving contributions; renormalization by surviving mass is in trusted controller; not retraining after dropout.',
            'Bytes are JSON application frames, both directions, not TLS/TCP wire bytes. Wall latency includes process creation and temporary certificates.',
            'No predictive test scores or hyperparameter reselection; BDG2 loader computes and discards test moments.',
            'Post-upload dropout and malicious-server security not established; no secret-sharing threshold recovery.']}
    f.dump(out,result)
    for item in records:print('TLS',item['missing'],'max error',item['max_sum_error'],'app bytes',item['tls_metrics']['tls_application_frame_bytes'],'wall sec',item['tls_metrics']['network_round_wall_sec'])
    print('Saved:',out)
if __name__=='__main__':main()
