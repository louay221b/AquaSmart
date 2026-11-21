"""V7B: full recursive TLS-aggregated training, frozen retrospective comparison.
Trusted controller holds data and computes updates; not a private distributed training system.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import argparse, contextlib, json, threading, time
from pathlib import Path
import numpy as np
import pandas as pd
import run_crypto_pilot as p
r=p.r; f=p.f; b=p.b; v=p.v; ROOT=p.ROOT
REFERENCE={
 'aqua':('synchronized_results/intervals_test.json','4b2bd2fedec3a68a3313ba2c79f04d8bee6e8453d52e4f082346cfa54e9e330d'),
 'bdg2':('bdg2_results/bdg2_test.json','0a8475274b46ebaa61bb96ec76964cc8534782bceed9faad07c8cb5760a66964')}

@contextlib.contextmanager
def rss_sampler():
    """Sample summed process-tree RSS; shared pages counted multiple times, not true RAM.
    Includes parent, children and any retained data; not isolated cryptographic allocations.
    """
    stats={'available':False,'interval_sec':.05,'baseline_bytes':None,'peak_bytes':None,'samples':0}
    try: import psutil
    except ImportError:
        yield stats; return
    proc=psutil.Process(); stop=threading.Event()
    def measure():
        try: procs=[proc]+proc.children(recursive=True)
        except psutil.Error: return
        total=0
        for child in procs:
            try: total+=child.memory_info().rss
            except psutil.Error: pass
        stats['peak_bytes']=max(stats['peak_bytes'] or 0,total);stats['samples']+=1
        return total
    stats['available']=True;stats['baseline_bytes']=measure()
    def sample():
        while not stop.wait(stats['interval_sec']): measure()
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try: yield stats
    finally: stop.set();thread.join();measure()

def prepare(domain,directory,m,seed):
    if domain=='aqua':
        ponds,clean=f.load_ponds(directory);parts,counts,audit=v.partition(ponds)
        if clean!=m['data'] or counts!=m['counts'] or audit!=m['audit'] or m['mode']!='intervals': raise ValueError('Aquaculture data or partition changed')
        S=v.a.prepare_two({n:(pd.concat([q[0],q[1]],ignore_index=True),q[2]) for n,q in parts.items()},seed,m['cap'],'intervals')
        def score(models): return v.a.metrics(models,S)
    else:
        samples,moments,_=b.collect(directory,m['audit'],[seed],'test',cap=m['cap'],val_cap=m['validation_cap'])
        S,mu,sd=b.prepared(samples,m['audit']['eligible_for_training'],seed,'test')
        def score(models): return b.moment_scores(models,moments,mu,sd)
    return {n:(q[0],q[0]) for n,q in S.items()},score

def trajectory(S,rec,rounds,secure=False,missing=(),timeout=30.,progress=False):
    names=list(S); active=[n for n in names if n not in missing]
    if len(active)<2 or not set(missing)<=set(names): raise ValueError('Invalid dropout cohort')
    sizes={n:len(S[n][0][1]) for n in names}; total=sum(sizes.values())
    if any(not x for x in sizes.values()):raise ValueError('Empty client')
    weights={n:sizes[n]/total for n in names}; mass=sum(weights[n] for n in active)
    c=rec['global_config'];rng=np.random.default_rng(rec['seed']);w=np.zeros(S[names[0]][0][0].shape[1]+1); history=[];records=[]
    start=time.perf_counter()
    with rss_sampler() as memory:
        for t in range(rounds):
            ts=time.perf_counter();updates={n:weights[n]*r.local_train(*S[n][0],w,c['alpha'],rng,epochs=c['epochs'],lr=c['lr']) for n in names};local=time.perf_counter()-ts
            ts=time.perf_counter();expected=np.array([updates[n] for n in active]).sum(0);sumtime=time.perf_counter()-ts
            if not np.isfinite(expected).all():raise ValueError('Nonfinite update')
            row={'round':t+1,'local_train_wall_sec':local,'plaintext_reference_sum_wall_sec':sumtime}
            if secure:
                ts=time.perf_counter();got,_,met=p.tls_secure_sum_round(updates,missing=missing,timeout_sec=timeout);latency=time.perf_counter()-ts
                err=float(np.max(np.abs(got-expected)));bound=len(active)/(2*p.SCALE)+1e-10
                if not np.isfinite(got).all() or err>bound:raise AssertionError(f'Round {t+1}: error {err} > {bound}')
                row.update(max_sum_error=err,sum_error_bound=bound,max_renormalized_step_error=err/mass,tls_metrics=met,tls_call_wall_sec=latency)
                w=got/mass if missing else got
            else: w=expected/mass if missing else expected
            history.append(w.copy());records.append(row)
            if progress and secure:print(f'  TLS round {t+1}/{rounds}: sum error={err:.3g}, bytes={met["tls_application_frame_bytes"]}, call={latency:.3f}s',flush=True)
    return w,history,records,{'training_wall_sec':time.perf_counter()-start,'memory_rss':memory,'survivor_weight_mass':mass,'survivors':active}

def reference(domain,mhash,seed):
    rel,h=REFERENCE[domain];path=ROOT/rel
    if f.sha(path)!=h:raise ValueError('Prior test file differs from recorded SHA256')
    obj=json.loads(path.read_text(encoding='utf-8'))
    if obj['manifest_sha256']!=mhash:raise ValueError('Prior test/manifest mismatch')
    return next(q['test'] for q in obj['runs'] if q['seed']==seed)

def metric_diff(x,y):
    diffs=[]
    for k in ('micro','macro','r2'):
        if x[k] is None or y[k] is None:
            if x[k]!=y[k]:raise ValueError('Incompatible metric')
        else: diffs.append(abs(x[k]-y[k]))
    for key in ('per_pond','per_site'):
        if key in x:
            if set(x[key])!=set(y[key]):raise ValueError('Different site sets')
            diffs.extend(abs(x[key][n]-y[key][n]) for n in x[key])
    return max(diffs,default=0.)

def experiment(S,score,rec,rounds,missing=(),timeout=30.,full=True,ref=None,progress=False):
    start=time.perf_counter();plain,ph,pr,pm=trajectory(S,rec,rounds,missing=missing);old=r.P;r.P=len(plain)
    try:
        baseline_diff=None
        if not missing:
            c=rec['global_config'];w=r.fedavg(S,list(S),c['alpha'],rec['seed'],T=rounds,epochs=c['epochs'],lr=c['lr'])['w'];baseline_diff=float(np.max(np.abs(w-plain)))
            if baseline_diff>1e-10:raise AssertionError('Plain trajectory mismatch')
        pc=rec['sgd_config']; personalize=lambda w:f.adapt(w,S,pc['alpha'],rec['seed'],pc['epochs'],pc['lr'])
        pp=personalize(plain);plain_scores=None
        if full:
            plain_scores={'global':score(dict.fromkeys(S,plain)),'personalized':score(pp)}
            if ref is not None:
                ref_error=max(metric_diff(plain_scores['global'],ref['fedavg']),metric_diff(plain_scores['personalized'],ref['personalized_sgd']))
                if ref_error>1e-7:raise AssertionError(f'Frozen score reproduction failed: {ref_error}')
            else:ref_error=None
        else:ref_error=None
        enc,eh,er,em=trajectory(S,rec,rounds,secure=True,missing=missing,timeout=timeout,progress=progress)
        sp=personalize(enc)
        scores=None
        if full:
            secure_scores={'global':score(dict.fromkeys(S,enc)),'personalized':score(sp)}
            scores={'plaintext':plain_scores,'tls':secure_scores,'max_score_difference':{k:metric_diff(plain_scores[k],secure_scores[k]) for k in plain_scores}}
        for q,x,y in zip(er,ph,eh):q['max_global_trajectory_difference']=float(np.max(np.abs(x-y)))
        bytes_=[q['tls_metrics']['tls_application_frame_bytes'] for q in er];times=[q['tls_call_wall_sec'] for q in er]
        return {'rounds_completed':rounds,'full_frozen_rounds':full,'missing_each_round':list(missing),'plain_replay_error':baseline_diff,'frozen_score_reproduction_error':ref_error,
                'final_global_parameter_diff':float(np.max(np.abs(enc-plain))),'max_personal_parameter_diff':max(float(np.max(np.abs(pp[n]-sp[n]))) for n in S),
                'max_trajectory_diff':max(q['max_global_trajectory_difference'] for q in er),'max_sum_error':max(q['max_sum_error'] for q in er),
                'scores':scores,'plain_training':pm,'tls_training':em,'plaintext_rounds':pr,'tls_rounds':er,'application_frame_bytes_total':sum(bytes_),
                'application_frame_bytes_mean':float(np.mean(bytes_)),'tls_call_seconds_mean':float(np.mean(times)),'tls_call_seconds_p95':float(np.percentile(times,95)),
                'ideal_plain_model_payload_bytes':int(rounds*(len(S)+len(pm['survivors']))*len(plain)*8),'comparison_wall_sec':time.perf_counter()-start}
    finally:r.P=old

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--domain',choices=p.MANIFESTS,required=True);ap.add_argument('--dataset-dir',type=Path,required=True);ap.add_argument('--seeds',default='0');ap.add_argument('--smoke',action='store_true');ap.add_argument('--dropout',action='store_true',help='Same first alphabetic client absent before upload each round; all clients still compute locally');ap.add_argument('--accept-retrospective-test',action='store_true');ap.add_argument('--timeout',type=float,default=30.);ap.add_argument('--output-dir',type=Path,default=ROOT/'crypto_full_results');args=ap.parse_args()
    seeds=[int(s) for s in args.seeds.split(',')]
    if not seeds or len(set(seeds))!=len(seeds) or any(s not in range(5) for s in seeds):ap.error('Unique seeds 0..4 required')
    if not args.smoke and not args.accept_retrospective_test:ap.error('Full evaluation requires --accept-retrospective-test')
    m,h=p.read_manifest(args.domain)
    for seed in seeds:
        label=('smoke' if args.smoke else 'full')+('_dropout' if args.dropout else '_all');out=args.output_dir/f'{args.domain}_{seed}_{label}.json'
        if out.exists():ap.error(f'Output exists: {out}; preserve it, or use unused output-dir')
        rec=next(q for q in m['runs'] if q['seed']==seed);S,score=prepare(args.domain,args.dataset_dir,m,seed);missing=[sorted(S)[0]] if args.dropout else []
        rounds=min(2,rec['global_config']['rounds']) if args.smoke else rec['global_config']['rounds']
        ref=reference(args.domain,h,seed) if not args.smoke and not missing else None
        print('Training',args.domain,seed,rounds,'rounds',missing,flush=True)
        res=experiment(S,score,rec,rounds,missing,args.timeout,not args.smoke,ref,True)
        obj={'protocol':'secfed-v7b-recursive-tls-v1','domain':args.domain,'seed':seed,'manifest_sha256':h,'global_config':rec['global_config'],'sgd_config':rec['sgd_config'],'dimension':next(iter(S.values()))[0][0].shape[1]+1,'clients':list(S),'code_sha256':{str(x.relative_to(ROOT)).replace(os.sep,'/'):f.sha(x) for x in [Path(__file__),Path(p.__file__),*sorted((p.HERE/'vendor/fl_smpc').glob('*.py'))]},**res,'notes':['Trusted controller computes all local updates and has plaintext; loopback, ephemeral PKI and processes per round; no threshold secret sharing, no malicious-server security.','Recursive TLS aggregate is used as next global model; paired plaintext reference uses same RNG permutations. Fixed pre-upload dropout scenario retains local computations and evaluates all sites including missing site; personalization remains local in conceptual model but controller performs it.','Application frames exclude TLS handshake and TCP/IP overhead, model distribution to training processes and centralized preprocessing. Plain payload is theoretical, not a measured network baseline.','Training time includes RSS sampling and startup/cleanup; sampled sum of process RSS counts shared pages repeatedly and may miss peaks; excludes data loading. Not isolated crypto cost.','No new hyperparameter search. Previously examined retrospective test; smoke omits scoring.']}
        f.dump(out,obj);print('Final parameter diff',res['final_global_parameter_diff'],'personal diff',res['max_personal_parameter_diff'],'bytes',res['application_frame_bytes_total'],flush=True)
        if res['scores']:print('Scores', {k:{n:{m:x[m] for m in ('micro','macro','r2')} for n,x in q.items()} for k,q in res['scores'].items() if k!='max_score_difference'},flush=True)
        print('Saved:',out,flush=True)
if __name__=='__main__':main()
