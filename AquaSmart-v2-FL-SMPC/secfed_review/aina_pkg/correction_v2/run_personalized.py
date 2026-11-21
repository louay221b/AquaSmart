"""SecFed correction phase 2: exploratory validation, never score test.
Run from aina_pkg. Requires correction_v1 and original sims/rcore.py.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'correction_v1'))
from protocol import load_ponds, prepare
from run_corrected import run as phase1, r


def metrics(models, S):
    names = list(S)
    ys = np.concatenate([S[n][1][1] for n in names])
    errors = {n: r.pred(models[n], S[n][1][0]) - S[n][1][1] for n in names}
    e = np.concatenate(list(errors.values())); ss = float(np.sum((ys - ys.mean())**2))
    per = {n: float(np.sqrt(np.mean(v**2))) for n, v in errors.items()}
    return {'micro': float(np.sqrt(np.mean(e**2))), 'macro': float(np.mean(list(per.values()))),
            'r2': float(1-np.sum(e**2)/ss) if ss>0 else None, 'per_pond': per}


def shared(w, S):
    return dict.fromkeys(S, w)


def adapt(w, S, alpha, seed, epochs, lr):
    # Separate deterministic RNG per pond; reset for every candidate and masked/plain pairing.
    if epochs == 0:
        return {n: w.copy() for n in S}
    return {n: r.local_train(*S[n][0], w, alpha,
            np.random.default_rng(np.random.SeedSequence([seed, i, 913])), epochs=epochs, lr=lr)
            for i,n in enumerate(S)}


def experiment(S, seed, rounds_grid=(20,40), lr_grid=(.01,.05,.1), epochs_grid=(1,5),
               alpha_grid=tuple(r.ALPHAS), personal_epochs=(0,1,3,5,10,20), personal_lrs=(.005,.01,.05),
               selection='micro'):
    # Strip test partition defensively. Everything below sees train/validation only.
    S = {n: (parts[0], parts[1]) for n,parts in S.items()}
    names = list(S)
    baseline = phase1(S, seed, 20)
    best = None; trials = []
    for a in alpha_grid:
        for lr in lr_grid:
            for ep in epochs_grid:
                # One trajectory supplies both checkpoints, avoiding redundant training.
                f = r.fedavg(S,names,a,seed,T=max(rounds_grid),lr=lr,epochs=ep,track=True)
                for T in rounds_grid:
                    w = f['hist'][T-1]; m = metrics(shared(w,S),S)
                    c = {'alpha':float(a),'lr':float(lr),'epochs':int(ep),'rounds':int(T)}
                    trials.append({'config':c,'micro':m['micro'],'macro':m['macro']})
                    key = (m[selection], T*ep, T, ep, lr, a)
                    if best is None or key < best[0]: best = (key,c,w.copy(),m)
    _,c,w,global_m = best
    sf = r.fedavg(S,names,c['alpha'],seed,T=c['rounds'],lr=c['lr'],epochs=c['epochs'],mode='masked')
    secure_w = sf['w']
    pbest = None; pgrid = []
    for ep in personal_epochs:
        for lr in (personal_lrs[:1] if ep==0 else personal_lrs):
            models = adapt(w,S,c['alpha'],seed,ep,lr); m = metrics(models,S)
            pc = {'epochs':int(ep),'lr':float(lr),'alpha':c['alpha']}
            pgrid.append({'config':pc,'micro':m['micro'],'macro':m['macro']})
            key = (m[selection],ep,lr)
            if pbest is None or key < pbest[0]: pbest=(key,pc,models,m)
    _,pc,models,pm = pbest
    sm = adapt(secure_w,S,c['alpha'],seed,pc['epochs'],pc['lr'])
    # Same local optimizer/epochs/lr starting at zero, isolates the initialization
    # effect at the selected adaptation budget (not a separately tuned optimum).
    scratch = adapt(np.zeros_like(w),S,c['alpha'],seed,pc['epochs'],pc['lr'])
    diff = max(float(np.max(np.abs(models[n]-sm[n]))) for n in names)
    if diff > 1e-8 or np.max(np.abs(w-secure_w)) > 1e-8:
        raise AssertionError('Unexpected numerical divergence of Gaussian masks')
    validation = dict(baseline['validation'])
    validation.update({'fedavg_tuned':global_m, 'secfed_tuned_gaussian':metrics(shared(secure_w,S),S),
                      'personalized_fedavg':pm,'personalized_secfed_gaussian':metrics(sm,S),
                      'local_sgd_matched_adaptation':metrics(scratch,S)})
    return {'seed':seed,'validation':validation,'baseline_alphas':baseline['alphas'],
            'global_config':c,'personal_config':pc,'selection':selection,'global_trials':trials,
            'personal_trials':pgrid,'global_max_param_diff':float(np.max(np.abs(w-secure_w))),
            'personal_max_param_diff':diff,'max_round_mask_error':sf['maxerr'],
            'cost_model_only':{'secfed_bytes':sf['bytes'],'secfed_messages':sf['msgs'],
                               'fedavg_bytes':16*len(names)*len(w)*c['rounds'],
                               'fedavg_messages':2*len(names)*c['rounds'],
                               'personalization_extra_communication_bytes':0}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--mode',choices=['rows','intervals'],required=True)
    p.add_argument('--seeds',default='0,1,2,3,4'); p.add_argument('--cap',type=int,default=3000)
    p.add_argument('--selection',choices=['micro','macro'],default='micro')
    p.add_argument('--output-dir',type=Path,default=Path('personalized_results'))
    p.add_argument('--quick',action='store_true',help='Smoke test with reduced search; not the main campaign')
    args=p.parse_args(); seeds=[int(v) for v in args.seeds.split(',')]
    if args.cap<1 or not seeds or min(seeds)<0: p.error('invalid cap or seed')
    ponds,clean=load_ponds(args.dataset_dir); args.output_dir.mkdir(parents=True,exist_ok=True)
    result={'protocol':'blocked-weight-runs-v1-personalized-v2','mode':args.mode,'cap':args.cap,'seeds':seeds,
            'selection':args.selection,'quick':args.quick,'cleaning':clean,'runs':[],
            'notes':['Exploratory VAL only; tuning and evaluation same VAL, not independent',
                     'Same data partitions and baseline as phase 1; pooled scaler not private',
                     'Gaussian masks simulation, not cryptographic SecAgg',
                     'Common personalization hyperparameters; local Ridge uses pond-specific alpha',
                     'Matched local SGD controls initialization only, not total optimization cost']}
    out=args.output_dir/f'{args.mode}_{args.selection}{"_quick" if args.quick else ""}.json'
    opts=dict(rounds_grid=(20,),lr_grid=(.05,),epochs_grid=(5,),alpha_grid=(.1,),personal_epochs=(0,3),personal_lrs=(.01,)) if args.quick else {}
    for seed in seeds:
        S,audit,scaler=prepare(ponds,seed,args.cap,args.mode)
        rec=experiment(S,seed,selection=args.selection,**opts); rec.update(split_audit=audit,scaler=scaler)
        result['runs'].append(rec)
        print(f'Seed {seed}:', {k:round(v['micro'],4) for k,v in rec['validation'].items()},flush=True)
        print('  Global:',rec['global_config'],'Personal:',rec['personal_config'],'Max diff:',rec['personal_max_param_diff'],flush=True)
        out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved:',out)

if __name__=='__main__': main()
