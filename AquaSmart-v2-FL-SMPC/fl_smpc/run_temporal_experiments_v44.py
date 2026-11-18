"""AquaSmart V4.4 dropout and OS-process isolation *validation-only* benchmark.

This deliberately does NOT re-run the expensive hyperparameter search and
NEVER reads the exploratory TEST split. FedAvg settings are frozen from V4.2.
The experiment compares plaintext aggregation and cryptographic recovery for
full participation vs named dropout(s) AFTER key announcement, BEFORE upload.
Each cryptographic round spawns a server and client OS processes on the local
host. Reported IPC bytes exclude vector process-start arguments and TCP/TLS.

Example CMD:
python -m fl_smpc.run_temporal_experiments_v44 --sample-cap 3000 --rounds 2 --seeds 42 --dropouts IoTPond10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
from pathlib import Path

import numpy as np

from fl_smpc.common import gd_update, get_params, split_params
from fl_smpc.run_temporal_experiments_v3 import (
    _cached_data, _zero_model, evaluate_by_pond,
)
from fl_smpc.run_temporal_experiments_v42 import fit_personalized_local
from fl_smpc.process_secagg_v44 import process_secure_sum
from fl_smpc.temporal_protocol import load_temporal_splits

DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'results' / 'temporal_v44'


def _parse_names(raw):
    if raw.strip().lower() in ('', 'none'):
        return ()
    names = tuple(s.strip() for s in raw.split(',') if s.strip())
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError('Duplicate dropout ID')
    return names


def _one_scenario(splits, seed, rounds, missing, *, epochs, lr, alpha,
                  personal_epochs, personal_lr, timeout_sec):
    train = _cached_data(splits)
    ids = list(train)
    if len(ids) - len(missing) < 2:
        raise ValueError('Dropout experiment requires at least 2 remaining clients')
    if not set(missing).issubset(train):
        raise ValueError('Unknown dropout client(s): ' + str(missing))
    total = sum(len(y) for _,y in train.values())
    weights = {k:len(y)/total for k,(_,y) in train.items()}
    present = [k for k in ids if k not in missing]
    present_weight = sum(weights[k] for k in present)
    rng = np.random.default_rng(seed)
    current = _zero_model()
    max_diff=0.0
    measured=[]
    training_seconds=0.0
    for rd in range(1,rounds+1):
        t0=time.perf_counter()
        contributions={}
        for k,(X,y) in train.items():
            w,b=gd_update(X,y,current[0].copy(),current[1],
                          epochs=epochs,lr=lr,alpha=alpha,rng=rng)
            contributions[k]=weights[k]*get_params(w,b)
        training_seconds += time.perf_counter()-t0
        plain=sum((contributions[k] for k in present),start=np.zeros_like(next(iter(contributions.values()))))
        plain /= present_weight
        # The coordinator knows plaintext inputs for this research harness,
        # not the separate spawned aggregator process.
        cryptosum,receipt,wire=process_secure_sum(
            contributions,missing=missing,session=secrets.token_hex(24),
            timeout_sec=timeout_sec)
        secure_model=cryptosum/present_weight
        diff=float(np.max(np.abs(plain-secure_model)))
        max_diff=max(max_diff,diff)
        if diff > 4e-6:
            raise AssertionError(f'Numerical secure-vs-plain divergence: {diff}')
        current=split_params(secure_model)
        measured.append({'round':rd,'n_uploaded':receipt['n_uploaded'],
                         'n_dropped':receipt['n_dropped'],
                         'ipc_payload_bytes':wire['ipc_payload_bytes'],
                         'ipc_messages':wire['ipc_messages'],
                         'wall_sec':wire['round_wall_sec'],
                         'server_verify_sum_sec':wire['server_verify_sum_sec'],
                         'max_parameter_diff':diff,
                         'breakdown':wire['ipc_payload_breakdown']})
    # This is a fixed-personalization evaluation, not tuned on this VAL subset.
    local_models,personal_seconds=fit_personalized_local(
        splits,current,seed,personal_lr,personal_epochs,alpha)
    score=evaluate_by_pond(splits,local_models,'val')
    return {'n_ponds':len(train), 'survivors':present,'dropped':sorted(missing),
            'rounds':rounds,'max_plain_secagg_param_diff':max_diff,
            'validation':score,
            'ipc_protocol_bytes':sum(x['ipc_payload_bytes'] for x in measured),
            'ipc_messages':sum(x['ipc_messages'] for x in measured),
            'process_round_wall_sec':sum(x['wall_sec'] for x in measured),
            'local_train_sec':training_seconds,
            'personalized_training_sec':personal_seconds,
            'rounds_detail':measured,
            'notes':['Validation-only; no TEST accessed',
                     'All clients train, but named client(s) disappear after key announcement, before upload',
                     'Survivor updates reweighted by sum of their pre-round sample fractions',
                     'Spawn-per-round and Pipe JSON only; not production networking',
                     'Experiment parent still has plaintext inputs; aggregator process does not']}


def run_seed(seed, args):
    splits=load_temporal_splits(args.sample_cap,seed,args.dataset_dir)
    print(f'\nSeed {seed}: {len(splits)} ponds; {args.rounds} round(s)')
    if not args.dropouts:
        raise ValueError('For a dropout comparison specify --dropouts <pond-id>')
    results={}
    for label,missing in (('all_online',()),('after_announcement_dropout',args.dropouts)):
        print(f'  Running {label}: missing={list(missing)}')
        results[label]=_one_scenario(
            splits,seed,args.rounds,missing,epochs=args.epochs,lr=args.lr,
            alpha=args.alpha,personal_epochs=args.personal_epochs,
            personal_lr=args.personal_lr,timeout_sec=args.timeout_sec)
        r=results[label]
        print('    n=%d, VAL micro=%.4f macro=%.4f, max Δ=%.2e, IPC=%d bytes, wall=%.2f s' % (
            len(r['survivors']),r['validation']['micro']['rmse'],
            r['validation']['macro']['rmse'],r['max_plain_secagg_param_diff'],
            r['ipc_protocol_bytes'],r['process_round_wall_sec']))
    return {'seed':seed,'scenarios':results,
            'configuration':{'lr':args.lr,'alpha':args.alpha,'local_epochs':args.epochs,
                             'personal_epochs':args.personal_epochs,
                             'personal_lr':args.personal_lr,'sample_cap':args.sample_cap},
            'warnings':['Synthetic feed-ration target, not measured ground truth',
                        'Validation only; do not interpret as independent performance',
                        'Only pre-upload dropout; failed recovery/late dropout abort',
                        'Honest-but-curious server and honest clients, no collusion',
                        'No malicious server, malicious clients, differential privacy, or network benchmark']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-cap',type=int,default=3000)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--seeds',type=str,default='42')
    p.add_argument('--dropouts',type=_parse_names,default=('IoTPond10',))
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--alpha',type=float,default=.1)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--personal-lr',type=float,default=.01)
    p.add_argument('--personal-epochs',type=int,default=3)
    p.add_argument('--dataset-dir',default=None)
    p.add_argument('--timeout-sec',type=float,default=45.0)
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    args=p.parse_args(argv)
    if (args.rounds<1 or args.sample_cap<1 or args.epochs<1 or
        args.personal_epochs<0 or args.timeout_sec<=0 or
        not np.isfinite([args.lr,args.alpha,args.personal_lr]).all() or
        args.lr<=0 or args.alpha<0 or args.personal_lr<=0):
        p.error('Invalid numeric configuration')
    try:
        seeds=[int(s.strip()) for s in args.seeds.split(',')]
        if not seeds or any(x<0 for x in seeds):raise ValueError()
    except ValueError:
        p.error('--seeds must be comma-separated non-negative integers')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    conf={k:str(v) for k,v in vars(args).items() if k!='output_dir'}
    run_id=hashlib.sha256(json.dumps(conf,sort_keys=True).encode()).hexdigest()[:10]
    print(f'AquaSmart V4.4: scenario ID={run_id}; OS-process IPC; validation-only')
    result=[]
    for seed in seeds:
        record=run_seed(seed,args)
        output=args.output_dir/f'seed{seed}_{run_id}.json'
        output.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
        print(f'  Saved: {output}')
        result.append(record)
    summary={'seeds':seeds,'run_id':run_id,
             'scenarios':{name:{'macro_rmse_mean':float(np.mean([r['scenarios'][name]['validation']['macro']['rmse'] for r in result])),
                                'micro_rmse_mean':float(np.mean([r['scenarios'][name]['validation']['micro']['rmse'] for r in result])),
                                'ipc_bytes_mean':float(np.mean([r['scenarios'][name]['ipc_protocol_bytes'] for r in result]))}
                          for name in ('all_online','after_announcement_dropout')},
             'note':'Validation-only research benchmark; NOT a deployed secure-aggregation protocol'}
    path=args.output_dir/f'summary_{run_id}.json'
    path.write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(f'  Summary: {path}')
    return result

if __name__=='__main__':
    main()