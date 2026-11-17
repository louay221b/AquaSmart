"""AquaSmart V4.2 — validation-selected personalized FedAvg (EXPLORATORY).

Run from AquaSmart-v2-FL-SMPC (V2, V3, V4.1 installed):

python -m fl_smpc.run_temporal_experiments_v42 --sample-cap 3000 \
  --rounds 5,10,20 --lr-grid 0.005,0.01 --alpha-grid 0.1,1 \
  --epochs-grid 1,5 --personal-lr-grid 0.001,0.005,0.01 \
  --personal-epochs-grid 0,1,3,5 --seeds 42 --selection macro_rmse

The server model uses FedAvg tuned on VAL; per-pond personalization starts from
this fitted global model, uses only local TRAIN, and selects one common
personalization hyperparameter pair by macro VAL RMSE. Zero personalization
is an included candidate. Test is accessed only after choices are frozen.

Single-process pairwise masking replay is NOT cryptographic SMPC. The feed
label is a synthetic proxy, and test windows were already explored in V2/V3.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from fl_smpc.common import gd_update, get_params
from fl_smpc.data import TARGET_COL
from fl_smpc.temporal_protocol import load_temporal_splits, public_transform
from fl_smpc.run_temporal_experiments_v3 import (
    fit_centralized_checkpoints, fit_local_checkpoints, fit_fed_checkpoints,
    fit_mask_emulation, evaluate_by_pond, metric_for_selection, _to_public_result,
)
from fl_smpc.run_temporal_experiments_v4 import (
    _float_grid, _int_grid, _validate_dataset, tune_fedavg_on_validation, BASELINE,
)

OUT_DIR = Path(__file__).resolve().parent / 'results' / 'temporal_v42'


def _epochs_grid(value: str) -> list[int]:
    try:
        vals = [int(t.strip()) for t in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('personal epochs must be comma-separated integers') from exc
    if not vals or any(x < 0 for x in vals) or len(set(vals)) != len(vals):
        raise argparse.ArgumentTypeError('personal epochs must be unique, nonnegative integers')
    return sorted(vals)


def fit_personalized_local(splits, global_model, seed, learning_rate, epochs, alpha):
    """Locally fine-tune a fixed shared checkpoint using ONLY TRAIN rows."""
    models = {}
    w0, b0 = global_model
    begin = time.perf_counter()
    for i, (name, part) in enumerate(splits.items()):
        if epochs == 0:
            models[name] = (w0.copy(), float(b0))
            continue
        X = public_transform(part.train)
        y = part.train[TARGET_COL].to_numpy(dtype=np.float64)
        rng = np.random.default_rng(seed * 104729 + i)
        w, b = gd_update(X, y, w0.copy(), float(b0), epochs=epochs,
                         lr=learning_rate, alpha=alpha, rng=rng)
        models[name] = (w, b)
    return models, time.perf_counter() - begin


def tune_personalization_on_validation(splits, global_model, seed, lr_grid,
                                       epochs_grid, alpha, selection):
    """Never read TEST. Use one shared setting across ponds to limit overfit."""
    histories = []
    best_key = None
    best_settings = None
    start = time.perf_counter()
    for epochs, lr in itertools.product(epochs_grid, lr_grid):
        models, _ = fit_personalized_local(splits, global_model, seed, lr, epochs, alpha)
        score = evaluate_by_pond(splits, models, 'val')
        selected_value = float(metric_for_selection(score, selection))
        if not math.isfinite(selected_value):
            raise ValueError('Non-finite validation error in personalization grid')
        rec = {
            'personal_epochs': epochs,
            'personal_lr': lr,
            'validation_selection_rmse': selected_value,
            'validation_micro_rmse': score['micro']['rmse'],
            'validation_macro_rmse': score['macro']['rmse'],
        }
        histories.append(rec)
        key = (selected_value, epochs, lr)  # prefer no/fewer steps when tied
        if best_key is None or key < best_key:
            best_key, best_settings = key, {'personal_epochs': epochs, 'personal_lr': lr}
    if best_settings is None:
        raise ValueError('No personalization candidates')
    search_time = time.perf_counter() - start
    chosen_models, local_time = fit_personalized_local(
        splits, global_model, seed, best_settings['personal_lr'],
        best_settings['personal_epochs'], alpha)
    chosen_validation = evaluate_by_pond(splits, chosen_models, 'val')
    histories.sort(key=lambda r: (r['validation_selection_rmse'],
                                  r['personal_epochs'], r['personal_lr']))
    return chosen_models, best_settings, chosen_validation, histories, search_time, local_time


def _personal_fit(global_fit, models, config, validation, tuning_history,
                  search_sec, local_sec):
    return {
        'models': models,
        'selected_round': global_fit['selected_round'],
        'validation': validation,
        'selected_personalization': config,
        'validation_grid': tuning_history,
        'personalization_grid_time_sec': search_sec,
        'personalization_fit_sec': local_sec,
        'time_sec': global_fit['time_sec'] + local_sec,
        'comm_bytes': global_fit['comm_bytes'],
        'n_messages': global_fit['n_messages'],
        'personalization_additional_comm_bytes': 0,
        'note': 'Post-training per-client fine-tuning only; no additional client-server exchanges',
    }


def run_seed(seed, args):
    splits = load_temporal_splits(args.sample_cap, seed, args.dataset_dir)
    _validate_dataset(splits)
    print(f'\nSeed {seed}: {len(splits)} ponds (TRAIN cap {args.sample_cap})')
    for name, p in splits.items():
        s = p.summary
        flag = ' [short VAL horizon]' if s['val_days'] < 3 or s['val_rows'] < 30 else ''
        print(f'  {name:<11} days {s["train_days"]:>3}/{s["val_days"]:>3}/{s["test_days"]:>3}'
              f' | rows {s["train_rows"]:>5}/{s["val_rows"]:>6}/{s["test_rows"]:>6}{flag}')
    print('  [1/7] Centralized / Local / FedAvg baseline VAL selection...')
    central = fit_centralized_checkpoints(splits,args.rounds,BASELINE['local_epochs'],seed,
                                          BASELINE['lr'],BASELINE['alpha'],args.selection)
    local = fit_local_checkpoints(splits,args.rounds,BASELINE['local_epochs'],seed,
                                  BASELINE['lr'],BASELINE['alpha'],args.selection)
    fed_base = fit_fed_checkpoints(splits,args.rounds,BASELINE['local_epochs'],seed,
                                   BASELINE['lr'],BASELINE['alpha'],args.selection)
    print('  [2/7] FedAvg grid search on VAL only...')
    tuned,cfg,fed_grid,fed_search_sec = tune_fedavg_on_validation(
        splits,args.rounds,seed,args.lr_grid,args.alpha_grid,args.epochs_grid,args.selection)
    print(f'         FedAvg: lr={cfg["lr"]} alpha={cfg["alpha"]} '
          f'epochs={cfg["local_epochs"]} rounds={tuned["selected_round"]}')
    print('  [3/7] Personalization grid search on VAL only...')
    models, pcfg, pval, pgrid, psearch, pfit = tune_personalization_on_validation(
        splits,tuned['models'],seed,args.personal_lr_grid,args.personal_epochs_grid,
        cfg['alpha'],args.selection)
    print(f'         Personalization: epochs={pcfg["personal_epochs"]} '
          f'lr={pcfg["personal_lr"]}; VAL {args.selection}='
          f'{metric_for_selection(pval,args.selection):.4f}')
    pers = _personal_fit(tuned,models,pcfg,pval,pgrid,psearch,pfit)
    print('  [4/7] Replay pairwise additive masking on global FedAvg...')
    masked = fit_mask_emulation(splits,tuned['selected_round'],cfg['local_epochs'],
                                seed,cfg['lr'],cfg['alpha'])
    global_diff = float(np.max(np.abs(get_params(*tuned['models']) - get_params(*masked['models']))))
    if global_diff > 1e-8:
        raise AssertionError(f'Masked FedAvg differs numerically: {global_diff}')
    print('  [5/7] Apply same locally selected personalization to masked global...')
    masked_models,masked_pfit = fit_personalized_local(splits,masked['models'],seed,
                        pcfg['personal_lr'],pcfg['personal_epochs'],cfg['alpha'])
    personal_diff = max(float(np.max(np.abs(get_params(*models[k])-get_params(*masked_models[k]))))
                        for k in splits)
    if personal_diff > 1e-8:
        raise AssertionError(f'Masked personalized models differ numerically: {personal_diff}')
    masked_pers = _personal_fit(masked,masked_models,pcfg,pval,[],0,masked_pfit)
    masked_pers['validation'] = pval
    masked_pers['selection_source'] = 'plain_tuned_fedavg_and_personalization_validation_only'
    masked_pers['max_personalized_parameter_difference'] = personal_diff
    masked_pers['max_global_parameter_difference'] = global_diff
    print('  [6/7] Freeze all hyperparameters / checkpoints before test...')
    tuned['selected_hyperparameters'] = cfg
    tuned['validation_grid'] = fed_grid
    tuned['grid_search_time_sec'] = fed_search_sec
    masked['validation'] = tuned['validation']
    masked['selected_hyperparameters'] = cfg
    masked['max_global_parameter_difference'] = global_diff
    fits = {
        'centralized':central,
        'local':local,
        'federated_baseline':fed_base,
        'federated_tuned':tuned,
        'personalized_fedavg':pers,
        'mask_emulation_tuned':masked,
        'mask_emulation_personalized':masked_pers,
    }
    print('  [7/7] Final exploratory TEST scoring (identical test partitions)...')
    methods = {}
    for name, fit in fits.items():
        result = _to_public_result(fit)
        result['metrics'] = evaluate_by_pond(splits,fit['models'],'test')
        methods[name] = result
        print(f'  {name:<29} VAL {metric_for_selection(fit["validation"],args.selection):.4f}'
              f'   TEST micro {result["metrics"]["micro"]["rmse"]:.4f}'
              f'   macro {result["metrics"]["macro"]["rmse"]:.4f}')
    meta = {
        'protocol_version':'v4.2',
        'seed':seed,
        'selection_metric':args.selection,
        'n_clients':len(splits),
        'n_val_rows':sum(len(p.val) for p in splits.values()),
        'n_test_rows':sum(len(p.test) for p in splits.values()),
        'split':'70/15/15 chronological local days per pond; TRAIN cap only',
        'ponds':{k:p.summary for k,p in splits.items()},
        'fed_round_grid':args.rounds,
        'fed_lr_grid':args.lr_grid,
        'fed_alpha_grid':args.alpha_grid,
        'fed_local_epochs_grid':args.epochs_grid,
        'selected_fed_hyperparameters':cfg,
        'selected_fed_rounds':tuned['selected_round'],
        'personal_lr_grid':args.personal_lr_grid,
        'personal_epochs_grid':args.personal_epochs_grid,
        'selected_personalization':pcfg,
        'test_policy':'exploratory test windows previously analyzed in V2/V3/V4.1',
        'label':'proxy simulated feed_ration_g, NOT measured feeding',
        'security':'single-process additive masking emulation; NOT cryptographic SMPC',
        'privacy':'plain personalization uses only private local TRAIN; no extra updates sent',
        'communication':'analytical estimates, excluding real key exchange/network latency',
        'training_budget':'methods have different training budgets, do not claim compute fairness',
        'tiny_validation_ponds':[k for k,p in splits.items() if p.summary['val_days']<3 or len(p.val)<30],
    }
    return {'_meta':meta,'methods':methods}


def summarize(results):
    methods = {}
    for key in results[0]['methods']:
        micro = [r['methods'][key]['metrics']['micro']['rmse'] for r in results]
        macro = [r['methods'][key]['metrics']['macro']['rmse'] for r in results]
        val = [metric_for_selection(r['methods'][key]['validation'],r['_meta']['selection_metric'])
               for r in results]
        methods[key] = {
            'micro_rmse_mean':float(np.mean(micro)),
            'micro_rmse_std_sample':float(np.std(micro,ddof=1)) if len(micro)>1 else None,
            'macro_rmse_mean':float(np.mean(macro)),
            'macro_rmse_std_sample':float(np.std(macro,ddof=1)) if len(macro)>1 else None,
            'selected_validation_rmse_mean':float(np.mean(val)),
        }
    return {
        'protocol_version':'v4.2',
        'seeds':[r['_meta']['seed'] for r in results],
        'n_seeds':len(results),
        'note':'Same already-explored temporal test windows across seeds. This is an exploratory stability assessment, not independent generalization.',
        'methods':methods,
        'selected_fed_configurations':[r['_meta']['selected_fed_hyperparameters'] for r in results],
        'selected_personalization_configurations':[r['_meta']['selected_personalization'] for r in results],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample-cap',type=int,default=3000)
    ap.add_argument('--rounds',type=lambda s:_int_grid(s,'rounds'),default=[5,10,20])
    ap.add_argument('--lr-grid',type=lambda s:_float_grid(s,'lr-grid',False),default=[0.005,0.01])
    ap.add_argument('--alpha-grid',type=lambda s:_float_grid(s,'alpha-grid',True),default=[0.1,1.0])
    ap.add_argument('--epochs-grid',type=lambda s:_int_grid(s,'epochs-grid'),default=[1,5])
    ap.add_argument('--personal-lr-grid',type=lambda s:_float_grid(s,'personal-lr-grid',False),
                    default=[0.001,0.005,0.01])
    ap.add_argument('--personal-epochs-grid',type=_epochs_grid,default=[0,1,3,5])
    ap.add_argument('--seeds',type=lambda s:_int_grid(s,'seeds'),default=[42])
    ap.add_argument('--selection',choices=['macro_rmse','micro_rmse'],default='macro_rmse')
    ap.add_argument('--dataset-dir',default=None)
    ap.add_argument('--output-dir',type=Path,default=OUT_DIR)
    args = ap.parse_args(argv)
    if args.sample_cap<=0:
        ap.error('--sample-cap must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    run_key = {
        'sample_cap':args.sample_cap,'rounds':args.rounds,'seeds':args.seeds,
        'lr_grid':args.lr_grid,'alpha_grid':args.alpha_grid,'epochs_grid':args.epochs_grid,
        'personal_lr_grid':args.personal_lr_grid,'personal_epochs_grid':args.personal_epochs_grid,
        'selection':args.selection,'dataset_dir':args.dataset_dir,
    }
    run_id = hashlib.sha256(json.dumps(run_key,sort_keys=True).encode()).hexdigest()[:10]
    print(f'V4.2 run ID: {run_id}')
    records=[]
    for seed in args.seeds:
        result = run_seed(seed,args)
        path = args.output_dir / f'seed{seed}_cap{args.sample_cap}_{args.selection}_{run_id}.json'
        path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        records.append(result)
        print(f'  Saved: {path}')
    output = summarize(records)
    summary_path = args.output_dir / f'summary_{args.selection}_{run_id}.json'
    summary_path.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print('\nMean exploratory TEST RMSE:')
    for k,v in output['methods'].items():
        print(f'  {k:<29} micro={v["micro_rmse_mean"]:.4f} macro={v["macro_rmse_mean"]:.4f}')
    print(f'Saved: {summary_path}')
    print('WARNING: simulated feeding labels; masking is not cryptographic SMPC; same test windows already inspected.')
    return records


if __name__=='__main__':
    main()
