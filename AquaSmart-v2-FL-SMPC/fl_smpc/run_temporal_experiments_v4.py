"""AquaSmart V4.1: FedAvg hyperparameter selection by VALIDATION ONLY.

From project root (with V2 temporal_protocol.py and V3 runner installed):

    python -m fl_smpc.run_temporal_experiments_v4 \
        --sample-cap 3000 --rounds 5,10,20,50 \
        --lr-grid 0.005,0.01,0.02 --alpha-grid 0,0.1,1 \
        --epochs-grid 1,5 --seeds 42 --selection macro_rmse

V4.1 retains the V3 Centralized/Local/FedAvg baselines and adds an optimized
FedAvg candidate and a matching simulated pairwise-masking replay. It does
NOT implement cryptographic SMPC or train on measured feeding labels.

CRUCIAL: benchmark test windows were seen during V2/V3 exploration. These
V4.1 test scores are diagnostic/exploratory, NOT an untouched final test.
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

from fl_smpc.temporal_protocol import load_temporal_splits
from fl_smpc.run_temporal_experiments_v3 import (
    evaluate_by_pond,
    fit_centralized_checkpoints,
    fit_local_checkpoints,
    fit_fed_checkpoints,
    fit_mask_emulation,
    metric_for_selection,
    _to_public_result,
)
from fl_smpc.common import get_params

OUT_DIR = Path(__file__).resolve().parent / 'results' / 'temporal_v4'
BASELINE = {'lr': 0.01, 'alpha': 1.0, 'local_epochs': 5}


def _float_grid(s: str, label: str, allow_zero: bool) -> list[float]:
    try:
        values = [float(t.strip()) for t in s.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f'{label}: comma-separated numbers required') from e
    if not values or any(not math.isfinite(x) or x < 0 or (x == 0 and not allow_zero) for x in values):
        raise argparse.ArgumentTypeError(f'{label}: finite values, ' + ('>= 0' if allow_zero else '> 0'))
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f'{label}: duplicate values')
    return sorted(values)


def _int_grid(s: str, label: str) -> list[int]:
    try:
        values = [int(t.strip()) for t in s.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f'{label}: comma-separated positive integers required') from e
    if not values or any(x <= 0 for x in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f'{label}: positive unique integers required')
    return sorted(values)


def tune_fedavg_on_validation(splits, rounds, seed, lr_grid, alpha_grid,
                              epochs_grid, selection):
    """Fit grid candidates without ever accessing TEST; select on VAL only.

    Returns the winning fitted model, chosen configuration, and all candidate
    validation diagnostics. Ties prefer fewer communication rounds, then
    fewer local epochs, then lexicographic config order (deterministic).
    """
    candidate_records = []
    best_key = None
    best_fit = None
    best_config = None
    start = time.perf_counter()
    for lr, alpha, epochs in itertools.product(lr_grid, alpha_grid, epochs_grid):
        fit = fit_fed_checkpoints(splits, rounds, epochs, seed, lr, alpha, selection)
        val_rmse = float(metric_for_selection(fit['validation'], selection))
        if not math.isfinite(val_rmse):
            raise ValueError('Non-finite validation loss: check training stability')
        config = {'lr': lr, 'alpha': alpha, 'local_epochs': epochs}
        candidate_records.append({
            **config,
            'selected_round': fit['selected_round'],
            'selection_validation_rmse': val_rmse,
            'validation_micro_rmse': fit['validation']['micro']['rmse'],
            'validation_macro_rmse': fit['validation']['macro']['rmse'],
            'validation_history': [
                {'round': item['round'],
                 'micro_rmse': item['validation']['micro']['rmse'],
                 'macro_rmse': item['validation']['macro']['rmse']}
                for item in fit['validation_history']
            ],
        })
        key = (val_rmse, fit['selected_round'], epochs, lr, alpha)
        if best_key is None or key < best_key:
            best_key = key
            best_fit = fit
            best_config = config
    if best_fit is None:
        raise ValueError('No candidates supplied')
    grid_seconds = time.perf_counter() - start
    candidate_records.sort(key=lambda item: (
        item['selection_validation_rmse'], item['selected_round'],
        item['local_epochs'], item['lr'], item['alpha']))
    return best_fit, best_config, candidate_records, grid_seconds


def _validate_dataset(splits):
    if len(splits) < 2:
        raise ValueError('Expected at least two independent FL clients')
    for name, p in splits.items():
        if min(len(p.train), len(p.val), len(p.test)) < 1:
            raise ValueError(f'{name}: empty train/val/test subset')


def run_seed(seed, args):
    # All methods see the same partitions for this seed.
    splits = load_temporal_splits(args.sample_cap, seed, args.dataset_dir)
    _validate_dataset(splits)
    print(f'\nSeed {seed}: {len(splits)} ponds, cap {args.sample_cap} TRAIN rows/pond')
    for name, part in splits.items():
        s = part.summary
        flag = ' [short validation horizon]' if s['val_days'] < 3 or s['val_rows'] < 30 else ''
        print(f'  {name:<11} days {s["train_days"]:>3}/{s["val_days"]:>3}/{s["test_days"]:>3}'
              f' | rows {s["train_rows"]:>5}/{s["val_rows"]:>6}/{s["test_rows"]:>6}{flag}')
    print('  [1/5] V3 centralized baseline: validation selection...')
    central = fit_centralized_checkpoints(
        splits, args.rounds, BASELINE['local_epochs'], seed,
        BASELINE['lr'], BASELINE['alpha'], args.selection)
    print('  [2/5] V3 local baseline: validation selection...')
    local = fit_local_checkpoints(
        splits, args.rounds, BASELINE['local_epochs'], seed,
        BASELINE['lr'], BASELINE['alpha'], args.selection)
    print('  [3/5] V3 FedAvg baseline: validation selection...')
    baseline_fed = fit_fed_checkpoints(
        splits, args.rounds, BASELINE['local_epochs'], seed,
        BASELINE['lr'], BASELINE['alpha'], args.selection)
    n_candidates = len(args.lr_grid) * len(args.alpha_grid) * len(args.epochs_grid)
    baseline_in_grid = (BASELINE['lr'] in args.lr_grid
                        and BASELINE['alpha'] in args.alpha_grid
                        and BASELINE['local_epochs'] in args.epochs_grid)
    if not baseline_in_grid:
        print('  NOTE: baseline (lr=0.01, alpha=1, epochs=5) is NOT in search grid;')
        print('        tuned FedAvg need not beat the baseline on validation.')
    print(f'  [4/5] Tune FedAvg on VAL only ({n_candidates} parameter combinations x '
          f'{len(args.rounds)} checkpoints)...')
    tuned, cfg, leaderboard, tune_time = tune_fedavg_on_validation(
        splits, args.rounds, seed, args.lr_grid, args.alpha_grid,
        args.epochs_grid, args.selection)
    print(f'         chosen: lr={cfg["lr"]}, alpha={cfg["alpha"]}, '
          f'local_epochs={cfg["local_epochs"]}, rounds={tuned["selected_round"]}')
    print('  [5/5] Numerical pairwise-masking replay at selected tuned settings...')
    masked = fit_mask_emulation(splits, tuned['selected_round'],
                                cfg['local_epochs'], seed, cfg['lr'], cfg['alpha'])
    diff = float(np.max(np.abs(get_params(*tuned['models']) - get_params(*masked['models']))))
    if diff > 1e-8:
        raise AssertionError(f'Masking changed aggregated model parameters by {diff}')

    fits = {
        'centralized': central,
        'local': local,
        'federated_baseline': baseline_fed,
        'federated_tuned': tuned,
        'mask_emulation_tuned': masked,
    }
    # All hyperparameter and checkpoint choices above were completed using VAL.
    # Only now access TEST, exactly once for each selected fitted approach.
    methods = {}
    for name, fit in fits.items():
        public = _to_public_result(fit)
        public['metrics'] = evaluate_by_pond(splits, fit['models'], 'test')
        if name == 'mask_emulation_tuned':
            public['validation'] = tuned['validation']
            public['selection_source'] = 'federated_tuned_validation_only'
            public['max_weight_difference_from_tuned_fedavg'] = diff
        methods[name] = public
    methods['federated_tuned']['selected_hyperparameters'] = cfg
    methods['federated_tuned']['grid_search_time_sec'] = tune_time
    methods['federated_tuned']['candidate_count'] = n_candidates
    methods['federated_tuned']['validation_grid'] = leaderboard
    methods['mask_emulation_tuned']['selected_hyperparameters'] = cfg
    meta = {
        'protocol_version': 'v4.1',
        'seed': seed,
        'selection_metric': args.selection,
        'hyperparameter_selection': 'validation only; minimize selected RMSE over parameter grid and checkpoints',
        'round_candidates': args.rounds,
        'lr_grid': args.lr_grid, 'alpha_grid': args.alpha_grid,
        'epochs_grid': args.epochs_grid,
        'baseline_hyperparameters': BASELINE,
        'baseline_in_grid': baseline_in_grid,
        'n_clients': len(splits),
        'n_val_rows': sum(len(p.val) for p in splits.values()),
        'n_test_rows': sum(len(p.test) for p in splits.values()),
        'ponds': {name: p.summary for name, p in splits.items()},
        'split': '70/15/15 per pond by distinct local calendar days, cap train only',
        'preprocessing': 'fixed published/public constants; no scaler fit on test or global pool',
        'label': 'synthetic proxy feed_ration_g; not measured feed',
        'privacy': 'masking is single-process additive emulation, NOT cryptographic SMPC',
        'time': 'local sequential training walltime; grid search separately accounted for; excludes network latency',
        'communication': 'analytical payload/handshake estimates, NOT measured network traffic',
        'training_budget': 'different across methods; no compute-matched claims',
        'test_policy': 'test evaluated after all validation choices, but same dates previously inspected in V2/V3: EXPLORATORY ONLY',
        'small_validation_clients': [name for name,p in splits.items() if len(p.val)<30 or p.summary['val_days'] < 3],
    }
    print('  Method                  checkpoint   VAL RMSE   TEST micro RMSE  TEST macro RMSE')
    for name, record in methods.items():
        val_rmse = metric_for_selection(record['validation'], args.selection)
        print(f'  {name:<24}{record["selected_round"]:>4}'
              f'       {val_rmse:>8.4f}      {record["metrics"]["micro"]["rmse"]:>8.4f}'
              f'          {record["metrics"]["macro"]["rmse"]:>8.4f}')
    return {'_meta': meta, 'methods': methods}


def summarize(records):
    output = {}
    for name in records[0]['methods']:
        values_micro = [rec['methods'][name]['metrics']['micro']['rmse'] for rec in records]
        values_macro = [rec['methods'][name]['metrics']['macro']['rmse'] for rec in records]
        output[name] = {
            'micro_rmse_mean': float(np.mean(values_micro)),
            'micro_rmse_std_sample': float(np.std(values_micro, ddof=1)) if len(records)>1 else None,
            'macro_rmse_mean': float(np.mean(values_macro)),
            'macro_rmse_std_sample': float(np.std(values_macro, ddof=1)) if len(records)>1 else None,
            'selected_rounds': [rec['methods'][name]['selected_round'] for rec in records],
        }
    return {
        'protocol_version': 'v4.1',
        'n_seeds': len(records),
        'seeds': [rec['_meta']['seed'] for rec in records],
        'note': 'Same test windows for seeds; descriptive variability only, not independent generalization trials.',
        'methods': output,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample-cap', type=int, default=3000,
                    help='maximum TRAIN observations per pond; validation/test not capped')
    ap.add_argument('--rounds', type=lambda t: _int_grid(t, 'rounds'), default=[5,10,20,50])
    ap.add_argument('--seeds', type=lambda t: _int_grid(t, 'seeds'), default=[42])
    ap.add_argument('--lr-grid', type=lambda t: _float_grid(t, 'lr-grid', False), default=[0.005,0.01,0.02])
    ap.add_argument('--alpha-grid', type=lambda t: _float_grid(t, 'alpha-grid', True), default=[0.0,0.1,1.0])
    ap.add_argument('--epochs-grid', type=lambda t: _int_grid(t, 'epochs-grid'), default=[1,5])
    ap.add_argument('--selection', choices=['macro_rmse','micro_rmse'], default='macro_rmse')
    ap.add_argument('--dataset-dir', default=None)
    ap.add_argument('--output-dir', type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    if args.sample_cap <= 0:
        ap.error('--sample-cap must be > 0')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_key = {
        'sample_cap': args.sample_cap, 'rounds': args.rounds,
        'seeds': args.seeds, 'lr_grid': args.lr_grid,
        'alpha_grid': args.alpha_grid, 'epochs_grid': args.epochs_grid,
        'selection': args.selection, 'dataset_dir': args.dataset_dir,
    }
    run_id = hashlib.sha256(json.dumps(run_key, sort_keys=True).encode()).hexdigest()[:10]
    print(f'Configuration ID: {run_id} (prevents overwriting another grid)')
    records=[]
    for seed in args.seeds:
        record = run_seed(seed,args)
        ident = f'seed{seed}_cap{args.sample_cap}_{args.selection}_r{"-".join(map(str,args.rounds))}_{run_id}'
        filename = args.output_dir / f'{ident}.json'
        filename.write_text(json.dumps(record,indent=2,allow_nan=False), encoding='utf-8')
        print(f'  Saved: {filename}')
        records.append(record)
    summary = summarize(records)
    summary_name = args.output_dir / ('summary_' + args.selection + '_seeds' + '-'.join(map(str,args.seeds)) + '_' + run_id + '.json')
    summary_name.write_text(json.dumps(summary,indent=2,allow_nan=False), encoding='utf-8')
    print('\nMean TEST RMSE (exploratory):')
    for method, record in summary['methods'].items():
        print(f'  {method:<24} micro={record["micro_rmse_mean"]:.4f}'
              f' macro={record["macro_rmse_mean"]:.4f}')
    print(f'Summary saved: {summary_name}')
    print('WARNING: simulated feeding target; additive masking is NOT cryptographic SMPC; test dates already explored in V2/V3.')
    return records


if __name__ == '__main__':
    main()
