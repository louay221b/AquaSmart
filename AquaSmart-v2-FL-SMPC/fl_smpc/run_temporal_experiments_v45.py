"""V4.5 AquaSmart: day-separated VALIDATION benchmark over TCP/mTLS loopback.

No test-set evaluation and no hyperparameter optimization; FedAvg and
personalization parameters are frozen from exploratory V4.2/V4.4. The parent
benchmarks local SGD and owns the updates; its network *server process* only
sees cryptographically masked updates. This is not privacy from the parent.

CMD:
 python -m fl_smpc.run_temporal_experiments_v45 --sample-cap 3000 --rounds 2 --seeds 42 --dropouts IoTPond2,IoTPond7
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
from fl_smpc.run_temporal_experiments_v3 import _cached_data, _zero_model, evaluate_by_pond
from fl_smpc.run_temporal_experiments_v42 import fit_personalized_local
from fl_smpc.temporal_protocol import load_temporal_splits
from fl_smpc.tls_secagg_v45 import tls_secure_sum_round

OUTPUT = Path(__file__).resolve().parent / 'results' / 'temporal_v45'


def parse_dropouts(value):
    if not value or value.strip().lower() == 'none':
        return ()
    names = tuple(s.strip() for s in value.split(',') if s.strip())
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError('Duplicate dropout client name')
    return names


def scenario(splits, seed, rounds, missing, *, epochs, lr, alpha,
             personal_epochs, personal_lr, timeout_sec):
    train = _cached_data(splits)
    ids = list(train)
    if not set(missing).issubset(ids):
        raise ValueError('Unknown dropout client: ' + ', '.join(set(missing) - set(ids)))
    present = [i for i in ids if i not in missing]
    if len(present) < 2:
        raise ValueError('At least two survivors required')
    total = sum(len(y) for _, y in train.values())
    weights = {i: len(y) / total for i, (_, y) in train.items()}
    denom = sum(weights[i] for i in present)
    rng = np.random.default_rng(seed)
    current = _zero_model()
    elapsed_train = 0.0
    details = []
    max_error = 0.0
    for round_index in range(1, rounds + 1):
        begin = time.perf_counter()
        # All clients train; named dropouts stop AFTER key announcement.
        # Parent harness intentionally sees all cleartext contributions.
        contributions = {}
        for client_id, (x, y) in train.items():
            w, b = gd_update(x, y, current[0].copy(), current[1],
                             epochs=epochs, lr=lr, alpha=alpha, rng=rng)
            contributions[client_id] = weights[client_id] * get_params(w, b)
        elapsed_train += time.perf_counter() - begin
        expected = sum((contributions[i] for i in present),
                       start=np.zeros_like(next(iter(contributions.values())))) / denom
        result, receipt, metric = tls_secure_sum_round(
            contributions, missing=missing, timeout_sec=timeout_sec,
            session=secrets.token_hex(24),
        )
        global_params = result / denom
        diff = float(np.max(np.abs(global_params - expected)))
        if diff > 4e-6:
            raise AssertionError('TLS secure aggregation and plaintext FedAvg diverged')
        max_error = max(max_error, diff)
        current = split_params(global_params)
        details.append({
            'round': round_index,
            'uploaded': receipt['n_uploaded'],
            'dropped': receipt['n_dropped'],
            'max_parameter_diff': diff,
            **metric,
        })
    personal_models, elapsed_personal = fit_personalized_local(
        splits, current, seed, personal_lr, personal_epochs, alpha)
    validation = evaluate_by_pond(splits, personal_models, 'val')
    return {
        'n_ponds': len(ids), 'survivors': present,
        'dropouts': sorted(missing), 'rounds': rounds,
        'validation': validation,
        'max_plain_tls_param_diff': max_error,
        'tls_application_frame_bytes': sum(x['tls_application_frame_bytes'] for x in details),
        'tls_application_frames': sum(x['tls_application_frames'] for x in details),
        'network_round_wall_sec': sum(x['network_round_wall_sec'] for x in details),
        'local_train_sec': elapsed_train,
        'personalization_sec': elapsed_personal,
        'rounds_detail': details,
        'scope': 'Loopback TCP with mTLS 1.3, validation-only; benchmark parent has plaintext',
    }


def run_seed(seed, args):
    splits = load_temporal_splits(args.sample_cap, seed, args.dataset_dir)
    print(f'\nSeed {seed}: {len(splits)} ponds; {args.rounds} round(s)')
    scenarios = {}
    for label, missing in [
        ('all_online', ()),
        ('after_announcement_dropout', args.dropouts),
    ]:
        print(f'  {label}: missing={list(missing)}', flush=True)
        result = scenario(
            splits, seed, args.rounds, missing,
            epochs=args.epochs, lr=args.lr, alpha=args.alpha,
            personal_epochs=args.personal_epochs, personal_lr=args.personal_lr,
            timeout_sec=args.timeout_sec,
        )
        scenarios[label] = result
        print('    n=%d  VAL micro=%.4f macro=%.4f  max Δ=%.2e  TLS-app=%d bytes  wall=%.2f s' % (
            len(result['survivors']), result['validation']['micro']['rmse'],
            result['validation']['macro']['rmse'], result['max_plain_tls_param_diff'],
            result['tls_application_frame_bytes'], result['network_round_wall_sec'],
        ), flush=True)
    return {
        'seed': seed,
        'scenarios': scenarios,
        'configuration': {
            'sample_cap': args.sample_cap, 'rounds': args.rounds,
            'epochs': args.epochs, 'lr': args.lr, 'alpha': args.alpha,
            'personal_epochs': args.personal_epochs, 'personal_lr': args.personal_lr,
            'dropouts': list(args.dropouts),
        },
        'warnings': [
            'Proxy feeding target generated from heuristic, not a measured feed ration',
            'Validation-only; do not infer unseen future prediction from these windows',
            'No independent test split accessed',
            'Parent benchmark knows plaintext; network server process does not',
            'Temporary CA; NOT production identity provisioning',
            'Server honest-but-curious; no malicious server, no collusion/DP',
            'Dropout after announcement but before upload; late dropout aborts',
            'Loopback TLS application frames; excludes TLS/TCP handshake/payload overhead',
            'All clients train, including ones that subsequently abandon upload',
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample-cap', type=int, default=3000)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--seeds', default='42')
    ap.add_argument('--dropouts', type=parse_dropouts, default=('IoTPond10',))
    ap.add_argument('--lr', type=float, default=.01)
    ap.add_argument('--alpha', type=float, default=.1)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--personal-lr', type=float, default=.01)
    ap.add_argument('--personal-epochs', type=int, default=3)
    ap.add_argument('--timeout-sec', type=float, default=45.0)
    ap.add_argument('--dataset-dir', default=None)
    ap.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = ap.parse_args(argv)
    if (args.sample_cap < 1 or args.rounds < 1 or args.epochs < 1
            or args.personal_epochs < 0 or args.timeout_sec <= 0
            or not all(np.isfinite(x) for x in (args.lr, args.alpha, args.personal_lr))
            or args.lr <= 0 or args.alpha < 0 or args.personal_lr <= 0):
        ap.error('Invalid numeric parameter')
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
        if not seeds or min(seeds) < 0:
            raise ValueError()
    except ValueError:
        ap.error('Seeds must be comma-separated nonnegative integers')
    if not args.dropouts:
        ap.error('Specify --dropouts with at least one client for the scenario comparison')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) for k, v in vars(args).items() if k != 'output_dir'}
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    print(f'AquaSmart V4.5 run={run_id}, loopback TCP/mTLS1.3, VAL-only', flush=True)
    records = []
    for seed in seeds:
        r = run_seed(seed, args)
        location = args.output_dir / f'seed{seed}_{run_id}.json'
        location.write_text(json.dumps(r, indent=2, allow_nan=False), encoding='utf-8')
        print('  Saved:', location, flush=True)
        records.append(r)
    summary = {'run_id': run_id, 'seeds': seeds,
               'validation_only': True,
               'tls_app_bytes_not_wire_bytes': True}
    # Compute summary explicitly to preserve the metric provenance.
    summary['scenario_means'] = {}
    for name in ('all_online', 'after_announcement_dropout'):
        samples = [r['scenarios'][name] for r in records]
        summary['scenario_means'][name] = {
            'micro_rmse_mean': float(np.mean([s['validation']['micro']['rmse'] for s in samples])),
            'macro_rmse_mean': float(np.mean([s['validation']['macro']['rmse'] for s in samples])),
            'tls_application_frame_bytes_mean': float(np.mean([s['tls_application_frame_bytes'] for s in samples])),
            'network_round_wall_sec_mean': float(np.mean([s['network_round_wall_sec'] for s in samples])),
            'max_parameter_diff_observed': float(max(s['max_plain_tls_param_diff'] for s in samples)),
        }
    final = args.output_dir / f'summary_{run_id}.json'
    final.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print('  Summary:', final, flush=True)
    return records


if __name__ == '__main__':
    main()