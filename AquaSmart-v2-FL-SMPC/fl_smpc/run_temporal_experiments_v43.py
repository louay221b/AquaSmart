"""AquaSmart V4.3: cryptographic secure-aggregation replay of V4.2.

This runs all clients in one process for an IN-PROCESS PROTOCOL BENCHMARK, not a
production privacy-preserving distributed system. Each SecureClient owns its
signing/X25519 keys and computes its masked weighted model locally; server API
gets only signed masked vectors. All clients must finish every round; fail
closed on dropout. Uses X25519 + Ed25519 + HKDF-SHA256 + ChaCha20 and 1e8
fixed-point modular arithmetic. No simulated normal-noise pairwise masks.

The plaintext V4.2 settings are selected only on VAL. Crypto replay is compared
against the plaintext counterpart with the SAME seed, rounds, optimizer,
client sequence and local fine-tuning. TEST is scored after all selections.

The label is a synthetic feed proxy and all existing test windows were already
used for development: these results remain EXPLORATORY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
from pathlib import Path

import cryptography
import numpy as np

from fl_smpc.common import get_params, split_params, gd_update, N_FEATURES
from fl_smpc.run_temporal_experiments_v3 import (
    _cached_data, _zero_model, fit_centralized_checkpoints,
    fit_local_checkpoints, fit_fed_checkpoints, evaluate_by_pond,
    metric_for_selection, _to_public_result,
)
from fl_smpc.run_temporal_experiments_v4 import (
    _float_grid, _int_grid, _validate_dataset, tune_fedavg_on_validation, BASELINE,
)
from fl_smpc.run_temporal_experiments_v42 import (
    fit_personalized_local, tune_personalization_on_validation, _epochs_grid,
)
from fl_smpc.secure_aggregation import (
    SecureClient, SecureServer, _wire_stats, canonical, PRIME, SCALE, VERSION,
)
from fl_smpc.temporal_protocol import load_temporal_splits

OUT_DIR = Path(__file__).resolve().parent / 'results' / 'temporal_v43'


def fit_cryptographic_fedavg(splits, seed: int, rounds: int,
                              local_epochs: int, lr: float, alpha: float):
    """Replay the exact V4.2 training schedule with signed, crypto-masked SUM.

    Only TRAIN data used; does not evaluate VAL or TEST. Client training and
    private keys share the same Python process in this simulation; server
    object only handles authenticated masked packets.
    """
    train = _cached_data(splits)
    ids = tuple(train)
    if len(ids) < 2:
        raise ValueError('Secure aggregation requires at least two clients')
    dimension = N_FEATURES + 1
    clients = {k: SecureClient(k) for k in ids}
    server = SecureServer({k: c.identity_public_key for k, c in clients.items()},
                          dimension=dimension)
    total_train_rows = sum(len(y) for _, y in train.values())
    rng = np.random.default_rng(seed)
    current = _zero_model()
    key_bytes = roster_bytes = update_bytes = model_bytes = messages = 0
    key_and_mask_sec = verify_sum_sec = 0.0
    local_and_mask_time = 0.0
    mask_crypto_sec = 0.0
    start_all = time.perf_counter()

    for rnd in range(1, rounds + 1):
        # Fresh unpredictable session identifier, new authenticated ephemeral
        # ECDH keys each round, independent of deterministic SGD seed.
        session_id = secrets.token_hex(24)
        start_key = time.perf_counter()
        announcements = [clients[k].start_round(session_id) for k in ids]
        key_setup_sec = time.perf_counter() - start_key
        updates = []
        # Exactly the same client order + RNG consumption as V4.2.
        # The coordinator calls client.train_and_mask and receives masked
        # packet only; plaintext vectors do not enter SecureServer methods.
        for k, (X, y) in train.items():
            start_local = time.perf_counter()
            update = clients[k].train_and_mask(
                X, y, current, local_epochs, lr, alpha, rng,
                len(y) / total_train_rows, announcements, server.trusted_keys,
                server.expected_ids, dimension)
            local_and_mask_time += time.perf_counter() - start_local
            mask_crypto_sec += clients[k].last_mask_crypto_sec
            updates.append(update)
        start_verify = time.perf_counter()
        aggregated, receipt = server.aggregate(announcements, updates, session_id)
        verify_sum_sec += time.perf_counter() - start_verify
        current = split_params(aggregated)
        # Counts serialized protocol bytes (in-memory) and an estimated model
        # broadcast payload. No TCP/TLS sockets are used.
        stats = _wire_stats(announcements, updates, dimension, 0,
                            receipt['server_verify_and_sum_sec'])
        key_bytes += stats.key_upload_bytes
        roster_bytes += stats.roster_broadcast_bytes
        update_bytes += stats.masked_upload_bytes
        model_bytes += stats.global_model_broadcast_bytes_estimate
        messages += stats.message_count
        key_and_mask_sec += key_setup_sec

    elapsed = time.perf_counter() - start_all
    return {
        'models': current, 'selected_round': rounds,
        'time_sec': elapsed, 'local_training_and_mask_sec': local_and_mask_time,
        'estimated_local_training_sec': max(0.0, local_and_mask_time - mask_crypto_sec),
        'crypto_mask_and_sign_sec': mask_crypto_sec,
        'ephemeral_key_creation_sec': key_and_mask_sec,
        'server_verify_and_sum_sec': verify_sum_sec,
        'comm_bytes': key_bytes + roster_bytes + update_bytes + model_bytes,
        'n_messages': messages,
        'communication_breakdown': {
            'key_upload_bytes': key_bytes,
            'roster_broadcast_bytes': roster_bytes,
            'signed_masked_update_bytes': update_bytes,
            'global_model_broadcast_bytes_estimate': model_bytes,
            'total_bytes_serialized_or_estimated': key_bytes + roster_bytes + update_bytes + model_bytes,
            'note': 'Serialized application payloads in one process; NOT measured network traffic/latency, TLS or transport headers',
        },
        'crypto': {
            'protocol': VERSION,
            'library': f'cryptography {cryptography.__version__}',
            'dh': 'X25519 ephemeral per round',
            'authentication': 'Ed25519 signatures + preregistered public verification keys',
            'kdf': 'HKDF-SHA256',
            'mask_stream': 'ChaCha20',
            'field_prime': PRIME,
            'fixed_point_scale': SCALE,
            'dropout_policy': 'ABORT round if any client absent; no recovery',
            'deployment': 'in-process demo, not network-distributed or audited',
        },
        'privacy_limitations': [
            'Honest-but-curious server, honest clients, all online, no server/client collusion',
            'No protection from inference based on aggregate itself; no differential privacy',
            'Not malicious-client robust, not dropout tolerant, not threshold secure aggregation',
            'The experiment orchestrator holds TRAIN datasets in the same process',
        ],
    }


def run_seed(seed, args):
    splits = load_temporal_splits(args.sample_cap, seed, args.dataset_dir)
    _validate_dataset(splits)
    print(f'\nSeed {seed}: {len(splits)} ponds, training cap {args.sample_cap}')
    print('  [1/6] Baselines and FedAvg search: validation only...')
    central = fit_centralized_checkpoints(
        splits, args.rounds, BASELINE['local_epochs'], seed,
        BASELINE['lr'], BASELINE['alpha'], args.selection)
    local = fit_local_checkpoints(
        splits, args.rounds, BASELINE['local_epochs'], seed,
        BASELINE['lr'], BASELINE['alpha'], args.selection)
    base = fit_fed_checkpoints(
        splits, args.rounds, BASELINE['local_epochs'], seed,
        BASELINE['lr'], BASELINE['alpha'], args.selection)
    tuned, cfg, leaderboard, grid_time = tune_fedavg_on_validation(
        splits, args.rounds, seed, args.lr_grid, args.alpha_grid,
        args.epochs_grid, args.selection)
    print(f'         FedAvg tuned: rounds={tuned["selected_round"]}, lr={cfg["lr"]},'
          f' alpha={cfg["alpha"]}, epochs={cfg["local_epochs"]}')
    print('  [2/6] Personalization search: validation only...')
    personal_models, pcfg, pval, pgrid, psearch, pfit = tune_personalization_on_validation(
        splits, tuned['models'], seed, args.personal_lr_grid,
        args.personal_epochs_grid, cfg['alpha'], args.selection)
    print(f'         Personalization: epochs={pcfg["personal_epochs"]}, lr={pcfg["personal_lr"]}')
    print('  [3/6] Replay with signed, cryptographic pairwise secure SUM...')
    secure = fit_cryptographic_fedavg(
        splits, seed, tuned['selected_round'], cfg['local_epochs'],
        cfg['lr'], cfg['alpha'])
    diff_global = float(np.max(np.abs(get_params(*tuned['models']) -
                                      get_params(*secure['models']))))
    if diff_global > args.max_numeric_diff:
        raise AssertionError(f'Cryptographic aggregation differs from plaintext: {diff_global}')
    print(f'         Max global parameter difference: {diff_global:.3g}')
    print('  [4/6] Replay exact personalization on secure global model...')
    secure_personal_models, local_seconds = fit_personalized_local(
        splits, secure['models'], seed, pcfg['personal_lr'],
        pcfg['personal_epochs'], cfg['alpha'])
    diff_personal = max(float(np.max(np.abs(get_params(*personal_models[k]) -
                                              get_params(*secure_personal_models[k]))))
                        for k in splits)
    if diff_personal > args.max_numeric_diff:
        raise AssertionError(f'Personalization changed with cryptographic SUM: {diff_personal}')
    secure['validation'] = evaluate_by_pond(splits, secure['models'], 'val')
    secure['max_plain_model_param_difference'] = diff_global
    secure['selection_source'] = 'plain_FedAvg_validation_only'
    secure_personal = {
        **{k: v for k, v in secure.items() if k != 'models'},
        'models': secure_personal_models,
        'validation': evaluate_by_pond(splits, secure_personal_models, 'val'),
        'selected_personalization': pcfg,
        'time_sec': secure['time_sec'] + local_seconds,
        'local_personalization_time_sec': local_seconds,
        'personalization_additional_comm_bytes': 0,
        'max_plain_model_param_difference': diff_personal,
        'selection_source': 'plain_FedAvg_and_personalization_validation_only',
    }
    tuned['selected_hyperparameters'] = cfg
    tuned['grid_search_time_sec'] = grid_time
    tuned['validation_grid'] = leaderboard
    personal_fit = {
        'models': personal_models,
        'selected_round': tuned['selected_round'],
        'selected_personalization': pcfg,
        'validation': pval,
        'personalization_grid': pgrid,
        'personalization_grid_time_sec': psearch,
        'time_sec': tuned['time_sec'] + pfit,
        'comm_bytes': tuned['comm_bytes'],
        'n_messages': tuned['n_messages'],
        'personalization_additional_comm_bytes': 0,
    }
    fits = {
        'centralized': central,
        'local': local,
        'federated_baseline': base,
        'federated_tuned': tuned,
        'personalized_fedavg': personal_fit,
        'cryptographic_secagg_fedavg': secure,
        'cryptographic_secagg_personalized': secure_personal,
    }
    print('  [5/6] Freeze hyperparameters and perform exploratory TEST scoring...')
    methods = {}
    for key, fitted in fits.items():
        out = _to_public_result(fitted)
        out['metrics'] = evaluate_by_pond(splits, fitted['models'], 'test')
        methods[key] = out
        print(f'  {key:<34} micro={out["metrics"]["micro"]["rmse"]:.4f}'
              f'  macro={out["metrics"]["macro"]["rmse"]:.4f}')
    print('  [6/6] Summarize cryptographic overhead and numerical equivalence...')
    print(f'         Serialized/estimated secagg payload: {secure["comm_bytes"]:,} bytes'
          f' (plain FedAvg estimate: {tuned["comm_bytes"]:,} bytes)')
    meta = {
        'protocol_version': 'v4.3-cryptographic-in-process-prototype',
        'seed': seed, 'n_clients': len(splits),
        'n_train_rows': sum(len(x.train) for x in splits.values()),
        'sample_cap': args.sample_cap,
        'split': 'same V4.2 per-pond day-grouped train/val/test splits; training cap only',
        'selection_metric': args.selection,
        'round_candidates': args.rounds,
        'global_selected_config': cfg,
        'global_selected_rounds': tuned['selected_round'],
        'personal_selected_config': pcfg,
        'max_numeric_diff_allowed': args.max_numeric_diff,
        'numeric_error_global': diff_global,
        'numeric_error_personalized': diff_personal,
        'cryptographic_secagg_status': 'real cryptographic primitives, SINGLE PROCESS, ALL-ONLINE ONLY; not full deployed SMPC',
        'security_assumptions': 'honest clients; honest-but-curious non-colluding server; pretrusted identity keys; no dropouts',
        'network_measurement': 'not collected: in-memory serialized bytes only',
        'test_policy': 'exploratory, repeated already-inspected test dates',
        'label': 'feed_ration_g synthetic proxy, not observed feeding',
        'compute_fairness': 'methods use different training budgets',
        'tiny_val_clients': [k for k,p in splits.items()
                             if p.summary['val_days'] < 3 or len(p.val) < 30],
    }
    return {'_meta': meta, 'methods': methods}


def summarize(records):
    methods = {}
    for key in records[0]['methods']:
        micro = [r['methods'][key]['metrics']['micro']['rmse'] for r in records]
        macro = [r['methods'][key]['metrics']['macro']['rmse'] for r in records]
        methods[key] = {
            'micro_rmse_mean': float(np.mean(micro)),
            'micro_rmse_sd': float(np.std(micro, ddof=1)) if len(micro) > 1 else None,
            'macro_rmse_mean': float(np.mean(macro)),
            'macro_rmse_sd': float(np.std(macro, ddof=1)) if len(macro) > 1 else None,
        }
    return {
        'protocol_version': 'v4.3-cryptographic-in-process-prototype',
        'seeds': [r['_meta']['seed'] for r in records],
        'methods': methods,
        'mean_secure_payload_bytes': float(np.mean([
            r['methods']['cryptographic_secagg_fedavg']['comm_bytes'] for r in records])),
        'max_numeric_error_global': float(max(r['_meta']['numeric_error_global'] for r in records)),
        'max_numeric_error_personalized': float(max(r['_meta']['numeric_error_personalized'] for r in records)),
        'warning': 'Not a production SMPC deployment; no dropout recovery, no malicious security, no network benchmark; proxy labels and explored test dates',
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-cap', type=int, default=3000)
    p.add_argument('--rounds', type=lambda s: _int_grid(s, 'rounds'), default=[5,10,20])
    p.add_argument('--lr-grid', type=lambda s: _float_grid(s, 'lr-grid', False), default=[0.005,0.01])
    p.add_argument('--alpha-grid', type=lambda s: _float_grid(s, 'alpha-grid', True), default=[0.1,1.0])
    p.add_argument('--epochs-grid', type=lambda s: _int_grid(s, 'epochs-grid'), default=[1,5])
    p.add_argument('--personal-lr-grid', type=lambda s: _float_grid(s, 'personal-lr-grid', False),
                   default=[0.001,0.005,0.01])
    p.add_argument('--personal-epochs-grid', type=_epochs_grid, default=[0,1,3,5])
    p.add_argument('--seeds', type=lambda s: _int_grid(s, 'seeds'), default=[42])
    p.add_argument('--selection', choices=['macro_rmse','micro_rmse'], default='macro_rmse')
    p.add_argument('--dataset-dir', default=None)
    p.add_argument('--output-dir', type=Path, default=OUT_DIR)
    p.add_argument('--max-numeric-diff', type=float, default=2e-6)
    args = p.parse_args(argv)
    if args.sample_cap <= 0 or not np.isfinite(args.max_numeric_diff) or args.max_numeric_diff <= 0:
        p.error('sample cap and numerical tolerance must be finite positive values')
    key = {k: v for k,v in vars(args).items() if k != 'output_dir'}
    run_id = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'V4.3 run ID: {run_id}; cryptography {cryptography.__version__}')
    results = []
    for seed in args.seeds:
        record = run_seed(seed, args)
        path = args.output_dir / f'seed{seed}_{args.selection}_{run_id}.json'
        path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
        results.append(record)
        print(f'  Saved: {path}')
    summary = summarize(results)
    summary_path = args.output_dir / f'summary_{args.selection}_{run_id}.json'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(f'\nSummary saved: {summary_path}')
    print('WARNING: cryptographic primitives in single-process all-online prototype only;'
          ' NOT deployed SMPC / no dropout recovery; synthetic labels, explored test windows.')
    return results


if __name__ == '__main__':
    main()
