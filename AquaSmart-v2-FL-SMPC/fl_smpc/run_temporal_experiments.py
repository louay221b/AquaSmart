"""Exploratory leakage-aware comparison (separate from original run_experiments).

Run from AquaSmart-v2-FL-SMPC:
  python -m fl_smpc.run_temporal_experiments --sample-cap 3000 --rounds 10

This is a methodology prototype, NOT validation on measured feed intake,
and the pairwise-masking backend is a single-process emulation, NOT SMPC.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fl_smpc.common import gd_update, get_params, split_params, N_FEATURES, PARAM_BYTES
from fl_smpc.data import TARGET_COL
from fl_smpc.temporal_protocol import load_temporal_splits, public_transform

RESULTS = Path(__file__).resolve().parent / "results" / "temporal_comparison_results.json"


def _evaluate_by_pond(splits, models):
    all_true, all_pred = [], []
    per_pond = {}
    for name, part in splits.items():
        w, b = models[name] if isinstance(models, dict) else models
        y = part.test[TARGET_COL].to_numpy(dtype=np.float64)
        pred = public_transform(part.test) @ w + b
        all_true.append(y)
        all_pred.append(pred)
        per_pond[name] = {
            "n": len(y), "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else None,
        }
    y, pred = np.concatenate(all_true), np.concatenate(all_pred)
    micro = {"mae": float(mean_absolute_error(y, pred)),
             "rmse": float(np.sqrt(mean_squared_error(y, pred))),
             "r2": float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else None}
    macro = {"mae": float(np.mean([v["mae"] for v in per_pond.values()])),
             "rmse": float(np.mean([v["rmse"] for v in per_pond.values()])),
             "r2": float(np.mean([v["r2"] for v in per_pond.values() if v["r2"] is not None]))}
    return {"micro": micro, "macro": macro, "per_pond": per_pond}


def _params():
    return np.zeros(N_FEATURES, dtype=np.float64), 0.0


def train_centralized(splits, epochs, seed):
    t0 = time.perf_counter()
    X = np.vstack([public_transform(part.train) for part in splits.values()])
    y = np.concatenate([part.train[TARGET_COL].to_numpy(dtype=np.float64)
                        for part in splits.values()])
    w, b = _params()
    w, b = gd_update(X, y, w, b, epochs=epochs, rng=np.random.default_rng(seed))
    return {"metrics": _evaluate_by_pond(splits, (w, b)),
            "time_sec": time.perf_counter()-t0,
            "comm_bytes": int((len(y) * (N_FEATURES + 1)) * 8),
            "n_messages": len(splits), "training_epochs": epochs}


def train_local(splits, epochs, seed):
    t0 = time.perf_counter()
    models = {}
    for i, (name, part) in enumerate(splits.items()):
        X = public_transform(part.train)
        y = part.train[TARGET_COL].to_numpy(dtype=np.float64)
        w, b = _params()
        models[name] = gd_update(X, y, w, b, epochs=epochs,
                                 rng=np.random.default_rng(seed+i))
    return {"metrics": _evaluate_by_pond(splits, models),
            "time_sec": time.perf_counter()-t0, "comm_bytes": 0,
            "n_messages": 0, "training_epochs": epochs}


def train_federated(splits, rounds, local_epochs, seed, masked=False):
    t0 = time.perf_counter()
    train = {name:(public_transform(part.train),
                   part.train[TARGET_COL].to_numpy(dtype=np.float64))
             for name,part in splits.items()}
    w_global,b_global = _params()
    rng = np.random.default_rng(seed)
    history = []
    max_agg_error = 0.0
    mask_compute_s = 0.0
    for rnd in range(1, rounds+1):
        weighted = []
        total = sum(len(y) for _, y in train.values())
        for name,(X,y) in train.items():
            w,b = gd_update(X, y, w_global.copy(), b_global,
                            epochs=local_epochs, rng=rng)
            weighted.append((len(y)/total)*get_params(w,b))
        plain_agg = np.sum(weighted, axis=0)
        if masked:
            start_mask = time.perf_counter()
            # SIMULATION: masks are constructed here in one process. A real
            # secure aggregation protocol needs independent client secrets.
            masks_rng = np.random.default_rng(seed * 1_000_003 + rnd)
            masked_parts = [v.copy() for v in weighted]
            for i in range(len(masked_parts)):
                for j in range(i+1,len(masked_parts)):
                    mask = masks_rng.normal(0,10,size=N_FEATURES+1)
                    masked_parts[i] += mask
                    masked_parts[j] -= mask
            aggregate = np.sum(masked_parts, axis=0)
            max_agg_error = max(max_agg_error,
                                float(np.max(np.abs(plain_agg-aggregate))))
            mask_compute_s += time.perf_counter()-start_mask
        else:
            aggregate = plain_agg
        w_global,b_global = split_params(aggregate)
        micro = _evaluate_by_pond(splits, (w_global, b_global))["micro"]
        history.append({"round":rnd, **micro})
    elapsed = time.perf_counter() - t0
    n_clients = len(splits)
    # Message size accounting is only an analytical approximation;
    # masked mode adds a hypothetical seed-exchange budget, not real traffic.
    base_messages = rounds*2*n_clients
    base_bytes = base_messages*PARAM_BYTES
    extra_messages = rounds*n_clients*(n_clients-1)//2 if masked else 0
    extra_bytes = extra_messages*16 if masked else 0
    return {"metrics": _evaluate_by_pond(splits, (w_global,b_global)),
            "time_sec":elapsed, "comm_bytes":base_bytes+extra_bytes,
            "n_messages":base_messages+extra_messages,
            "history":history, "rounds":rounds, "local_epochs":local_epochs,
            "mask_compute_sec":mask_compute_s if masked else 0.0,
            "max_aggregate_error":max_agg_error if masked else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-cap", type=int, default=3000,
                    help="maximum TRAIN rows per pond; val/test are not capped")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-dir", default=None)
    args = ap.parse_args()
    if args.rounds < 1 or args.local_epochs < 1 or args.sample_cap < 1:
        ap.error("rounds, local-epochs et sample-cap doivent etre >= 1")
    splits = load_temporal_splits(args.sample_cap,args.seed,args.dataset_dir)
    print("\nPer-pond temporal splits (day groups never overlap):")
    for name, part in splits.items():
        p=part.summary
        note=" [small test horizon: sensitivity check needed]" if p["n_days"]<20 else ""
        print(f"  {name:<11} days {p['train_days']:>3}/{p['val_days']:>3}/{p['test_days']:>3}"
              f" | rows {p['train_rows']:>5}/{p['val_rows']:>6}/{p['test_rows']:>6}{note}")
    epochs=args.rounds*args.local_epochs
    methods={}
    print("\n[1/4] Centralized (training only)...")
    methods["centralized"]=train_centralized(splits,epochs,args.seed)
    print("[2/4] Local (own-pond test)...")
    methods["local"]=train_local(splits,epochs,args.seed)
    print("[3/4] FedAvg...")
    methods["federated"]=train_federated(splits,args.rounds,args.local_epochs,args.seed)
    print("[4/4] FedAvg + pairwise masking (emulation ONLY)...")
    methods["mask_emulation"]=train_federated(splits,args.rounds,args.local_epochs,args.seed,masked=True)
    result={"_meta":{
        "target":"synthetic_feed_ration_g_proxy_NOT_measured_feed",
        "split":"70/15/15 by distinct local calendar days per pond (rounded)",
        "preprocessing":"public fixed feature centers/scales and clipping; no train/test fitted scaler",
        "validation":"reserved, NOT used for hyperparameter selection in this prototype",
        "privacy":"single-process pairwise masking emulator; NOT cryptographically secure SMPC",
        "communication":"analytical estimates, NOT measured network traffic",
        "times":"local sequential simulation wall times, NOT distributed latency",
        "sample_cap_train_only":args.sample_cap,"seed":args.seed,
        "ponds":{name:p.summary for name,p in splits.items()},
        "n_test_rows":sum(len(p.test) for p in splits.values())},
        "methods":methods}
    RESULTS.parent.mkdir(parents=True,exist_ok=True)
    RESULTS.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("\nMethod                  micro MAE    micro RMSE     micro R2      time(s)")
    for name,r in methods.items():
        m=r["metrics"]["micro"]
        print(f"{name:<23} {m['mae']:>9.4f} {m['rmse']:>13.4f} {m['r2']:>12.4f} {r['time_sec']:>12.2f}")
    print(f"\nSaved: {RESULTS}")
    print("NOTE: synthetic feed target; pairwise masking is NOT cryptographic SMPC.")


if __name__ == "__main__":
    main()
