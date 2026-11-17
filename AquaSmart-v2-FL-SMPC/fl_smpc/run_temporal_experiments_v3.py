"""Validation-selected temporal baseline for AquaSmart (V3).

Run from the AquaSmart-v2-FL-SMPC directory:
    python -m fl_smpc.run_temporal_experiments_v3 --sample-cap 3000 --rounds 5,10,20,50 --seeds 11,12,13,14,15

The test set is never inspected for model/checkpoint selection. The same
per-pond chronological split is shared by all four approaches within a seed.
This is a proxy-label benchmark, NOT validation on measured feeding amounts.
The pairwise masking is a single-process numerical emulation, NOT SMPC.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fl_smpc.common import gd_update, get_params, split_params, N_FEATURES, PARAM_BYTES
from fl_smpc.data import TARGET_COL
from fl_smpc.temporal_protocol import load_temporal_splits, public_transform

OUT_DIR = Path(__file__).resolve().parent / "results" / "temporal_v3"


def _score(y, pred):
    """Regression metrics for a defined subset; R2 may be undefined."""
    if len(y) < 1:
        raise ValueError("Cannot evaluate an empty split")
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 and np.var(y) > 0 else None,
    }


def evaluate_by_pond(splits, models, stage):
    """Compute metrics on VAL or TEST. Caller controls when TEST is accessed."""
    if stage not in ("val", "test"):
        raise ValueError("stage must be 'val' or 'test'")
    per_pond = {}
    ys, preds = [], []
    for name, part in splits.items():
        frame = getattr(part, stage)
        w, b = models[name] if isinstance(models, dict) else models
        y = frame[TARGET_COL].to_numpy(dtype=np.float64)
        pred = public_transform(frame) @ w + b
        per_pond[name] = _score(y, pred)
        ys.append(y)
        preds.append(pred)
    micro = _score(np.concatenate(ys), np.concatenate(preds))
    macro = {}
    for key in ("mae", "rmse", "r2"):
        values = [m[key] for m in per_pond.values() if m[key] is not None]
        macro[key] = float(np.mean(values)) if values else None
    macro["n_clients"] = len(per_pond)
    return {"micro": micro, "macro": macro, "per_pond": per_pond}


def metric_for_selection(scores, selection):
    if selection == "micro_rmse":
        return scores["micro"]["rmse"]
    if selection == "macro_rmse":
        return scores["macro"]["rmse"]
    raise ValueError(f"Unknown selection metric: {selection}")


def choose_checkpoint(history, selection):
    """Select only from validation history; tie-break prefers fewer rounds."""
    if not history:
        raise ValueError("Empty validation history")
    best = min(history, key=lambda item: (metric_for_selection(item["validation"], selection), item["round"]))
    return best


def _cached_data(splits):
    train = {}
    for name, part in splits.items():
        x = public_transform(part.train)
        y = part.train[TARGET_COL].to_numpy(dtype=np.float64)
        train[name] = (x, y)
    return train


def _zero_model():
    return np.zeros(N_FEATURES, dtype=np.float64), 0.0


def fit_centralized_checkpoints(splits, checkpoints, local_epochs, seed, lr, alpha, selection):
    train = _cached_data(splits)
    X = np.vstack([x for x, _ in train.values()])
    y = np.concatenate([y for _, y in train.values()])
    w, b = _zero_model()
    rng = np.random.default_rng(seed)
    history, snapshots = [], {}
    training_s, prev = 0.0, 0
    for rnd in checkpoints:
        start = time.perf_counter()
        w, b = gd_update(X, y, w, b, epochs=(rnd - prev) * local_epochs,
                         lr=lr, alpha=alpha, rng=rng)
        training_s += time.perf_counter() - start
        prev = rnd
        validation = evaluate_by_pond(splits, (w, b), "val")
        history.append({"round": rnd, "validation": validation,
                        "training_time_sec": training_s})
        snapshots[rnd] = (w.copy(), float(b))
    selected = choose_checkpoint(history, selection)
    r = selected["round"]
    return {
        "models": snapshots[r], "selected_round": r,
        "validation": selected["validation"], "validation_history": history,
        "time_sec": selected["training_time_sec"],
        "training_epochs": r * local_epochs,
        "comm_bytes": int(len(y) * (N_FEATURES + 1) * 8),
        "n_messages": len(train),
    }


def fit_local_checkpoints(splits, checkpoints, local_epochs, seed, lr, alpha, selection):
    train = _cached_data(splits)
    models = {name: _zero_model() for name in train}
    rngs = {name: np.random.default_rng(seed + i) for i, name in enumerate(train)}
    history, snapshots = [], {}
    training_s, prev = 0.0, 0
    for rnd in checkpoints:
        start = time.perf_counter()
        for name, (X, y) in train.items():
            w, b = models[name]
            models[name] = gd_update(X, y, w, b, epochs=(rnd - prev) * local_epochs,
                                     lr=lr, alpha=alpha, rng=rngs[name])
        training_s += time.perf_counter() - start
        prev = rnd
        validation = evaluate_by_pond(splits, models, "val")
        history.append({"round": rnd, "validation": validation,
                        "training_time_sec": training_s})
        snapshots[rnd] = {name:(w.copy(),float(b)) for name,(w,b) in models.items()}
    selected = choose_checkpoint(history, selection)
    r = selected["round"]
    return {
        "models": snapshots[r], "selected_round": r,
        "validation": selected["validation"], "validation_history": history,
        "time_sec": selected["training_time_sec"],
        "training_epochs": r * local_epochs,
        "comm_bytes": 0, "n_messages": 0,
    }


def _fed_single_round(train, current, rng, local_epochs, lr, alpha, mask_rng=None):
    """One simulated round; masked mode keeps the same client SGD schedule."""
    w_global, b_global = current
    total = sum(len(y) for _, y in train.values())
    weighted = []
    for X, y in train.values():
        w, b = gd_update(X, y, w_global.copy(), b_global,
                         epochs=local_epochs, lr=lr, alpha=alpha, rng=rng)
        weighted.append((len(y) / total) * get_params(w, b))
    plain = np.sum(weighted, axis=0)
    if mask_rng is None:
        return split_params(plain), 0.0, 0.0
    start = time.perf_counter()
    masked = [vec.copy() for vec in weighted]
    for i in range(len(masked)):
        for j in range(i + 1, len(masked)):
            mask = mask_rng.normal(0, 10, size=N_FEATURES + 1)
            masked[i] += mask
            masked[j] -= mask
    aggregated = np.sum(masked, axis=0)
    overhead = time.perf_counter() - start
    error = float(np.max(np.abs(plain - aggregated)))
    return split_params(aggregated), overhead, error


def fit_fed_checkpoints(splits, checkpoints, local_epochs, seed, lr, alpha, selection):
    train = _cached_data(splits)
    current = _zero_model()
    rng = np.random.default_rng(seed)
    history, snapshots = [], {}
    training_s, last = 0.0, 0
    chosen_rounds = set(checkpoints)
    for rnd in range(1, max(checkpoints) + 1):
        start = time.perf_counter()
        current, _, _ = _fed_single_round(train, current, rng, local_epochs, lr, alpha)
        training_s += time.perf_counter() - start
        if rnd in chosen_rounds:
            validation = evaluate_by_pond(splits, current, "val")
            history.append({"round": rnd, "validation": validation,
                            "training_time_sec": training_s})
            snapshots[rnd] = (current[0].copy(), float(current[1]))
            last = rnd
    selected = choose_checkpoint(history, selection)
    r = selected["round"]
    n_clients = len(train)
    base_msg = 2 * n_clients * r
    return {
        "models": snapshots[r], "selected_round": r,
        "validation": selected["validation"], "validation_history": history,
        "time_sec": selected["training_time_sec"],
        "training_epochs": r * local_epochs,
        "comm_bytes": base_msg * PARAM_BYTES, "n_messages": base_msg,
    }


def fit_mask_emulation(splits, rounds, local_epochs, seed, lr, alpha):
    """Replay FedAvg training with masks, using round count chosen by plain FedAvg.

    No validation/test is accessed in this function. Test is evaluated once
    by run_seed after returning. Pairwise secrets are simulated centrally.
    """
    train = _cached_data(splits)
    current = _zero_model()
    rng = np.random.default_rng(seed)
    training_s, mask_s, max_error = 0.0, 0.0, 0.0
    for rnd in range(1, rounds + 1):
        # Simulated pairwise masks have a fixed per-round PRNG seed.
        mask_rng = np.random.default_rng(seed * 1_000_003 + rnd)
        start = time.perf_counter()
        current, overhead, error = _fed_single_round(train, current, rng,
                                                      local_epochs, lr, alpha, mask_rng)
        training_s += time.perf_counter() - start
        mask_s += overhead
        max_error = max(max_error, error)
    n = len(train)
    messages = 2 * n * rounds
    hypothetical_handshakes = rounds * n * (n - 1) // 2
    return {
        "models": current, "selected_round": rounds,
        "time_sec": training_s, "mask_compute_sec": mask_s,
        "max_aggregate_error": max_error,
        "training_epochs": rounds * local_epochs,
        "n_messages": messages + hypothetical_handshakes,
        "comm_bytes": messages * PARAM_BYTES + hypothetical_handshakes * 16,
        "handshake_messages_estimate": hypothetical_handshakes,
        "handshake_bytes_estimate": hypothetical_handshakes * 16,
    }


def _to_public_result(fitted):
    """Avoid putting weight arrays in JSON; retain full validation histories."""
    return {key:value for key,value in fitted.items() if key != "models"}


def run_seed(seed, args):
    splits = load_temporal_splits(args.sample_cap, seed, args.dataset_dir)
    print(f"\nSeed {seed}: {len(splits)} ponds, training cap {args.sample_cap}")
    for name, part in splits.items():
        info = part.summary
        note = " [small validation/test horizon]" if info["val_days"] < 3 or info["test_days"] < 3 else ""
        print(f"  {name:<11} days {info['train_days']:>3}/{info['val_days']:>3}/{info['test_days']:>3}"
              f" | rows {info['train_rows']:>5}/{info['val_rows']:>6}/{info['test_rows']:>6}{note}")
    methods = {}
    checkpoints = args.rounds
    print("  [1/4] Centralized: choose epochs via validation...")
    central = fit_centralized_checkpoints(splits, checkpoints, args.local_epochs,
                                         seed, args.lr, args.alpha, args.selection)
    print("  [2/4] Local: choose epochs via validation...")
    local = fit_local_checkpoints(splits, checkpoints, args.local_epochs,
                                 seed, args.lr, args.alpha, args.selection)
    print("  [3/4] FedAvg: choose rounds via validation...")
    federated = fit_fed_checkpoints(splits, checkpoints, args.local_epochs,
                                   seed, args.lr, args.alpha, args.selection)
    print(f"  [4/4] Pairwise masking emulation at FedAvg's selected {federated['selected_round']} rounds...")
    masked = fit_mask_emulation(splits, federated["selected_round"], args.local_epochs,
                                seed, args.lr, args.alpha)
    # Deliberately compute TEST only here, once after every selection is frozen.
    fitted = {"centralized":central, "local":local,
              "federated":federated, "mask_emulation":masked}
    for name, result in fitted.items():
        methods[name] = _to_public_result(result)
        methods[name]["metrics"] = evaluate_by_pond(splits, result["models"], "test")
        if name == "mask_emulation":
            methods[name]["selection_source"] = "federated_validation_only"
            methods[name]["validation"] = federated["validation"]
    difference = float(np.max(np.abs(get_params(*federated["models"]) - get_params(*masked["models"]))))
    methods["mask_emulation"]["max_weight_difference_from_fedavg"] = difference
    out = {
        "_meta": {
            "protocol_version":"v3", "seed":seed, "n_clients":len(splits),
            "round_candidates":checkpoints, "local_epochs":args.local_epochs,
            "lr":args.lr, "alpha":args.alpha, "selection_metric":args.selection,
            "selection_criterion":"minimum validation RMSE; tie: fewer rounds",
            "test_policy":"test is evaluated only after model selection; identical test per seed for all methods",
            "split":"per-pond 70/15/15 by whole local calendar days, after filtering",
            "preprocessing":"fixed public normalizer (not fitted on held-out data)",
            "train_cap_only":args.sample_cap,
            "local_baseline":"separate pond-specific model, one global epoch budget chosen on aggregate validation",
            "training_budget":"central/local train r*local_epochs; FedAvg trains r rounds of local_epochs each; not compute-matched",
            "target":"simulated feed_ration_g label, NOT measured feeding",
            "security":"pairwise mask numerical emulation, NOT cryptographic SMPC",
            "communication":"analytical bytes/messages, not actual network traffic",
            "time":"local sequential training only; excludes validation/test evaluation and I/O; no network latency",
            "ponds": {name:p.summary for name,p in splits.items()},
            "n_val_rows": sum(len(p.val) for p in splits.values()),
            "n_test_rows": sum(len(p.test) for p in splits.values()),
        },
        "methods":methods,
    }
    print("  Method              selected rounds  VAL RMSE    TEST RMSE")
    for name, result in methods.items():
        print(f"  {name:<20}{result['selected_round']:>6}"
              f"       {metric_for_selection(result['validation'],args.selection):>8.4f}"
              f"    {result['metrics']['micro']['rmse']:>9.4f}")
    return out


def _summary(results):
    """Mean/SD over seeds; descriptive only (same test windows, not independent)."""
    keys = ("centralized", "local", "federated", "mask_emulation")
    report = {}
    for key in keys:
        per_seed = [res["methods"][key]["metrics"]["micro"]["rmse"] for res in results]
        report[key] = {
            "micro_rmse_mean": float(np.mean(per_seed)),
            "micro_rmse_std_sample": float(np.std(per_seed, ddof=1)) if len(per_seed)>1 else None,
            "selected_rounds": [res["methods"][key]["selected_round"] for res in results],
            "micro_rmse_per_seed": per_seed,
        }
    return {"n_seeds":len(results), "seeds":[r["_meta"]["seed"] for r in results],
            "notes":"Same chronological test windows across seeds; descriptive variability, not independent-window CI.",
            "methods":report}


def _positive_int_list(text, label):
    try:
        values = [int(s.strip()) for s in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label}: comma-separated positive integers required") from exc
    if not values or any(v < 1 for v in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{label}: positive, unique integers required")
    return sorted(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-cap", type=int, default=3000,
                    help="cap on TRAIN rows per pond; val/test kept intact")
    ap.add_argument("--rounds", type=lambda s:_positive_int_list(s,"rounds"), default=[5,10,20,50],
                    help="candidate round counts, e.g. 5,10,20,50 (default)")
    ap.add_argument("--seeds", type=lambda s:_positive_int_list(s,"seeds"), default=[42],
                    help="seed list, e.g. 11,12,13,14,15")
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--selection", choices=("micro_rmse", "macro_rmse"), default="micro_rmse")
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    if args.sample_cap < 1 or args.local_epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.alpha) or args.alpha < 0:
        ap.error("sample-cap/local-epochs must be > 0; lr finite > 0; alpha finite >= 0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results=[]
    for seed in args.seeds:
        record = run_seed(seed, args)
        path = args.output_dir / f"seed_{seed}_cap{args.sample_cap}_r{'-'.join(map(str,args.rounds))}.json"
        path.write_text(json.dumps(record,indent=2,allow_nan=False),encoding="utf-8")
        print(f"  Saved: {path}")
        results.append(record)
    summary = _summary(results)
    path = args.output_dir / f"summary_cap{args.sample_cap}_r{'-'.join(map(str,args.rounds))}_seeds{'-'.join(map(str,args.seeds))}.json"
    path.write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print("\nMean micro TEST RMSE (across specified seeds):")
    for name,data in summary["methods"].items():
        sd=data["micro_rmse_std_sample"]
        print(f"  {name:<20} {data['micro_rmse_mean']:.4f}" + (f" +/- {sd:.4f}" if sd is not None else ""))
    print(f"Summary saved: {path}")
    print("WARNING: feed_ration_g is synthetic; masking is an emulator, not secure multi-party computation.")


if __name__ == "__main__":
    main()
