"""Leakage-aware, day-grouped per-pond splits for the AquaSmart proxy-label benchmark.

This module is ADDITIVE: it does not change the original ML/FL scripts,
original CSVs, or original experiment results. Timestamp and label caveats
are recorded in the result metadata by the runner.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fl_smpc.data import FEATURE_COLS, TARGET_COL, _normalise_columns, _engineer, _derive_feed_target

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["temperature", "dissolved_oxygen", "ph", "turbidity", "ammonia", "fish_weight_g"]
NUMERIC = REQUIRED + ["nitrate"]
# Public, fixed calibration parameters; no pooled client data required.
# Deterministic feature clipping to [-10, 10] avoids domination by sensor anomalies.
PUBLIC_CENTER = dict(zip(FEATURE_COLS, [
    25, 6, 7, 30, 0.2, 10, 100, 0, 0, 0, 0, 0, 0,
    0, 0, 60, 5, 150, 40, 750, 625, 2,
]))
PUBLIC_SCALE = dict(zip(FEATURE_COLS, [
    10, 5, 2, 100, 1, 20, 200, 10, 5, 1, 2, 1, 1,
    1, 1, 40, 10, 200, 50, 5000, 600, 2,
]))
assert len(FEATURE_COLS) == len(PUBLIC_CENTER) == len(PUBLIC_SCALE)


@dataclass
class PondSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    summary: dict


def _timestamp_column(raw: pd.DataFrame) -> str:
    if "created_at" in raw.columns:
        return "created_at"
    if len(raw) and len(raw.columns):
        first = str(raw.iloc[0, 0])
        if re.match(r"^\d{4}-\d{2}-\d{2}", first):
            return str(raw.columns[0])
    raise ValueError("Aucune colonne temporelle reconnue: created_at ou premiere colonne ISO")


def parse_calendar_days(s: pd.Series) -> pd.Series:
    """Parse mixed local display dates; retain only the calendar day.

    IMPORTANT: CET/CEST/UTC/offset information is stripped, NOT converted.
    This is suitable for pond-local DAY grouping, not UTC time comparisons.
    """
    text = s.astype("string").str.strip()
    text = text.str.replace(
        r"(?:\s*(?:CET|CEST|UTC|GMT|Z|[+-]\d{2}:?\d{2}))+$",
        "", regex=True, case=False,
    ).str.replace("T", " ", regex=False)
    parsed = pd.to_datetime(text, format="mixed", dayfirst=True, errors="coerce")
    return parsed.dt.normalize()


def public_transform(df: pd.DataFrame) -> np.ndarray:
    x = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    center = np.array([PUBLIC_CENTER[k] for k in FEATURE_COLS], dtype=np.float64)
    scale = np.array([PUBLIC_SCALE[k] for k in FEATURE_COLS], dtype=np.float64)
    out = np.clip((x - center) / scale, -10, 10)
    if not np.isfinite(out).all():
        raise ValueError("NaN ou inf dans les features apres transformation")
    return out


def _partition(days: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    unique = np.sort(days.dropna().unique())
    n = len(unique)
    if n < 3:
        raise ValueError(f"Seulement {n} jour(s): impossible de creer Train/Val/Test disjoints")
    n_train = max(1, int(np.floor(n * 0.70)))
    n_val = max(1, int(np.floor(n * 0.15)))
    if n - n_train - n_val < 1:
        n_val = 1
        n_train = n - 2
    train_days = unique[:n_train]
    val_days = unique[n_train:n_train+n_val]
    test_days = unique[n_train+n_val:]
    t = days.isin(train_days).to_numpy()
    v = days.isin(val_days).to_numpy()
    e = days.isin(test_days).to_numpy()
    if not ((t | v | e).all() and not (t & v).any() and not (t & e).any() and not (v & e).any()):
        raise AssertionError("Partition temporelle invalide")
    info = {"n_days": n, "train_days": len(train_days), "val_days": len(val_days),
            "test_days": len(test_days), "train_last_day": str(train_days[-1])[:10],
            "val_first_day": str(val_days[0])[:10], "val_last_day": str(val_days[-1])[:10],
            "test_first_day": str(test_days[0])[:10]}
    return t, v, e, info


def load_temporal_splits(sample_cap: int = 5000, seed: int = 42,
                         dataset_dir: str | Path | None = None) -> dict[str, PondSplit]:
    """Split full filtered data by distinct days FIRST, cap TRAIN only.

    The SAME returned splits are used for all methods. The simulated label
    is generated once per pond before the split, independent of method.
    """
    root = Path(dataset_dir) if dataset_dir is not None else ROOT / "dataset"
    paths = sorted(root.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Aucun fichier CSV dans {root}")
    clients = {}
    for i, path in enumerate(paths):
        raw = pd.read_csv(path, low_memory=False)
        day = parse_calendar_days(raw[_timestamp_column(raw)])
        df = _normalise_columns(raw)
        for col in NUMERIC:
            if col not in df:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[NUMERIC] = df[NUMERIC].replace([np.inf, -np.inf], np.nan)
        keep = (df[REQUIRED].notna().all(axis=1)
                & df["fish_weight_g"].gt(0) & df["fish_weight_g"].lt(2000)
                & df["temperature"].between(5, 40)
                & df["ph"].between(3, 11)
                & day.notna())
        dropped_timestamp_rows = int((day.isna() & df[REQUIRED].notna().all(axis=1)).sum())
        df = df.loc[keep].copy()
        day = day.loc[keep].copy()
        if df.empty:
            raise ValueError(f"{path.stem}: aucune ligne exploitable")
        # Domain bounds are fixed, not inferred from validation/test data.
        df["ammonia"] = df["ammonia"].clip(0, 5)
        df["nitrate"] = df["nitrate"].fillna(2.0).clip(0, 100)
        df["turbidity"] = df["turbidity"].clip(0, 500)
        features = _engineer(df).reset_index(drop=True)
        day = day.reset_index(drop=True)
        # Deterministic synthetic proxy target, explicitly NOT real feed.
        features[TARGET_COL] = _derive_feed_target(features, np.random.default_rng(10_000 + i))
        train_mask, val_mask, test_mask, info = _partition(day)
        train = features.loc[train_mask].copy()
        val = features.loc[val_mask].copy()
        test = features.loc[test_mask].copy()
        uncapped_train_n = len(train)
        if sample_cap > 0 and len(train) > sample_cap:
            train = train.sample(n=sample_cap, random_state=seed + i).sort_index()
        summary = {"pond": path.stem, "raw_rows": len(raw), "filtered_rows": len(features),
                   "timestamp_unusable_core_complete": dropped_timestamp_rows,
                   "train_rows_before_cap": uncapped_train_n,
                   "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
                   **info}
        if min(len(train), len(val), len(test)) < 1:
            raise AssertionError(f"{path.stem}: partition vide")
        clients[path.stem] = PondSplit(train.reset_index(drop=True),
                                       val.reset_index(drop=True),
                                       test.reset_index(drop=True), summary)
    return clients
