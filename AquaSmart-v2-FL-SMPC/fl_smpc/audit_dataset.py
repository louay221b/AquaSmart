"""Read-only audit of the original IoTPond sensor CSVs.

Run from the AquaSmart-v2-FL-SMPC directory:
    python -m fl_smpc.audit_dataset

Writes JSON and Markdown diagnostics to fl_smpc/results/.
No source files are changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fl_smpc.data import _normalise_columns

ROOT = Path(__file__).resolve().parents[1]
CORE = ["temperature", "dissolved_oxygen", "ph", "turbidity",
        "ammonia", "fish_weight_g"]
NUMERIC = CORE + ["nitrate"]


def count(mask):
    return int(mask.fillna(False).sum())


def finite_or_none(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def audit_pond(path: Path):
    raw = pd.read_csv(path, low_memory=False)
    df = _normalise_columns(raw)
    absent = [name for name in CORE if name not in df.columns]
    for name in NUMERIC:
        if name not in df:
            df[name] = np.nan
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df[NUMERIC] = df[NUMERIC].replace([np.inf, -np.inf], np.nan)

    missing_core = df[CORE].isna().any(axis=1)
    weight_ok = df["fish_weight_g"].gt(0) & df["fish_weight_g"].lt(2000)
    temp_ok = df["temperature"].between(5, 40)
    ph_ok = df["ph"].between(3, 11)
    retained = ~missing_core & weight_ok & temp_ok & ph_ok

    ts_info = {"available": False, "valid": 0, "duplicates": None,
               "out_of_order": None, "median_interval_s": None}
    # Some CSVs use created_at; IoTPond8 stores ISO timestamps in its
    # unnamed first column. Use the first field only when it resembles
    # YYYY-MM-DD. Report pond-local wall-clock times without inferring UTC.
    ts_col = "created_at" if "created_at" in raw.columns else None
    if ts_col is None and len(raw):
        first = str(raw.iloc[0, 0])
        if len(first) >= 10 and first[4:5] == "-" and first[7:8] == "-":
            ts_col = raw.columns[0]
    if ts_col is not None:
        raw_ts = (raw[ts_col].astype(str).str.slice(0, 19)
                  .str.replace("T", " ", regex=False))
        parsed = pd.to_datetime(raw_ts,
                                format="%Y-%m-%d %H:%M:%S",
                                errors="coerce")
        delta = parsed.diff().dt.total_seconds()
        positive = delta[delta > 0]
        ts_info = {
            "available": True,
            "valid": int(parsed.notna().sum()),
            "duplicates": int(parsed[parsed.notna()].duplicated().sum()),
            "out_of_order": int((delta < 0).sum()),
            "median_interval_s": finite_or_none(positive.median()),
        }

    flags = {
        "temperature_outside_5_40": count(~temp_ok & df["temperature"].notna()),
        "ph_outside_3_11": count(~ph_ok & df["ph"].notna()),
        "do_negative": count(df["dissolved_oxygen"] < 0),
        "do_above_20_review_units": count(df["dissolved_oxygen"] > 20),
        "ammonia_above_5_currently_clipped": count(df["ammonia"] > 5),
        "nitrate_above_100_currently_clipped": count(df["nitrate"] > 100),
        "turbidity_above_500_currently_clipped": count(df["turbidity"] > 500),
        "nitrate_missing": int(df["nitrate"].isna().sum()),
    }
    feed_cols = [name for name in raw.columns
                 if any(t in name.lower() for t in
                        ("feed", "ration", "pellet", "intake"))]
    return {
        "pond": path.stem,
        "raw_columns": list(raw.columns),
        "raw_rows": len(raw),
        "missing_required_columns": absent,
        "missing_core_rows": count(missing_core),
        "invalid_weight_rows": count(~weight_ok),
        "rows_after_existing_filters": int(retained.sum()),
        "rows_removed_by_existing_filters": int((~retained).sum()),
        "unique_fish_weights": int(df["fish_weight_g"].nunique(dropna=True)),
        "duplicate_measurement_vectors": int(df.duplicated(subset=NUMERIC).sum()),
        "measured_feed_columns": feed_cols,
        "timestamp": ts_info,
        "flags": flags,
    }


def markdown_report(result):
    lines = [
        "# AquaSmart - Dataset quality audit", "",
        "Automatically generated from unmodified IoTPond CSV files.", "",
        "The source CSVs hold environmental and fish measurements, not",
        "measured feed intake. The current feed_ration_g label is simulated",
        "from a fixed heuristic plus noise (see fl_smpc/data.py). Therefore",
        "existing performance scores measure recovery of a synthetic rule.", "",
        "Ponds audited: %d. Raw rows: %d. Retained rows: %d." % (
            result["n_ponds"], result["raw_rows"], result["retained_rows"]), "",
        "| Pond | Raw | Retained | Missing core | Ammonia >5 | DO >20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p in result["ponds"]:
        f = p["flags"]
        lines.append("| %s | %d | %d | %d | %d | %d |" % (
            p["pond"], p["raw_rows"], p["rows_after_existing_filters"],
            p["missing_core_rows"], f["ammonia_above_5_currently_clipped"],
            f["do_above_20_review_units"]))
    lines += [
        "", "## Methodological findings", "",
        "1. Original created_at timestamps are currently dropped during feature",
        "   preparation. Use chronological splits to test future prediction.",
        "2. Current centralized and per-client experiments use independently",
        "   generated random splits, so train/test overlap is possible.",
        "3. Current shared RobustScaler is fitted on pooled rows before the",
        "   held-out split; it accesses test observations and every client.",
        "4. Review flag counts and sensor units before clipping/removing outliers.",
        "5. Each pond is a simulated FL client, not necessarily a separate farm.",
        "6. The label is heuristic-generated, not measured optimal feeding.",
        "", "Source dataset:",
        "https://www.kaggle.com/datasets/ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets",
        "", "Threshold flags are diagnostics, not proof of invalid biological states.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "fl_smpc" / "results")
    args = parser.parse_args()
    paths = sorted(args.dataset_dir.glob("*.csv"))
    if not paths:
        parser.error("No CSV files found in " + str(args.dataset_dir))

    ponds = []
    for path in paths:
        p = audit_pond(path)
        ponds.append(p)
        print("%s: %d raw, %d after filters" % (
            p["pond"], p["raw_rows"], p["rows_after_existing_filters"]))

    summary = {
        "n_ponds": len(ponds),
        "raw_rows": sum(p["raw_rows"] for p in ponds),
        "retained_rows": sum(p["rows_after_existing_filters"] for p in ponds),
        "measured_feed_label_available": any(p["measured_feed_columns"]
                                             for p in ponds),
        "target_is_heuristic_generated": True,
        "ponds": ponds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jpath = args.output_dir / "data_audit.json"
    mpath = args.output_dir / "data_audit.md"
    jpath.write_text(json.dumps(summary, indent=2, allow_nan=False),
                     encoding="utf-8")
    mpath.write_text(markdown_report(summary), encoding="utf-8")
    print("Audit saved: %s and %s" % (jpath, mpath))


if __name__ == "__main__":
    main()
