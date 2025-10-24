#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness_suite_v3_oos.py  —  CLEAN OOS evaluator (CSV-first)
----------------------------------------------------------------
- Accepts a comma-separated list of **CSV paths** or plain symbols.
- If at least one path ends with ".csv", we assume CSV-only mode and
  DO NOT touch exchanges. Instead, we call the local evaluator
  (theta_eval_hbatch_biquat_max.py) once for all CSVs and then load
  the generated per-bar eval CSVs (eval_h_*.csv).
- Works with the various filename patterns that the evaluator emits
  (e.g., eval_h_BTCUSDT_1hcsv.csv, eval_h_BTCUSDT_1HCSV.csv, etc.).
- Produces an OOS split (default 0.7) and prints/saves corr + hit rate.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# --------------------------- Helpers ---------------------------------

def _safe_key(sym: str) -> str:
    """Stable key for dicts: use uppercase basename without dirs."""
    return Path(sym).name.upper()


def is_csv_path(s: str) -> bool:
    s2 = s.strip()
    return s2.lower().endswith(".csv")


def _find_eval_csv_for(path: str, interval: str) -> str:
    """
    Return the matching eval CSV written by theta_eval_hbatch_biquat_max.py
    for a given input CSV `path`.
    We try multiple filename candidates because the evaluator historically
    emitted several variants.
    """
    p = Path(path)
    base = p.name              # e.g. BTCUSDT_1h.csv
    stem = p.stem              # e.g. BTCUSDT_1h
    stem_upper = stem.upper()
    stem_nodot_upper = base.replace(".", "").upper()  # e.g. BTCUSDT_1HCSV

    candidates = [
        f"eval_h_{stem_upper}.csv",
        f"eval_h_{stem_nodot_upper}.csv",
        f"eval_h_{stem_upper}_{interval}csv.csv",       # legacy
        f"eval_h_{stem_nodot_upper}_{interval}csv.csv", # legacy
        f"eval_h_{stem_upper}csv.csv",                  # very legacy
    ]

    for c in candidates:
        if Path(c).exists():
            return c

    raise FileNotFoundError(
        f"Could not locate eval CSV for '{path}'. Tried:\n" +
        "\n".join(" - "+c for c in candidates)
    )


def _load_eval_csv_any(path: str, interval: str) -> pd.DataFrame:
    """
    Load the per-bar evaluation CSV for a given original input CSV path.
    Ensures canonical columns and basic sanity checks.
    """
    found = _find_eval_csv_for(path, interval)
    df = pd.read_csv(found)

    lower = {c.lower(): c for c in df.columns}
    # normalize expected columns
    need = ["pred_delta", "true_delta", "correct_pred"]
    for col in need:
        if col not in lower:
            raise KeyError(f"Required column '{col}' not found in {found}. "
                           f"Columns present: {list(df.columns)}")
    # rename to canonical
    df = df.rename(columns={lower[c]: c for c in need})

    # ensure numeric
    for col in need:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=need).reset_index(drop=True)
    return df


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    c = np.corrcoef(a, b)
    return float(c[0, 1])


def _hit_rate(correct_pred: np.ndarray) -> float:
    if correct_pred.size == 0:
        return float("nan")
    return float(np.nanmean(correct_pred.astype(float)))


def compute_oos_metrics(df: pd.DataFrame, split_ratio: float, horizon: int) -> Dict[str, float]:
    """
    Compute OOS corr and hit-rate on the *test* slice only.
    - split_at is applied on the cleaned df (i.e., after dropping NaNs).
    - corr_lag_h: correlation of pred_delta with true_delta shifted by +horizon.
    - corr_shuffle_true: correlation after shuffling true_delta (leak sanity).
    """
    n = len(df)
    split_at = int(np.floor(n * split_ratio))
    test = df.iloc[split_at:].reset_index(drop=True)

    pred = test["pred_delta"].to_numpy()
    true = test["true_delta"].to_numpy()
    corr_oos = _corr(pred, true)
    hit_oos = _hit_rate(test["correct_pred"].to_numpy())

    # lag-h correlation: align pred with future true by shifting true forward
    # (for sanity; with proper labeling this should be near 0 on OOS)
    true_lag_h = np.roll(true, -horizon)
    true_lag_h[-horizon:] = np.nan
    mask = ~np.isnan(true_lag_h)
    corr_lag_h = _corr(pred[mask], true_lag_h[mask])

    # shuffled true (should ~0 if no spurious structure)
    rng = np.random.default_rng(12345)
    shuf = true.copy()
    rng.shuffle(shuf)
    corr_shuffle_true = _corr(pred, shuf)

    return {
        "n_total": n,
        "n_test": int(test.shape[0]),
        "split_at": split_at,
        "corr_oos": corr_oos,
        "hit_rate_oos": hit_oos,
        "corr_lag_h": corr_lag_h,
        "corr_shuffle_true": corr_shuffle_true,
    }


# --------------------------- Main ------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True,
                    help="Comma-separated list of CSV paths or symbols. "
                         "If any item ends with .csv, CSV-only mode is used.")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--minP", type=int, default=24)
    ap.add_argument("--maxP", type=int, default=480)
    ap.add_argument("--nP", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--lam", "--lambda", dest="lam", type=float, default=1e-3)
    ap.add_argument("--pred-ensemble", choices=["avg", "max"], default="avg")
    ap.add_argument("--max-by", choices=["transform", "contrib"], default="transform")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--oos-split", type=float, default=0.7,
                    help="Train/test split ratio. OOS is the tail (1 - split).")
    ap.add_argument("--out", required=True, help="Output CSV for OOS summary")
    args = ap.parse_args()

    symbols_raw = [s.strip() for s in args.symbols.split(",") if s.strip()]
    csv_mode = any(is_csv_path(s) for s in symbols_raw)

    # Resolve script dir for evaluator
    script_dir = Path(__file__).resolve().parent
    evaluator = script_dir / "theta_eval_hbatch_biquat_max.py"
    if not evaluator.exists():
        print("theta_eval_hbatch_biquat_max.py not found in the same directory. "
              "Place this script next to the evaluator, or adjust the path.", file=sys.stderr)
        sys.exit(1)

    # If CSV mode -> run evaluator once for all given CSVs (so that eval_h_*.csv are created)
    if csv_mode:
        # Keep only CSV paths (ignore plain symbols if user mixed them)
        csvs = [s for s in symbols_raw if is_csv_path(s)]
        if not csvs:
            print("[error] CSV mode triggered but no CSVs found.", file=sys.stderr)
            sys.exit(2)

        cmd = [
            sys.executable, str(evaluator),
            "--symbols", ",".join(csvs),
            "--interval", args.interval,
            "--window", str(args.window),
            "--horizon", str(args.horizon),
            "--minP", str(args.minP),
            "--maxP", str(args.maxP),
            "--nP", str(args.nP),
            "--sigma", str(args.sigma),
            "--lambda", str(args.lam),
            "--pred-ensemble", args.pred_ensemble,
            "--max-by", args.max_by,
            "--out", "results/hbatch_biquat_summary.csv",
        ]
        os.makedirs("results", exist_ok=True)
        print("\n=== Spouštím evaluátor ===")
        print(">>", " ".join(cmd))
        subprocess.run(cmd, check=True)
        work_items = csvs
    else:
        print("[warn] Non-CSV symbols are not supported in this OOS standalone right now.")
        sys.exit(3)

    # Build OOS rows
    rows = []
    for src in work_items:
        try:
            df_eval = _load_eval_csv_any(src, args.interval)
            metrics = compute_oos_metrics(df_eval, args.oos_split, args.horizon)
            rows.append({
                "symbol": _safe_key(src),
                "idx_ok": True,
                **metrics,
            })
            print(f"[OOS] {_safe_key(src)}: corr={metrics['corr_oos']:.3f}, "
                  f"hit={metrics['hit_rate_oos']:.3f}, n_test={metrics['n_test']}")
        except Exception as e:
            rows.append({
                "symbol": _safe_key(src),
                "idx_ok": False,
                "error": str(e),
            })
            print(f"[OOS] {_safe_key(src)}: ERROR -> {e}", file=sys.stderr)

    # Save summary
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, index=False)
    print(f"\nSaved OOS summary -> {out}")


if __name__ == "__main__":
    main()
