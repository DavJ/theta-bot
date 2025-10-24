#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness_suite_v3_oos.py  —  CLEAN OOS evaluator (CSV-first, with auto correct_pred)
----------------------------------------------------------------
Same as v3, but now auto-computes 'correct_pred' if missing from eval CSVs.
"""

import argparse
import os
import sys
import subprocess

# === evaluator autodetect ===
from pathlib import Path as _P
_EVAL_CANDS = [
    _P(__file__).with_name('theta_eval_hbatch_biquat_max.py'),
    _P(__file__).parent / 'theta_bot_averaging' / 'theta_eval_hbatch_biquat_max.py',
    _P.cwd() / 'theta_eval_hbatch_biquat_max.py',
    _P.cwd() / 'theta_bot_averaging' / 'theta_eval_hbatch_biquat_max.py',
]
for _c in _EVAL_CANDS:
    if _c.exists():
        _EVAL_PY = str(_c.resolve())
        break
else:
    raise FileNotFoundError(
        'Evaluator theta_eval_hbatch_biquat_max.py not found in any expected location: '
        + ', '.join(map(str, _EVAL_CANDS))
    )

# === evaluator path
from pathlib import Path as _P

# robustní dohledání evaluátoru
_EVAL_CANDS = [
    _P(__file__).with_name("theta_eval_hbatch_biquat_max.py"),
    _P(__file__).parent / "theta_bot_averaging" / "theta_eval_hbatch_biquat_max.py",
    _P.cwd() / "theta_eval_hbatch_biquat_max.py",
    _P.cwd() / "theta_bot_averaging" / "theta_eval_hbatch_biquat_max.py",
]
for _c in _EVAL_CANDS:
    if _c.exists():
        _EVAL_PY = str(_c)
        break
else:
    raise FileNotFoundError("theta_eval_hbatch_biquat_max.py not found. Tried: " + ", ".join(map(str, _EVAL_CANDS)))

from pathlib import Path
import numpy as np
import pandas as pd


def _safe_key(sym: str) -> str:
    return Path(sym).name.upper()


def is_csv_path(s: str) -> bool:
    return s.strip().lower().endswith(".csv")


def _find_eval_csv_for(path: str, interval: str) -> str:
    p = Path(path)
    base = p.name
    stem = p.stem
    stem_upper = stem.upper()
    stem_nodot_upper = base.replace(".", "").upper()
    candidates = [
        f"eval_h_{stem_upper}.csv",
        f"eval_h_{stem_nodot_upper}.csv",
        f"eval_h_{stem_upper}_{interval}csv.csv",
        f"eval_h_{stem_nodot_upper}_{interval}csv.csv",
        f"eval_h_{stem_upper}csv.csv",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError(f"No eval CSV for {path}. Tried: {candidates}")


def _load_eval_csv_any(path: str, interval: str) -> pd.DataFrame:
    found = _find_eval_csv_for(path, interval)
    df = pd.read_csv(found)
    lower = {c.lower(): c for c in df.columns}

    # Auto-compute correct_pred if missing
    if 'correct_pred' not in lower:
        if 'pred_delta' in lower and 'true_delta' in lower:
            print(f"[auto] Computing correct_pred for {found}")
            pred_col = lower['pred_delta']
            true_col = lower['true_delta']
            pred_dir = np.sign(df[pred_col].astype(float).values)
            true_dir = np.sign(df[true_col].astype(float).values)
            df['correct_pred'] = (pred_dir == true_dir).astype(int)
        else:
            raise KeyError(f"Missing columns for fallback correct_pred in {found}")

    # Standardize
    for col in ['pred_delta', 'true_delta', 'correct_pred']:
        if col not in lower and col not in df.columns:
            raise KeyError(f"Missing column {col} in {found}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=['pred_delta', 'true_delta', 'correct_pred']).reset_index(drop=True)


def _corr(a, b):
    if len(a) < 2: return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _hit_rate(arr):
    return float(np.nanmean(arr.astype(float))) if len(arr) > 0 else np.nan


def compute_oos_metrics(df, split_ratio, horizon):
    n = len(df)
    split_at = int(np.floor(n * split_ratio))
    test = df.iloc[split_at:].reset_index(drop=True)
    pred, true = test["pred_delta"], test["true_delta"]
    corr_oos = _corr(pred, true)
    hit_oos = _hit_rate(test["correct_pred"])
    true_lag_h = np.roll(true, -horizon)
    true_lag_h[-horizon:] = np.nan
    mask = ~np.isnan(true_lag_h)
    corr_lag_h = _corr(pred[mask], true_lag_h[mask])
    shuf = true.copy().values
    np.random.default_rng(12345).shuffle(shuf)
    corr_shuffle_true = _corr(pred, shuf)
    return dict(n_total=n, n_test=len(test), split_at=split_at,
                corr_oos=corr_oos, hit_rate_oos=hit_oos,
                corr_lag_h=corr_lag_h, corr_shuffle_true=corr_shuffle_true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
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
    ap.add_argument("--oos-split", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    symbols_raw = [s.strip() for s in args.symbols.split(",") if s.strip()]
    csv_mode = any(is_csv_path(s) for s in symbols_raw)
    script_dir = Path(__file__).resolve().parent
    evaluator = script_dir / "theta_eval_hbatch_biquat_max.py"
    if csv_mode:
        csvs = [s for s in symbols_raw if is_csv_path(s)]
        os.makedirs("results", exist_ok=True)
        cmd = [sys.executable, str(evaluator),
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
               "--out", "results/hbatch_biquat_summary.csv"]
        print("\n=== Spouštím evaluátor ===")
        print(">>", " ".join(cmd))
        subprocess.run(cmd, check=True)
        work_items = csvs
    else:
        sys.exit(3)

    rows = []
    for src in work_items:
        try:
            df_eval = _load_eval_csv_any(src, args.interval)
            metrics = compute_oos_metrics(df_eval, args.oos_split, args.horizon)
            rows.append({"symbol": _safe_key(src), **metrics, "ok": True})
            print(f"[OOS] {_safe_key(src)}: corr={metrics['corr_oos']:.3f}, hit={metrics['hit_rate_oos']:.3f}")
        except Exception as e:
            rows.append({"symbol": _safe_key(src), "ok": False, "error": str(e)})
            print(f"[OOS] {_safe_key(src)}: ERROR -> {e}", file=sys.stderr)

    df_out = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    print(f"\nSaved OOS summary -> {out}")


if __name__ == "__main__":
    main()
