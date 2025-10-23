#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness_suite_v2.py
- Fixes the leakage test to account for overlap of windows:
  Uses LAG = horizon (non-overlapping) instead of +1 bar.
- Also reports autocorrelation of true_delta at lag=1 and lag=h
  to contextualize any residual corr.
"""
import argparse, os, sys, subprocess
import pandas as pd
import numpy as np
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True, type=str)
    p.add_argument("--interval", default="1h", type=str)
    p.add_argument("--window", "--W", dest="W", default=256, type=int)
    p.add_argument("--horizon", default=4, type=int)
    p.add_argument("--minP", default=24, type=int)
    p.add_argument("--maxP", default=480, type=int)
    p.add_argument("--nP", default=8, type=int)
    p.add_argument("--sigma", default=0.8, type=float)
    p.add_argument("--lambda", dest="lam", default=1e-3, type=float)
    p.add_argument("--limit", default=2000, type=int)
    p.add_argument("--pred-ensemble", dest="pred_ensemble", choices=["avg","max"], default="avg")
    p.add_argument("--max-by", dest="max_by", choices=["transform","contrib"], default="transform")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--dense-np", default=None, type=int)
    p.add_argument("--out", default="robustness_report_v2.csv", type=str)
    return p.parse_args()

def ensure_prices_csv(symbols, interval, limit):
    cmd = [sys.executable, "make_prices_csv.py",
           "--symbols", ",".join(symbols),
           "--interval", interval,
           "--limit", str(int(max(1000, limit))),
           "--outdir", "prices"]
    subprocess.run(cmd, check=True)

def find_new_eval():
    for c in ["biquat_max_standalone/theta_eval_hbatch_biquat_max.py",
              "theta_eval_hbatch_biquat_max.py"]:
        if os.path.exists(c):
            return c
    raise SystemExit("theta_eval_hbatch_biquat_max.py not found.")

def run_eval(symbols, interval, W, horizon, minP, maxP, nP, sigma, lam, pred_ensemble, max_by, suffix):
    eval_path = find_new_eval()
    csv_paths = [os.path.abspath(f"prices/{s}_{interval}.csv") for s in symbols]
    out_sum = f"hbatch_biquat_summary{suffix}.csv"
    cmd = [sys.executable, eval_path,
           "--symbols", ",".join(csv_paths),
           "--interval", interval,
           "--window", str(W),
           "--horizon", str(horizon),
           "--minP", str(int(minP)),
           "--maxP", str(int(maxP)),
           "--nP", str(int(nP)),
           "--sigma", str(sigma),
           "--lambda", str(lam),
           "--pred-ensemble", pred_ensemble,
           "--max-by", max_by,
           "--out", out_sum]
    subprocess.run(cmd, check=True)
    perbar = { s: f"eval_h_{s}_{interval.replace('/','') + 'csv'}.csv" for s in symbols }
    return out_sum, perbar

def read_perbar(path):
    df = pd.read_csv(path)
    # normalize names
    cols = {c.lower(): c for c in df.columns}
    def need(n):
        for k,v in cols.items():
            if k == n: return v
        raise KeyError(f"Missing `{n}` in {path}")
    for k in ["entry_idx","compare_idx"]:
        df[need(k)] = df[need(k)].astype(int)
    for k in ["pred_delta","true_delta"]:
        df[need(k)] = pd.to_numeric(df[need(k)], errors="coerce")
    return df, need

def safe_corr(a,b):
    if len(a) < 3: return 0.0
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0: return 0.0
    return float(np.corrcoef(a,b)[0,1])

def diagnostics(symbol, path, horizon, seed=42):
    np.random.seed(seed)
    df, need = read_perbar(path)
    # integrity
    idx_ok = bool(((df[need("compare_idx")] - df[need("entry_idx")]) == horizon).all())
    # arrays
    pred = df[need("pred_delta")].to_numpy()
    true = df[need("true_delta")].to_numpy()
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]
    n = len(true)
    # base corr
    corr_base = safe_corr(pred, true)
    # context: autocorr of true at lag 1 and h
    def autocorr(x, k):
        if k >= len(x): return 0.0
        return safe_corr(x[:-k], x[k:])
    ac_true_lag1 = autocorr(true, 1)
    ac_true_lagh = autocorr(true, horizon)
    # leakage test: non-overlap lag = horizon
    if horizon < n:
        corr_lagh = safe_corr(pred[:-horizon], true[horizon:])
    else:
        corr_lagh = 0.0
    # shuffle sanity
    true_shuf = true.copy(); np.random.shuffle(true_shuf)
    corr_shuffle_true = safe_corr(pred, true_shuf)
    leak_flag = (abs(corr_lagh) > max(0.1, abs(ac_true_lagh)))  # must not exceed natural autocorr at same lag
    return {
        "symbol": symbol,
        "corr_base": corr_base,
        "corr_lag_h": corr_lagh,
        "ac_true_lag1": ac_true_lag1,
        "ac_true_lag_h": ac_true_lagh,
        "corr_shuffle_true": corr_shuffle_true,
        "leak_flag": bool(leak_flag),
        "idx_ok": idx_ok,
        "count": int(n),
    }

def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    ensure_prices_csv(symbols, args.interval, args.limit)
    # baseline
    _, perbar = run_eval(symbols, args.interval, args.W, args.horizon,
                         args.minP, args.maxP, args.nP, args.sigma, args.lam,
                         args.pred_ensemble, args.max_by, suffix="")
    rows = [diagnostics(s, perbar[s], args.horizon, args.seed) for s in symbols]
    df = pd.DataFrame(rows)
    # dense optional
    if args.dense_np:
        _, perbar2 = run_eval(symbols, args.interval, args.W, args.horizon,
                              args.minP, args.maxP, args.dense_np, args.sigma, args.lam,
                              args.pred_ensemble, args.max_by, suffix=f"_dense{args.dense_np}")
        rows2 = [diagnostics(s, perbar2[s], args.horizon, args.seed+1) for s in symbols]
        df2 = pd.DataFrame(rows2)
        df2 = df2.add_prefix("dense_")
        df = pd.concat([df, df2], axis=1)
        if "dense_corr_base" in df.columns:
            df["delta_corr_dense"] = df["dense_corr_base"] - df["corr_base"]
    out = Path(args.out)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
