#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness_suite_v3_oos.py
- Causal OOS validation for theta_eval_hbatch_biquat_max.py
- Chronological split (train first, test last)
- Reports OOS metrics and lag-h leakage check
"""
import argparse, os, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True, type=str)
    p.add_argument("--interval", default="1h", type=str)
    p.add_argument("--window", "--W", dest="W", default=256, type=int)
    p.add_argument("--horizon", default=4, type=int)
    p.add_argument("--minP", default=24, type=int)
    p.add_argument("--maxP", default=480, type=int)
    p.add_argument("--nP", default=16, type=int)
    p.add_argument("--sigma", default=0.8, type=float)
    p.add_argument("--lambda", dest="lam", default=1e-3, type=float)
    p.add_argument("--limit", default=2000, type=int)
    p.add_argument("--pred-ensemble", dest="pred_ensemble", choices=["avg","max"], default="avg")
    p.add_argument("--max-by", dest="max_by", choices=["transform","contrib"], default="transform")
    p.add_argument("--oos-split", default=0.7, type=float)
    p.add_argument("--out", default="results/robustness_report_v3.csv", type=str)
    return p.parse_args()

def ensure_prices(symbols, interval, limit):
    if os.path.exists("make_prices_csv.py"):
        cmd = [sys.executable, "make_prices_csv.py",
               "--symbols", ",".join(symbols),
               "--interval", interval,
               "--limit", str(int(max(1000, limit))),
               "--outdir", "prices"]
        subprocess.run(cmd, check=True)

def find_eval():
    for c in ["theta_eval_hbatch_biquat_max.py", "biquat_max_standalone/theta_eval_hbatch_biquat_max.py"]:
        if os.path.exists(c):
            return c
    raise SystemExit("theta_eval_hbatch_biquat_max.py not found")

def run_eval(symbols, interval, W, horizon, minP, maxP, nP, sigma, lam, pred_ensemble, max_by):
    eval_path = find_eval()
    csv_candidates = [f"prices/{s}_{interval}.csv" for s in symbols]
    if all(os.path.exists(p) for p in csv_candidates):
        arg_symbols = ",".join(os.path.abspath(p) for p in csv_candidates)
    else:
        arg_symbols = ",".join(symbols)
    out_sum = "results/hbatch_biquat_summary.csv"
    cmd = [sys.executable, eval_path,
           "--symbols", arg_symbols,
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
    return perbar

def safe_corr(a,b):
    if len(a) < 3: return 0.0
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0: return 0.0
    return float(np.corrcoef(a,b)[0,1])

def read_perbar(path):
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    def need(name):
        key = name.lower()
        if key in lower: return lower[key]
        raise KeyError(f"Missing '{name}' in {path}")
    for k in ["entry_idx","compare_idx"]:
        df[need(k)] = df[need(k)].astype(int)
    for k in ["pred_delta","true_delta"]:
        df[need(k)] = pd.to_numeric(df[need(k)], errors="coerce")
    df = df[np.isfinite(df[need('pred_delta')]) & np.isfinite(df[need('true_delta')])]
    return df, need

def autocorr(x, k):
    if k <= 0 or k >= len(x): return 0.0
    return safe_corr(x[:-k], x[k:])

def eval_symbol(symbol, path, horizon, split):
    df, need = read_perbar(path)
    n = len(df)
    idx_ok = bool(((df[need("compare_idx")] - df[need("entry_idx")]) == horizon).all())
    cut = max(1, min(n-1, int(np.floor(n * float(split)))))
    test = df.iloc[cut:].reset_index(drop=True)
    pred = test[need("pred_delta")].to_numpy()
    true = test[need("true_delta")].to_numpy()
    corr_oos = safe_corr(pred, true)
    hit_rate_oos = float((np.sign(pred) == np.sign(true)).mean())
    h = int(horizon)
    if len(test) > h:
        corr_lag_h = safe_corr(pred[:-h], true[h:])
        ac_true_1 = autocorr(true, 1)
        ac_true_h = autocorr(true, h)
        leak_flag = (abs(corr_lag_h) > max(0.1, abs(ac_true_h)))
    else:
        corr_lag_h, ac_true_1, ac_true_h, leak_flag = (np.nan, np.nan, np.nan, False)
    return dict(symbol=symbol, count=n, corr_oos=corr_oos, hit_rate_oos=hit_rate_oos,
                corr_lag_h=corr_lag_h, ac_true_lag1=ac_true_1, ac_true_lag_h=ac_true_h,
                leak_flag=leak_flag, idx_ok=idx_ok)

def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("prices").mkdir(parents=True, exist_ok=True)

    ensure_prices(symbols, args.interval, args.limit)
    perbar = run_eval(symbols, args.interval, args.W, args.horizon,
                      args.minP, args.maxP, args.nP, args.sigma, args.lam,
                      args.pred_ensemble, args.max_by)

    rows = [eval_symbol(s, perbar[s], args.horizon, args.oos_split) for s in symbols]
    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {out}")
if __name__ == "__main__":
    main()
