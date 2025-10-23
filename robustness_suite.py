#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness_suite.py
Leakage/overfitting diagnostics + optional denser period sampling test for BIQUAT MAX evaluator.

What it does:
1) Runs theta_eval_hbatch_biquat_max.py on given symbols (CSV inputs via make_prices_csv.py).
2) Loads per-bar CSVs (eval_h_*.csv) produced by the evaluator.
3) Computes baseline corr(pred_delta, true_delta).
4) Runs three leakage tests:
   - LAG(+1): corr(pred_delta, true_delta.shift(+1))
   - SHUFFLE_TRUE: corr(pred_delta, permuted(true_delta))
   - SHUFFLE_PRED: corr(permuted(pred_delta), true_delta)
   Expectation: All ≈ 0. If not, leakage is suspected.
5) Integrity checks:
   - compare_idx == entry_idx + horizon for all rows
   - time strictly increasing
6) (Optional) Denser psi grid: rerun with higher nP and compare corr.

Outputs:
- robustness_report.csv with columns:
  symbol, corr_base, corr_lag1, corr_shuffle_true, corr_shuffle_pred,
  leak_flag (yes/no), idx_ok, time_monotonic, count
- If --dense-np passed: columns corr_dense, delta_corr_dense
"""
import argparse, os, sys, json, subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser(description="Leakage diagnostics + dense psi test for BIQUAT MAX")
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
    p.add_argument("--dense-np", default=None, type=int, help="If set, rerun with higher nP and report corr improvement")
    p.add_argument("--out", default="robustness_report.csv", type=str)
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
    raise SystemExit("[ERROR] theta_eval_hbatch_biquat_max.py not found.")

def run_eval(symbols, interval, W, horizon, minP, maxP, nP, sigma, lam, pred_ensemble, max_by, suffix):
    # Run evaluator on CSV paths
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
    # evaluator also writes per-bar files as eval_h_<SYMBOL>_<interval>csv.csv — construct names
    perbar = { s: f"eval_h_{s}_{interval.replace('/','') + 'csv'}.csv" for s in symbols }
    return out_sum, perbar

def read_perbar(path):
    df = pd.read_csv(path)
    # Expected columns from your run: time, entry_idx, compare_idx, last_price, pred_price, future_price, pred_delta, true_delta, ...
    # Be permissive:
    cols = {c.lower(): c for c in df.columns}
    def need(name): 
        for k in cols:
            if k == name: return cols[k]
        raise KeyError(f"Missing column `{name}` in {path}. Have: {list(df.columns)}")
    # Cast
    for k in ["entry_idx","compare_idx"]:
        df[need(k)] = df[need(k)].astype(int)
    for k in ["pred_delta","true_delta"]:
        df[need(k)] = pd.to_numeric(df[need(k)], errors="coerce")
    return df, need

def diagnostics_for_symbol(symbol, perbar_path, horizon, seed):
    np.random.seed(seed)
    df, need = read_perbar(perbar_path)
    # integrity checks
    idx_ok = bool(((df[need("compare_idx")] - df[need("entry_idx")]) == horizon).all())
    time_monotonic = True
    if "time" in [c.lower() for c in df.columns]:
        tcol = [c for c in df.columns if c.lower()=="time"][0]
        try:
            t = pd.to_datetime(df[tcol])
            time_monotonic = bool((t.sort_values().reset_index(drop=True).equals(t.reset_index(drop=True))))
        except Exception:
            time_monotonic = False
    # correlations
    pred = df[need("pred_delta")].values
    true = df[need("true_delta")].values
    mask = np.isfinite(pred) & np.isfinite(true)
    pred = pred[mask]; true = true[mask]
    def safe_corr(a,b):
        if len(a) < 3 or np.std(a)==0 or np.std(b)==0: return 0.0
        return float(np.corrcoef(a,b)[0,1])
    corr_base = safe_corr(pred, true)
    # lag +1
    true_lag = np.roll(true, -1)  # shift future further by +1; drop last
    corr_lag1 = safe_corr(pred[:-1], true_lag[:-1])
    # shuffle true/pred
    true_shuf = true.copy(); np.random.shuffle(true_shuf)
    pred_shuf = pred.copy(); np.random.shuffle(pred_shuf)
    corr_shuffle_true = safe_corr(pred, true_shuf)
    corr_shuffle_pred = safe_corr(pred_shuf, true)
    leak_flag = (abs(corr_lag1) > 0.2) or (abs(corr_shuffle_true) > 0.2) or (abs(corr_shuffle_pred) > 0.2)
    return dict(symbol=symbol, corr_base=corr_base, corr_lag1=corr_lag1,
                corr_shuffle_true=corr_shuffle_true, corr_shuffle_pred=corr_shuffle_pred,
                leak_flag=bool(leak_flag), idx_ok=bool(idx_ok), time_monotonic=bool(time_monotonic),
                count=int(len(df)))

def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    ensure_prices_csv(symbols, args.interval, args.limit)
    # Baseline run
    _, perbar = run_eval(symbols, args.interval, args.W, args.horizon, args.minP, args.maxP, args.nP,
                         args.sigma, args.lam, args.pred_ensemble, args.max_by, suffix="")
    rows = []
    for s in symbols:
        rows.append(diagnostics_for_symbol(s, perbar[s], args.horizon, args.seed))
    df = pd.DataFrame(rows)
    # Optional dense nP rerun
    if args.dense_np:
        _, perbar_dense = run_eval(symbols, args.interval, args.W, args.horizon, args.minP, args.maxP, args.dense_np,
                                   args.sigma, args.lam, args.pred_ensemble, args.max_by, suffix=f"_dense{args.dense_np}")
        rows2 = []
        for s in symbols:
            d = diagnostics_for_symbol(s, perbar_dense[s], args.horizon, args.seed+1)
            d = {"symbol": s, "corr_dense": d["corr_base"]}
            rows2.append(d)
        df2 = pd.DataFrame(rows2)
        df = df.merge(df2, on="symbol", how="left")
        df["delta_corr_dense"] = df["corr_dense"] - df["corr_base"]
    out = Path(args.out)
    df.to_csv(out, index=False)
    print("\n=== Robustness report ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
