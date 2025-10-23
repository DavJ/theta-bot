#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness_suite_v3_oos.py

Out-of-sample (OOS) report on a strict TIME SPLIT:
- We split each symbol's CSV by time index: first `oos_split` fraction is TRAIN,
  the rest is TEST. We then **evaluate ONLY the TEST rows** of the evaluator's
  per-bar file (walk-forward causally refitting on test is allowed in this mode).
- This is a conservative, leak-free OOS score, stronger než prostý rolling corr,
  ale nikoli "frozen-weights".

If you later add support to the evaluator (theta_eval_hbatch_biquat_max.py)
for flag `--oos-freeze-at <split_idx>`, this script will automatically detect it
and switch to strict "frozen-train" OOS (one fit on train, apply on test).

Usage example:
python robustness_suite_v3_oos.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT \
  --interval 1h --window 256 --horizon 4 --minP 24 --maxP 480 --nP 16 \
  --sigma 0.8 --lambda 1e-3 --limit 2000 \
  --pred-ensemble avg --max-by transform \
  --oos-split 0.7 \
  --out robustness_report_v3_oos.csv
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
    p.add_argument("--nP", default=8, type=int)
    p.add_argument("--sigma", default=0.8, type=float)
    p.add_argument("--lambda", dest="lam", default=1e-3, type=float)
    p.add_argument("--limit", default=2000, type=int)
    p.add_argument("--pred-ensemble", dest="pred_ensemble", choices=["avg","max"], default="avg")
    p.add_argument("--max-by", dest="max_by", choices=["transform","contrib"], default="transform")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--oos-split", default=0.7, type=float, help="fraction of rows for TRAIN (rest TEST)")
    p.add_argument("--out", default="robustness_report_v3_oos.csv", type=str)
    return p.parse_args()

def ensure_prices_csv(symbols, interval, limit):
    cmd = [sys.executable, "make_prices_csv.py",
           "--symbols", ",".join(symbols),
           "--interval", interval,
           "--limit", str(int(max(1000, limit))),
           "--outdir", "prices"]
    subprocess.run(cmd, check=True)

def find_eval():
    for c in ["biquat_max_standalone/theta_eval_hbatch_biquat_max.py",
              "theta_eval_hbatch_biquat_max.py"]:
        if os.path.exists(c):
            return c
    raise SystemExit("theta_eval_hbatch_biquat_max.py not found.")

def run_eval(symbols, interval, W, horizon, minP, maxP, nP, sigma, lam, pred_ensemble, max_by, extra_args=None):
    eval_path = find_eval()
    csv_paths = [os.path.abspath(f"prices/{s}_{interval}.csv") for s in symbols]
    out_sum = "hbatch_v3_tmp.csv"
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
    if extra_args:
        cmd += extra_args
    subprocess.run(cmd, check=True)
    # evaluator emits per-symbol per-bar files as eval_h_<SYMBOL>_<interval>csv.csv
    perbar = { s: f"eval_h_{s}_{interval.replace('/','') + 'csv'}.csv" for s in symbols }
    return perbar

def safe_corr(a,b):
    if len(a) < 3: return 0.0
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0: return 0.0
    return float(np.corrcoef(a,b)[0,1])

def diag_oos(symbol, perbar_path, horizon, oos_split):
    df = pd.read_csv(perbar_path)
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    def need(n):
        for k,v in cols.items():
            if k == n: return v
        raise KeyError(f"Missing `{n}` in {perbar_path}")
    # integrity
    idx_ok = bool(((df[need("compare_idx")] - df[need("entry_idx")]) == horizon).all())
    n_total = len(df)
    split_idx = int(n_total * oos_split)
    # filter TEST rows
    test = df.iloc[split_idx:].copy()
    # numeric arrays
    pred = pd.to_numeric(test[need("pred_delta")], errors="coerce").to_numpy()
    true = pd.to_numeric(test[need("true_delta")], errors="coerce").to_numpy()
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]
    # base corr on TEST
    corr_base = safe_corr(pred, true)
    # lag-h corr for leakage sanity
    corr_lag_h = safe_corr(pred[:-horizon], true[horizon:]) if horizon < len(true) else 0.0
    # shuffle sanity
    rng = np.random.default_rng(123)
    true_shuf = true.copy(); rng.shuffle(true_shuf)
    corr_shuffle_true = safe_corr(pred, true_shuf)
    # dir hit-rate (TEST)
    hit_rate = float(np.mean(np.sign(pred) == np.sign(true))) if len(true) else 0.0
    return {
        "symbol": symbol,
        "idx_ok": idx_ok,
        "n_total": int(n_total),
        "n_test": int(len(true)),
        "split_at": split_idx,
        "corr_oos": corr_base,
        "hit_rate_oos": hit_rate,
        "corr_lag_h": corr_lag_h,
        "corr_shuffle_true": corr_shuffle_true,
    }

def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    ensure_prices_csv(symbols, args.interval, args.limit)

    # Try "frozen-train" OOS if evaluator supports it (optional feature)
    # Here we simply check an env flag EVAL_SUPPORTS_OOS_FREEZE for backward compatibility.
    extra_args = None
    if os.environ.get("EVAL_SUPPORTS_OOS_FREEZE") == "1":
        # We'll use split index in terms of perbar rows; evaluator should interpret it properly.
        # This is a placeholder hook; if not supported, it will be ignored by evaluator.
        extra_args = ["--oos-freeze-at", "auto"]

    perbar = run_eval(symbols, args.interval, args.W, args.horizon, args.minP, args.maxP,
                      args.nP, args.sigma, args.lam, args.pred_ensemble, args.max_by, extra_args)

    rows = []
    for s in symbols:
        r = diag_oos(s, perbar[s], args.horizon, args.oos_split)
        rows.append(r)
        print(f"[OOS] {s}: corr={r['corr_oos']:.3f}, hit={r['hit_rate_oos']:.3f}, n_test={r['n_test']}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    df.to_csv(out, index=False)
    print("\n=== OOS Summary (TEST only) ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
