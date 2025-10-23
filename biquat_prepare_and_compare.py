#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
biquat_prepare_and_compare.py  (FULL, FIXED)
Porovnání "nový BIQUAT MAX" vs. "původní theta RAW/HOLD" napříč páry.
- Připraví CSV z Binance
- Spustí nový evaluátor na CSV cestách
- Spustí starý batch evaluátor na tickerech
- Sloučí a spočítá delta_corr

Použití:
  python biquat_prepare_and_compare.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT \
    --interval 1h --window 256 --horizon 4 --minP 24 --maxP 480 --nP 8 \
    --sigma 0.8 --lambda 1e-3 --limit 2000 \
    --pred-ensemble avg --max-by transform \
    --out compare_corr.csv
"""
import argparse
import subprocess
import pandas as pd
import json
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(description="BIQUAT vs. THETA comparátor (corr_pred_true)")
    p.add_argument("--symbols", required=True, type=str)
    p.add_argument("--interval", default="1h", type=str)
    p.add_argument("--window", "--W", dest="W", default=256, type=int)
    p.add_argument("--horizon", default=4, type=int)
    p.add_argument("--minP", default=24, type=float)
    p.add_argument("--maxP", default=480, type=float)
    p.add_argument("--nP", default=8, type=int)
    p.add_argument("--sigma", default=0.8, type=float)
    p.add_argument("--lambda", dest="lam", default=1e-3, type=float)
    p.add_argument("--limit", default=2000, type=int)
    p.add_argument("--pred-ensemble", dest="pred_ensemble", choices=["avg","max"], default="avg")
    p.add_argument("--max-by", dest="max_by", choices=["transform","contrib"], default="transform")
    p.add_argument("--out", default="compare_corr.csv", type=str)
    return p.parse_args()

def ensure_prices_csv(symbols, interval, limit):
    # calls make_prices_csv.py in current working directory
    cmd = [sys.executable, "make_prices_csv.py",
           "--symbols", ",".join(symbols),
           "--interval", interval,
           "--limit", str(int(max(1000, limit))),
           "--outdir", "prices"]
    subprocess.run(cmd, check=True)

def find_new_eval():
    candidates = [
        "biquat_max_standalone/theta_eval_hbatch_biquat_max.py",
        "theta_eval_hbatch_biquat_max.py"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise SystemExit("[ERROR] Nenalezen theta_eval_hbatch_biquat_max.py (balíček BIQUAT MAX musí být rozbalen ve stejné složce).")

def norm_sym_from_path(p):
    base = os.path.basename(str(p))
    # e.g. BTCUSDT_1h.csv -> BTCUSDT
    if "_" in base:
        base = base.split("_")[0]
    if "." in base:
        base = base.split(".")[0]
    return base.upper()

def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Force integer strings for new evaluator where it expects ints
    minP_i = str(int(round(args.minP)))
    maxP_i = str(int(round(args.maxP)))
    nP_i   = str(int(round(args.nP)))
    W_i    = str(int(round(args.W)))
    h_i    = str(int(round(args.horizon)))
    limit_i= str(int(round(max(1000, args.limit))))
    sigma_s= str(args.sigma)
    lam_s  = str(args.lam)

    # 1) Prepare CSVs
    ensure_prices_csv(symbols, args.interval, args.limit)

    # 2) Run NEW evaluator (BIQUAT) on CSV paths
    csv_paths = [os.path.abspath(f"prices/{s}_{args.interval}.csv") for s in symbols]
    print("CSV paths:", csv_paths)
    new_eval = find_new_eval()
    out_new = "hbatch_biquat_summary.csv"
    cmd_new = [sys.executable, new_eval,
               "--symbols", ",".join(csv_paths),
               "--interval", args.interval,
               "--window", W_i,
               "--horizon", h_i,
               "--minP", minP_i,
               "--maxP", maxP_i,
               "--nP", nP_i,
               "--sigma", sigma_s,
               "--lambda", lam_s,
               "--pred-ensemble", args.pred_ensemble,
               "--max-by", args.max_by,
               "--out", out_new]
    subprocess.run(cmd_new, check=True)
    df_new = pd.read_csv(out_new)
    # guess columns
    corr_col_new = next((c for c in df_new.columns if c in ["corr","c","corr_pred_true"]), None)
    sym_col_new = next((c for c in df_new.columns if c.lower() in ["symbol","sym","ticker","name"]), None)
    if corr_col_new is None or sym_col_new is None:
        raise SystemExit(f"[ERROR] V {out_new} chybí očekávané sloupce (symbol/corr). Má: {list(df_new.columns)}")
    df_new = df_new.rename(columns={sym_col_new:"symbol", corr_col_new:"corr_new"})
    # Normalize symbol names from CSV paths
    df_new["symbol"] = df_new["symbol"].map(norm_sym_from_path)
    df_new = df_new[["symbol","corr_new"]]

    # 3) Run OLD evaluator on tickers
    out_old = "hbatch_summary_old.csv"
    cmd_old = [sys.executable, "theta_eval_hbatch.py",
               "--symbols", ",".join(symbols),
               "--interval", args.interval,
               "--window", W_i,
               "--horizon", h_i,
               "--minP", minP_i,
               "--maxP", maxP_i,
               "--nP", nP_i,
               "--sigma", sigma_s,
               "--lambda", lam_s,
               "--limit", limit_i,
               "--out", out_old]
    subprocess.run(cmd_old, check=True)
    df_old = pd.read_csv(out_old)
    if "corr_pred_true" not in df_old.columns:
        raise SystemExit(f"[ERROR] V {out_old} chybí corr_pred_true (sloupce: {list(df_old.columns)})")
    df_old = df_old.rename(columns={"corr_pred_true":"corr_old"})
    df_old = df_old[["symbol","corr_old","hit_rate_pred","hit_rate_hold","delta_hit","count"]]

    # 4) Merge & compute delta_corr
    df = df_old.merge(df_new, on="symbol", how="inner")
    df["delta_corr"] = df["corr_new"] - df["corr_old"]
    df.to_csv(args.out, index=False)
    print("\n=== Srovnání corr (nový vs. starý) ===")
    print(df.to_string(index=False))
    print(f"\nUloženo: {args.out}")

if __name__ == "__main__":
    main()
