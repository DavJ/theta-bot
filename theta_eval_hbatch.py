#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_eval_hbatch.py
Batch evaluace RAW prediktor vs. HOLD baseline přes více symbolů.

Použití (příklad):
  python theta_eval_hbatch.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT \
    --interval 1h --window 256 --horizon 4 --minP 24 --maxP 480 --nP 8 \
    --sigma 0.8 --lambda 1e-3 --limit 2000 --out hbatch_summary.csv

Vytvoří:
  - per-symbol JSON summary (sum_h_<symbol>.json)
  - souhrnný CSV s metrikami (hit_rate_pred, hit_rate_hold, delta_hit, corr_pred_true, count)
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Batch RAW vs HOLD evaluace přes více symbolů")
    p.add_argument("--symbols", required=True, type=str, help="seznam symbolů oddělený čárkou, např. BTCUSDT,ETHUSDT,...")
    p.add_argument("--interval", default="1h", type=str)
    p.add_argument("--window", "--W", dest="W", default=256, type=int)
    p.add_argument("--horizon", default=4, type=int)
    p.add_argument("--minP", default=24.0, type=float)
    p.add_argument("--maxP", default=480.0, type=float)
    p.add_argument("--nP", default=8, type=int)
    p.add_argument("--sigma", default=0.8, type=float)
    p.add_argument("--lambda", dest="lam", default=1e-3, type=float)
    p.add_argument("--limit", default=2000, type=int)
    p.add_argument("--out", default="hbatch_summary.csv", type=str, help="souhrnný CSV výstup")
    return p.parse_args()

def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    records = []
    for sym in symbols:
        csv = f"eval_h_{sym}.csv"
        summary = f"sum_h_{sym}.json"
        cmd = [
            sys.executable, "theta_eval_hstrategy.py",
            "--symbol", sym,
            "--interval", args.interval,
            "--window", str(args.W),
            "--horizon", str(args.horizon),
            "--minP", str(args.minP),
            "--maxP", str(args.maxP),
            "--nP", str(args.nP),
            "--sigma", str(args.sigma),
            "--lambda", str(args.lam),
            "--limit", str(args.limit),
            "--out", csv,
            "--summary", summary
        ]
        print(f"\n=== Running {sym} ===")
        subprocess.run(cmd, check=True)
        with open(summary, "r", encoding="utf-8") as f:
            s = json.load(f)
        s["symbol"] = sym
        s["delta_hit"] = s.get("hit_rate_pred", 0) - s.get("hit_rate_hold", 0)
        records.append(s)

    df = pd.DataFrame(records)
    # přeskládat sloupce, pokud existují
    cols = [c for c in ["symbol","hit_rate_pred","hit_rate_hold","delta_hit","corr_pred_true","count"] if c in df.columns]
    df = df[cols]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nUloženo: {out_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
