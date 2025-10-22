#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_batch.py
Aktuální predikce pro více symbolů najednou, s použitím funkcí z theta_predictor.py.

Použití:
  python theta_batch.py --symbols BTCUSDT,ETHUSDT,BNBUSDT --interval 1h \
    --horizons 1,4,24 --window 256 --out batch.csv
"""
from __future__ import annotations
import argparse, math, sys
from typing import List
import pandas as pd

import theta_predictor as tp


def parse_args():
    p = argparse.ArgumentParser(description="Batch predikce (více symbolů) pro theta-basis")
    p.add_argument("--symbols", required=True, type=str, help="čárkou oddělený seznam symbolů")
    p.add_argument("--interval", default="1h", type=str)
    p.add_argument("--window", "--W", dest="W", default=256, type=int)
    p.add_argument("--horizons", default="1,4,24", type=str)
    p.add_argument("--minP", default=24.0, type=float)
    p.add_argument("--maxP", default=480.0, type=float)
    p.add_argument("--nP", default=8, type=int)
    p.add_argument("--sigma", default=0.8, type=float)
    p.add_argument("--lambda", dest="lam", default=1e-3, type=float)
    p.add_argument("--no-qr", action="store_true")
    p.add_argument("--no-kalman", action="store_true")
    p.add_argument("--fee-bps", default=5.0, type=float)
    p.add_argument("--theta-terms", default=20, type=int)
    p.add_argument("--out", default=None, type=str)
    return p.parse_args()


def main():
    args = parse_args()
    q = math.exp(- (math.pi * args.sigma)**2)
    periods = tp.make_period_grid(args.minP, args.maxP, args.nP)
    horizons = [int(h.strip()) for h in str(args.horizons).split(",") if h.strip()]
    rows = []

    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        try:
            df = tp.fetch_klines(sym, args.interval, limit=max(1000, args.W + 10))
        except Exception as e:
            print(f"[ERROR] fetching {sym}: {e}", file=sys.stderr)
            continue
        y = tp.prepare_series(df, use_log=True)
        for h in horizons:
            pred_log, last_log, dir_score = tp.fit_predict(
                y=y, W=args.W, horizon=h, q=q, periods=periods,
                lam=args.lam, use_qr=not args.no_qr, use_kalman=not args.no_kalman,
                theta_terms=args.theta_terms
            )
            signal = tp.decide_signal(dir_score, fee_bps=args.fee_bps)
            rows.append(dict(
                symbol=sym,
                interval=args.interval,
                window=args.W,
                horizon=h,
                sigma=args.sigma,
                q=q,
                lam=args.lam,
                fee_bps=args.fee_bps,
                last_price=math.exp(last_log),
                pred_price=math.exp(pred_log),
                delta=math.exp(pred_log) - math.exp(last_log),
                dir_score=dir_score,
                signal=signal,
                t_observed=df.index[-1].isoformat()
            ))

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"\nUloženo: {args.out}")


if __name__ == "__main__":
    main()
