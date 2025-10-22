#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_eval_hstrategy.py
RAW prediktor vs. HSTRATEGY baseline "HOLD" (buy & hold přes horizon h).

Pro každý krok:
- spočítáme predikci pred_price z tehdejšího stavu,
- budoucí cenu po h barech future_price,
- true delta = future_price - last_price,
- pred delta = pred_price  - last_price,
- hold return = (future_price - last_price) / last_price,
- vyhodnotíme směrovou přesnost prediktoru (sign match) a baseline HOLD (sign(true_delta)>0).

Uloží CSV a JSON shrnutí:
CSV sloupce:
 time, entry_idx, compare_idx, last_price, pred_price, future_price,
 pred_delta, true_delta, pred_dir, true_dir, hold_ret, correct_pred, hold_up

JSON shrnutí:
 hit_rate_pred, hit_rate_hold, corr_pred_vs_true, mae_price, mae_return, count
"""
from __future__ import annotations
import argparse, math, sys, json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path

import theta_predictor as tp


def parse_args():
    p = argparse.ArgumentParser(description="RAW evaluace vs. hstrategy HOLD baseline")
    p.add_argument("--symbol", required=True, type=str)
    p.add_argument("--interval", default="1h", type=str)
    p.add_argument("--window", "--W", dest="W", default=256, type=int)
    p.add_argument("--horizon", default=4, type=int)
    p.add_argument("--minP", default=24.0, type=float)
    p.add_argument("--maxP", default=480.0, type=float)
    p.add_argument("--nP", default=8, type=int)
    p.add_argument("--sigma", default=0.8, type=float)
    p.add_argument("--lambda", dest="lam", default=1e-3, type=float)
    p.add_argument("--no-qr", action="store_true")
    p.add_argument("--no-kalman", action="store_true")
    p.add_argument("--theta-terms", default=20, type=int)
    p.add_argument("--limit", default=3000, type=int)
    p.add_argument("--out", default=None, type=str, help="CSV per-bar výsledků")
    p.add_argument("--summary", default=None, type=str, help="JSON shrnutí metrik")
    return p.parse_args()


def sgn(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def main():
    args = parse_args()

    q = math.exp(- (math.pi * args.sigma)**2)
    periods = tp.make_period_grid(args.minP, args.maxP, args.nP)

    # Data
    limit = max(1000, args.limit)
    try:
        df = tp.fetch_klines(args.symbol, args.interval, limit=limit)
    except Exception as e:
        print(f"[ERROR] fetching klines: {e}", file=sys.stderr)
        sys.exit(2)

    y = tp.prepare_series(df, use_log=True)
    prices = df["close"].values.astype(float)
    T = len(y)
    h = args.horizon

    rows = []
    for t in range(args.W, T - h):
        yw = y[:t]
        w_eff = min(args.W, t-2)
        try:
            pred_log, last_log, dir_score = tp.fit_predict(
                y=yw, W=w_eff, horizon=h, q=q, periods=periods,
                lam=args.lam, use_qr=not args.no_qr, use_kalman=not args.no_kalman,
                theta_terms=args.theta_terms
            )
        except Exception:
            continue

        last_price   = math.exp(last_log)
        pred_price   = math.exp(pred_log)
        future_price = prices[t-1 + h]

        pred_delta = pred_price - last_price
        true_delta = future_price - last_price

        pred_dir = sgn(pred_delta)
        true_dir = sgn(true_delta)

        # HSTRATEGY HOLD baseline: vždy LONG -> "správně", když true_delta > 0
        hold_up = int(true_delta > 0)
        hold_ret = (future_price - last_price) / last_price

        rows.append(dict(
            time=df.index[t-1].isoformat(),
            entry_idx=t-1,
            compare_idx=t-1+h,
            last_price=last_price,
            pred_price=pred_price,
            future_price=future_price,
            pred_delta=pred_delta,
            true_delta=true_delta,
            pred_dir=pred_dir,
            true_dir=true_dir,
            hold_ret=hold_ret,
            correct_pred=int(pred_dir == true_dir),
            hold_up=hold_up
        ))

    eva = pd.DataFrame(rows)
    if eva.empty:
        print("HSTRATEGY eval nemá dost dat.")
        sys.exit(0)

    # Shrnutí metrik
    hit_pred = float(eva["correct_pred"].mean())
    hit_hold = float(eva["hold_up"].mean())  # baseline: jak často je true_delta>0
    if eva["pred_delta"].std() > 1e-12 and eva["true_delta"].std() > 1e-12:
        corr = float(np.corrcoef(eva["pred_delta"].values, eva["true_delta"].values)[0,1])
    else:
        corr = 0.0
    mae_price  = float(np.mean(np.abs(eva["pred_price"].values - eva["future_price"].values)))
    mae_return = float(np.mean(np.abs(np.log(eva["pred_price"].values) - np.log(eva["future_price"].values))))

    print(eva.head().to_string(index=False))
    print("\n--- HSTRATEGY vs HOLD ---")
    print(f"hit_rate_pred:  {hit_pred:.6f}")
    print(f"hit_rate_hold:  {hit_hold:.6f}")
    print(f"corr_pred_true: {corr:.6f}")
    print(f"mae_price:      {mae_price:.6f}")
    print(f"mae_return:     {mae_return:.6f}")
    print(f"count:          {len(eva)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        eva.to_csv(args.out, index=False)
        print(f"\nUloženo CSV: {args.out}")
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(dict(
                hit_rate_pred=hit_pred,
                hit_rate_hold=hit_hold,
                corr_pred_true=corr,
                mae_price=mae_price,
                mae_return=mae_return,
                count=int(len(eva))
            ), f, ensure_ascii=False, indent=2)
        print(f"Uloženo summary: {args.summary}")


if __name__ == "__main__":
    main()
