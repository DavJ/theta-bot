#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_eval_raw.py
"RAW" vyhodnocení prediktoru: porovnání predikce (bez exekuce, bez poplatků)
s reálnou budoucí cenou po zadaném horizontu.

Výstup CSV (per-bar):
 time, entry_idx, compare_idx, last_price, pred_price, future_price,
 delta, future_delta, dir_score, signal, correct

Shrnutí JSON:
 hit_rate_raw, corr_pred_vs_real, mae_price, mae_return, count

Pozn.: časová konvence je stejná jako v backtestu:
- v čase t-1 (poslední dostupná svíčka) uděláme fit a predikci pro horizon h,
- porovnáváme s cenou v čase (t-1+h).
"""
from __future__ import annotations
import argparse, math, sys, json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path

import theta_predictor as tp


def parse_args():
    p = argparse.ArgumentParser(description="RAW eval prediktoru (bez exekuce/poplatků)")
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


def sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def main():
    args = parse_args()

    q = math.exp(- (math.pi * args.sigma)**2)
    periods = tp.make_period_grid(args.minP, args.maxP, args.nP)

    limit = max(1000, args.limit)
    try:
        df = tp.fetch_klines(args.symbol, args.interval, limit=limit)
    except Exception as e:
        print(f"[ERROR] fetching klines: {e}", file=sys.stderr)
        sys.exit(2)

    y = tp.prepare_series(df, use_log=True)
    prices = df["close"].values.astype(float)
    T = len(y)

    rows = []
    h = args.horizon
    for t in range(args.W, T - h):
        # rozhodovací čas = t-1
        yw = y[:t]
        w_eff = min(args.W, t-2)
        try:
            pred_log, last_log, dir_score = tp.fit_predict(
                y=yw, W=w_eff, horizon=h, q=q, periods=periods,
                lam=args.lam, use_qr=not args.no_qr, use_kalman=not args.no_kalman,
                theta_terms=args.theta_terms
            )
        except Exception as e:
            # přeskočit problematické body (vzácné)
            continue
        last_price   = math.exp(last_log)
        pred_price   = math.exp(pred_log)
        future_price = prices[t-1 + h]

        delta_pred   = pred_price - last_price
        delta_real   = future_price - last_price

        correct = int(sign(delta_pred) == sign(delta_real))
        rows.append(dict(
            time=df.index[t-1].isoformat(),
            entry_idx=t-1,
            compare_idx=t-1+h,
            last_price=last_price,
            pred_price=pred_price,
            future_price=future_price,
            delta=delta_pred,
            future_delta=delta_real,
            dir_score=dir_score,
            signal=tp.decide_signal(dir_score, fee_bps=0.0),  # čistý směr bez poplatků
            correct=correct
        ))

    eva = pd.DataFrame(rows)
    if eva.empty:
        print("RAW eval nemá dost dat.")
        sys.exit(0)

    # Metriky
    correct_mask = eva["correct"].values
    hit_rate = float(correct_mask.mean())
    # korelace mezi predikovaným a skutečným delta (přepočet na log-returny je podobný)
    if len(eva) > 2 and eva["delta"].std() > 1e-12 and eva["future_delta"].std() > 1e-12:
        corr = float(np.corrcoef(eva["delta"].values, eva["future_delta"].values)[0,1])
    else:
        corr = 0.0
    mae_price = float(np.mean(np.abs(eva["pred_price"].values - eva["future_price"].values)))
    # MAE v relativním smyslu (na returnech aprox pomocí logů)
    mae_return = float(np.mean(np.abs(np.log(eva["pred_price"].values) - np.log(eva["future_price"].values))))

    print(eva.head().to_string(index=False))
    print("\n--- RAW Summary ---")
    print(f"hit_rate_raw: {hit_rate:.6f}")
    print(f"corr_pred_vs_real: {corr:.6f}")
    print(f"mae_price: {mae_price:.6f}")
    print(f"mae_return: {mae_return:.6f}")
    print(f"count: {len(eva)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        eva.to_csv(args.out, index=False)
        print(f"\nUloženo CSV: {args.out}")
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(dict(
                hit_rate_raw=hit_rate,
                corr_pred_vs_real=corr,
                mae_price=mae_price,
                mae_return=mae_return,
                count=int(len(eva))
            ), f, ensure_ascii=False, indent=2)
        print(f"Uloženo summary: {args.summary}")


if __name__ == "__main__":
    main()
