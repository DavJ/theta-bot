#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_backtest.py
Rolling walk-forward backtest pro theta-basis prediktor.

Použití:
  python theta_backtest.py --symbol BTCUSDT --interval 1h --window 256 --horizon 4 \
    --minP 24 --maxP 480 --nP 8 --sigma 0.8 --lambda 1e-3 --fee-bps 5 --limit 2000 \
    --out equity.csv --summary summary.json

Výstupy:
  - CSV equity křivky a obchodů (sloupce: time, price, signal, entry_price, exit_price, pnl, equity, ...)
  - JSON souhrn metrik (total_return, trades, hit_rate, sharpe, max_drawdown, ...)
"""
from __future__ import annotations
import argparse, math, sys, json
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import pandas as pd

# Importujeme funkce z theta_predictor.py
import theta_predictor as tp


def calc_pnl_series(prices: np.ndarray, entries: np.ndarray, exits: np.ndarray,
                    signals: np.ndarray, fee_bps: float) -> np.ndarray:
    """
    Jednoduché PnL: vstup na close[t], výstup na close[t+h].
    Signál ∈ {LONG, SHORT, FLAT}. Poplatek účtujeme 2× fee (round-trip).
    """
    fee = fee_bps / 10000.0
    pnl = np.zeros_like(signals, dtype=np.float64)
    for i, sig in enumerate(signals):
        if sig == "LONG":
            pnl[i] = (exits[i] - entries[i]) / entries[i] - 2*fee
        elif sig == "SHORT":
            pnl[i] = (entries[i] - exits[i]) / entries[i] - 2*fee
        else:
            pnl[i] = 0.0
    return pnl


def rolling_backtest(df: pd.DataFrame, W: int, horizon: int, q: float, periods: List[float],
                     lam: float, use_qr: bool, use_kalman: bool, theta_terms: int, fee_bps: float,
                     exec_mode: str = 'close') -> pd.DataFrame:
    y = tp.prepare_series(df, use_log=True)
    prices_close = df["close"].values.astype(float)
    prices_open = df["open"].values.astype(float)
    T = len(y)
    rows = []
    # predikujeme pro každý t, kde máme okno W a budoucnost t+h exists
    for t in range(W, T - horizon):
        yw = y[:t]
        # fit + pred (window clip)
        w_eff = min(W, t-2)
        pred_log, last_log, dir_score = tp.fit_predict(
            y=yw, W=w_eff, horizon=horizon, q=q, periods=periods,
            lam=lam, use_qr=use_qr, use_kalman=use_kalman, theta_terms=theta_terms
        )
        signal = tp.decide_signal(dir_score, fee_bps=fee_bps)
        if exec_mode == 'close':
            entry_price = prices_close[t-1]
            exit_price  = prices_close[t-1 + horizon]
        else:
            entry_price = prices_open[t]
            exit_price  = prices_open[t + horizon]
        rows.append(dict(
            time=df.index[t-1].isoformat(),
            entry_idx=t-1,
            exit_idx=t-1+horizon,
            last_price=math.exp(last_log),
            pred_price=math.exp(pred_log),
            signal=signal,
            entry_price=entry_price,
            exit_price=exit_price,
            dir_score=dir_score
        ))
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt
    bt["pnl"] = calc_pnl_series(prices=None,
                                entries=bt["entry_price"].values,
                                exits=bt["exit_price"].values,
                                signals=bt["signal"].values,
                                fee_bps=fee_bps)
    bt["equity"] = (1.0 + bt["pnl"]).cumprod()
    return bt


def metrics(bt: pd.DataFrame) -> Dict[str, Any]:
    if bt.empty:
        return {"trades": 0}
    trades = (bt["signal"].values != "FLAT").sum()
    wins = ((bt["pnl"] > 0) & (bt["signal"] != "FLAT")).sum()
    hit_rate = float(wins) / trades if trades else 0.0
    ret = bt["equity"].iloc[-1] - 1.0
    # Sharpe (per-trade): mean/std * sqrt(N)
    pnl_nonzero = bt.loc[bt["signal"] != "FLAT", "pnl"].values
    if len(pnl_nonzero) > 1 and np.std(pnl_nonzero) > 1e-12:
        sharpe = float(np.mean(pnl_nonzero) / np.std(pnl_nonzero) * np.sqrt(len(pnl_nonzero)))
    else:
        sharpe = 0.0
    # Max drawdown
    rollmax = bt["equity"].cummax()
    dd = (bt["equity"] / rollmax - 1.0).min()
    return {
        "trades": int(trades),
        "hit_rate": hit_rate,
        "total_return": ret,
        "sharpe_trades": sharpe,
        "max_drawdown": float(dd),
        "start": bt["time"].iloc[0],
        "end": bt["time"].iloc[-1],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Rolling backtest pro theta-basis prediktor")
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
    p.add_argument("--fee-bps", default=5.0, type=float)
    p.add_argument("--exec", dest="exec_mode", choices=["close","open"], default="close", help="exekuce na close(t-1)->close(t-1+h) nebo open(t)->open(t+h)")
    p.add_argument("--limit", default=3000, type=int, help="kolik svíček stáhnout z Binance (max 1000 per call, zde jednoduché omezení)")
    p.add_argument("--out", default=None, type=str, help="CSV equity/trades")
    p.add_argument("--summary", default=None, type=str, help="JSON souhrn metrik")
    return p.parse_args()


def main():
    args = parse_args()

    q = math.exp(- (math.pi * args.sigma)**2)
    periods = tp.make_period_grid(args.minP, args.maxP, args.nP)

    # Vytáhneme data (jedno volání; pro víc než 1000 svíček by bylo potřeba stránkovat)
    limit = max(1000, args.limit)
    try:
        df = tp.fetch_klines(args.symbol, args.interval, limit=limit)
    except Exception as e:
        print(f"[ERROR] fetching klines: {e}", file=sys.stderr)
        sys.exit(2)

    bt = rolling_backtest(df, W=args.W, horizon=args.horizon, q=q, periods=periods,
                          lam=args.lam, use_qr=not args.no_qr, use_kalman=not args.no_kalman,
                          theta_terms=args.theta_terms, fee_bps=args.fee_bps,
                          exec_mode=args.exec_mode)

    if bt.empty:
        print("Backtest nemá dost dat.")
        sys.exit(0)

    m = metrics(bt)

    print(bt.head().to_string(index=False))
    print("\n--- Summary ---")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        bt.to_csv(args.out, index=False)
        print(f"\nUloženo CSV: {args.out}")
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        print(f"Uloženo summary: {args.summary}")


if __name__ == "__main__":
    main()
