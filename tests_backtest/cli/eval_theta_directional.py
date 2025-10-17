#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directional evaluator for theta forecasts.
- Loads forecast CSV from run_theta_forecast
- For each horizon h and variant v:
    * true return r_t = close(t+h) - close(t)
    * pred return  p_t = pred_v_h(t) - close(t)
    * Metrics: sign acc, balanced acc, precision/recall (positive=up), MAE/MSE on returns
    * Simple P&L: long if p_t>thr, else flat (or short if --allow-short), include fee bps
Saves a summary CSV and prints tables.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def sign_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    # y_true, y_pred: real-valued returns
    s_true = np.sign(y_true)
    s_pred = np.sign(y_pred)
    # map zeros to 0 (ignored in some metrics); treat as no position
    mask = (s_true != 0)
    if mask.sum() == 0:
        return dict(n=0, acc=np.nan, bacc=np.nan, prec=np.nan, rec=np.nan)
    s_true = s_true[mask]
    s_pred = s_pred[mask]
    tp = int(((s_true>0)&(s_pred>0)).sum())
    tn = int(((s_true<0)&(s_pred<0)).sum())
    fp = int(((s_true<0)&(s_pred>0)).sum())
    fn = int(((s_true>0)&(s_pred<0)).sum())
    n = int(len(s_true))
    acc = (tp+tn)/n if n else np.nan
    # balanced acc
    pos = (s_true>0).sum()
    neg = (s_true<0).sum()
    tpr = tp/pos if pos else np.nan
    tnr = tn/neg if neg else np.nan
    bacc = np.nanmean([tpr, tnr])
    prec = tp/(tp+fp) if (tp+fp)>0 else np.nan
    rec  = tpr
    return dict(n=n, acc=acc, bacc=bacc, prec=prec, rec=rec, tp=tp, tn=tn, fp=fp, fn=fn)

def pnl_strategy(y_true: np.ndarray, y_pred: np.ndarray, fee_bps: float=1.0, allow_short: bool=False, thr: float=0.0):
    # Simple strategy: if predicted return > thr -> long for h bars; else (short if allowed) flat
    # For evaluation alignment, we use the return exactly y_true as realized P&L (per unit notional)
    # Fees: charged when position opens (and closes); assume symmetric -> 2 * fee_bps
    fee = 2.0 * (fee_bps * 1e-4)
    if allow_short:
        pos = np.where(y_pred>thr, 1.0, -1.0)
    else:
        pos = np.where(y_pred>thr, 1.0, 0.0)
    gross = pos * y_true
    trades = (pos!=0).astype(float)
    net = gross - fee * trades
    cum = np.cumsum(net)
    return dict(trades=int(trades.sum()), pnl_total=float(net.sum()), pnl_avg=float(net.mean()), pnl_cum_last=float(cum[-1] if len(cum)>0 else 0.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="forecast CSV from run_theta_forecast")
    ap.add_argument("--outdir", default="reports_forecast")
    ap.add_argument("--fee-bps", type=float, default=1.0)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--thr", type=float, default=0.0, help="threshold on predicted return")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Identify horizons and variants
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        raise RuntimeError("No pred_* columns found.")
    # parse e.g. pred_thetaOrtho_h96
    items = []
    for c in pred_cols:
        try:
            v, h = c.split("_")[1], int(c.split("_h")[1])
            items.append((v, h, c))
        except Exception:
            continue
    # compute returns for each horizon
    closes = df["close"].to_numpy(dtype=float)
    res_rows = []
    for v, h, c in sorted(items, key=lambda x:(x[1], x[0])):
        # align: for t where t+h exists
        y_true = closes[h:] - closes[:-h]
        y_pred = df[c].to_numpy(dtype=float)[:-h] - closes[:-h]
        # Core metrics
        s = sign_metrics(y_true, y_pred)
        mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true)>0 else np.nan
        mse = float(np.mean((y_true - y_pred)**2)) if len(y_true)>0 else np.nan
        pnl = pnl_strategy(y_true, y_pred, fee_bps=args.fee_bps, allow_short=args.allow_short, thr=args.thr)
        res_rows.append({
            "variant": v, "horizon_bars": h, "N": s.get("n", len(y_true)),
            "sign_acc": s.get("acc", np.nan),
            "sign_bacc": s.get("bacc", np.nan),
            "precision_up": s.get("prec", np.nan),
            "recall_up": s.get("rec", np.nan),
            "MAE_ret": mae, "MSE_ret": mse,
            "trades": pnl["trades"], "pnl_total": pnl["pnl_total"], "pnl_avg": pnl["pnl_avg"]
        })
    res = pd.DataFrame(res_rows)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / (Path(args.csv).stem + "_metrics_directional.csv")
    res.to_csv(out_csv, index=False)

    # Pretty print by horizon
    for h in sorted(res["horizon_bars"].unique()):
        sub = res[res["horizon_bars"]==h].sort_values("sign_acc", ascending=False)
        print(f"\n=== Directional leaderboard (h={h}) — sign_acc (vyšší lepší) ===")
        print(sub[["variant","N","sign_acc","sign_bacc","precision_up","recall_up","MAE_ret","MSE_ret","trades","pnl_total","pnl_avg"]].to_string(index=False))
    print(f"\nUloženo: {out_csv}")

if __name__ == "__main__":
    main()
