#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_theta_directional_plus.py
==============================
Rozšířená evaluace směrové přesnosti a PnL s podporou:
- --confirm-k N       : vyžaduje shodu posledních N signálů (hysterese).
- --deadband-sigma S  : neobchodovat, pokud |pred_delta| < S * sigma(pred_delta).
- --thr T             : absolutní práh na |pred_delta| (stejně jako stávající skript).
- --fee-bps F         : poplatek v bodech (0.5 = 0.5 bps).

Poznámky k metrikám:
- sign_acc / sign_bacc se počítají přes VŠECHNY řádky (kde existuje t+H),
  tj. nezávisle na tom, jestli by skript obchod vzal.
- PnL a trades počítáme jen pro řádky, které prošly filtry (thr, deadband, confirm-k).
- PnL model: otevřu v t, zavřu v t+H, velikost 1 jednotka (např. 1 BTC).
  Fee = 2 * (fee_bps/1e4) * close[t].
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Directional eval PLUS (confirm-k / deadband)")
    p.add_argument("--csv", required=True, type=str)
    p.add_argument("--outdir", default="reports_forecast", type=str)
    p.add_argument("--fee-bps", default=1.0, type=float, dest="fee_bps")
    p.add_argument("--thr", default=0.0, type=float, help="Absolutní práh na |pred - close|.")
    p.add_argument("--confirm-k", default=0, type=int, help="Počet posledních shodných signálů vyžadovaných pro trade.")
    p.add_argument("--deadband-sigma", default=0.0, type=float, help="Mrtvé pásmo: |pred_delta| < S * sigma(pred_delta) => no trade.")
    return p.parse_args()

def extract_variants_and_horizons(df: pd.DataFrame):
    import re
    variants = {}
    for c in df.columns:
        m = re.match(r"^pred_(?P<var>[A-Za-z0-9_]+)_h(?P<h>\d+)$", c)
        if m:
            v = m.group("var")
            h = int(m.group("h"))
            variants.setdefault(v, []).append(h)
    for v in variants:
        variants[v] = sorted(set(variants[v]))
    return variants

def directional_metrics(df: pd.DataFrame, variant: str, H: int, args):
    df = df.copy()
    if "close" not in df.columns:
        raise RuntimeError("CSV neobsahuje sloupec 'close'.")
    df["close_fwd"] = df["close"].shift(-H)
    df = df.dropna(subset=["close_fwd"]).reset_index(drop=True)

    pred_col = f"pred_{variant}_h{H}"
    if pred_col not in df.columns:
        return None

    df["pred_delta"] = df[pred_col] - df["close"]
    df["true_delta"] = df["close_fwd"] - df["close"]

    s_pred = np.sign(df["pred_delta"].values)
    s_true = np.sign(df["true_delta"].values)
    mask_nonzero = (s_true != 0)
    N = int(mask_nonzero.sum()) if mask_nonzero.any() else len(df)

    def safe_mean(x):
        return float(np.mean(x)) if len(x)>0 else float('nan')

    sign_acc = safe_mean((s_pred[mask_nonzero] == s_true[mask_nonzero]).astype(float))

    up_mask   = (s_true > 0)
    down_mask = (s_true < 0)
    recall_up = safe_mean((s_pred[up_mask] == 1).astype(float)) if up_mask.any() else float('nan')
    recall_dn = safe_mean((s_pred[down_mask] == -1).astype(float)) if down_mask.any() else float('nan')
    sign_bacc = np.nanmean([recall_up, recall_dn])
    pred_up_mask = (s_pred == 1)
    precision_up = safe_mean((s_true[pred_up_mask] == 1).astype(float)) if pred_up_mask.any() else float('nan')

    MAE_ret = float(np.mean(np.abs(df["true_delta"].values)))
    MSE_ret = float(np.mean((df["true_delta"].values)**2))

    pred_abs = np.abs(df["pred_delta"].values)
    if args.deadband_sigma > 0:
        sigma = float(np.std(df["pred_delta"].values, ddof=1))
        deadband = args.deadband_sigma * sigma
    else:
        deadband = 0.0

    trade_mask = (pred_abs >= max(args.thr, deadband))

    if args.confirm_k and args.confirm_k > 0:
        k = int(args.confirm_k)
        s = s_pred.copy()
        ok = np.zeros_like(trade_mask, dtype=bool)
        run = 0
        prev = 0
        for i in range(len(s)):
            if s[i] == 0:
                run = 0
                prev = 0
                continue
            if s[i] == prev:
                run += 1
            else:
                run = 1
                prev = s[i]
            if run >= k:
                ok[i] = True
        trade_mask = trade_mask & ok

    fee = (args.fee_bps / 10000.0) * 2.0
    signed_ret = np.sign(df["pred_delta"].values) * df["true_delta"].values
    fee_cost = fee * df["close"].values
    pnl = signed_ret - fee_cost
    pnl = pnl[trade_mask]
    trades = int(trade_mask.sum())
    pnl_total = float(np.sum(pnl)) if trades>0 else 0.0
    pnl_avg = float(np.mean(pnl)) if trades>0 else 0.0

    row = {
        "variant": variant,
        "horizon_bars": H,
        "N": int(len(df)),
        "sign_acc": float(sign_acc),
        "sign_bacc": float(sign_bacc),
        "precision_up": float(precision_up),
        "recall_up": float(recall_up),
        "MAE_ret": float(MAE_ret),
        "MSE_ret": float(MSE_ret),
        "trades": trades,
        "pnl_total": pnl_total,
        "pnl_avg": pnl_avg,
    }
    return row

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    variants = extract_variants_and_horizons(df)

    all_rows = []
    for var, horizons in variants.items():
        for H in sorted(horizons):
            r = directional_metrics(df, var, H, args)
            if r is not None:
                all_rows.append(r)
    if not all_rows:
        print("Nenalezeny žádné predikce v CSV.")
        return

    out_df = pd.DataFrame(all_rows)
    out_df = out_df.sort_values(["horizon_bars", "variant"]).reset_index(drop=True)

    print("\n=== Directional leaderboard (rozšířený) — sign_acc (vyšší lepší) ===")
    print(out_df.to_string(index=False,
        columns=["variant","horizon_bars","N","sign_acc","sign_bacc","precision_up","recall_up","MAE_ret","MSE_ret","trades","pnl_total","pnl_avg"],
        justify="right",
        formatters={
            "sign_acc": "{:.6f}".format,
            "sign_bacc": "{:.6f}".format,
            "precision_up": "{:.6f}".format,
            "recall_up": "{:.6f}".format,
            "MAE_ret": "{:.6f}".format,
            "MSE_ret": "{:.6f}".format,
            "pnl_total": "{:.6f}".format,
            "pnl_avg": "{:.6f}".format,
        }))

    base = Path(args.csv).stem.replace("forecast_", "")
    out_csv = outdir / f"{base}_metrics_directional_plus.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nUloženo: {out_csv}")

if __name__ == "__main__":
    main()
