#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_theta_fractal_plus.py
==========================
Rozšířená evaluace směrové prediktivity a PnL pro fraktální/skáčící průběhy.

Novinky oproti eval_theta_directional_plus:
- --weighted-sign-acc : vážená směrová přesnost (váhy |true_delta|^p)
- --wexp p            : exponent p pro váhy (default 1.0)
- --multi-horizons    : vyhodnotí jen vybrané horizonty (v barech nebo časových značkách: 4,8,12 nebo 15m,30m,1h,4h)
- --buckets N         : report přes kvantily absolutní true_delta (kvalita na velkých pohybech)
- zachovány filtry: --thr, --confirm-k, --deadband-sigma, --fee-bps

Pozn.: Base interval (pro převod 15m,1h→bars) se pokusíme detekovat z názvu CSV (např. "..._15m_...").
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd


# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Fractal directional eval (weighted, multi-horizons, buckets)")
    p.add_argument("--csv", required=True, type=str)
    p.add_argument("--outdir", default="reports_forecast", type=str)
    p.add_argument("--fee-bps", default=1.0, type=float, dest="fee_bps")
    p.add_argument("--thr", default=0.0, type=float, help="Absolutní práh na |pred - close|.")
    p.add_argument("--confirm-k", default=0, type=int, help="Počet posledních shodných signálů vyžadovaných pro trade.")
    p.add_argument("--deadband-sigma", default=0.0, type=float, help="Mrtvé pásmo: |pred_delta| < S * sigma(pred_delta) => no trade.")
    # nové
    p.add_argument("--weighted-sign-acc", action="store_true", help="Zapni váženou směrovou přesnost (váhy |true_delta|^p).")
    p.add_argument("--wexp", default=1.0, type=float, help="Exponent p pro váhy (|true_delta|^p).")
    p.add_argument("--multi-horizons", default=None, type=str, help="Seznam horizontů (např. '4,8,12' nebo '15m,30m,1h,4h').")
    p.add_argument("--buckets", default=0, type=int, help="Počet kvantilových bucketů přes |true_delta| (např. 10).")
    return p.parse_args()


# ----------------- helpers -----------------
def infer_base_minutes_from_name(path: str) -> int | None:
    """Najde v názvu CSV token typu '_15m_' / '_1h_'… a vrátí počet minut základního intervalu."""
    s = Path(path).name.lower()
    m = re.search(r"_(\d+)([mhdw])_", s)
    if not m:
        return None
    num = int(m.group(1))
    unit = m.group(2)
    return {'m': num, 'h': 60 * num, 'd': 1440 * num, 'w': 10080 * num}[unit]


def parse_multi_horizons(token: str | None, base_minutes: int | None) -> list[int] | None:
    """Rozparsuje '--multi-horizons' na seznam H v barech. Podporuje čísla i štítky (15m,1h,…)."""
    if token is None:
        return None
    tokens = [x.strip().lower() for x in token.split(",") if x.strip() != ""]
    out: list[int] = []
    for t in tokens:
        if t.isdigit():
            out.append(int(t))
            continue
        # časové štítky → vyžadují base_minutes
        if base_minutes is None:
            raise ValueError("--multi-horizons používá časové štítky, ale nepodařilo se detekovat základní interval z názvu CSV.")
        num_str = "".join(ch for ch in t if ch.isdigit())
        unit = "".join(ch for ch in t if ch.isalpha())
        if not num_str or not unit:
            raise ValueError(f"Neplatný token v --multi-horizons: {t}")
        num = int(num_str)
        minutes = {'m': num, 'h': 60 * num, 'd': 1440 * num, 'w': 10080 * num}[unit]
        bars = max(1, round(minutes / base_minutes))
        out.append(bars)
    return sorted(set(out))


def extract_variants_and_horizons(df: pd.DataFrame):
    """Najde sloupce pred_<variant>_h<H> a vrátí {variant: [H,...]}."""
    variants: dict[str, list[int]] = {}
    for c in df.columns:
        m = re.match(r"^pred_(?P<var>[A-Za-z0-9_]+)_h(?P<h>\d+)$", c)
        if m:
            v = m.group("var")
            h = int(m.group("h"))
            variants.setdefault(v, []).append(h)
    for v in variants:
        variants[v] = sorted(set(variants[v]))
    return variants


def safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if len(arr) > 0 else float('nan')


# ----------------- metriky -----------------
def compute_directional_rows(df: pd.DataFrame, variant: str, H: int, args):
    """Spočítá metriky pro zvolený variant+H. Vrací (row_dict, buckets_df|None)."""
    df = df.copy()
    if "close" not in df.columns:
        raise RuntimeError("CSV neobsahuje sloupec 'close'.")

    pred_col = f"pred_{variant}_h{H}"
    if pred_col not in df.columns:
        return None, None

    # budoucí close
    df["close_fwd"] = df["close"].shift(-H)
    df = df.dropna(subset=["close_fwd"]).reset_index(drop=True)

    # delty
    df["pred_delta"] = df[pred_col] - df["close"]
    df["true_delta"] = df["close_fwd"] - df["close"]

    # signální metriky (na všech řádcích s dostupnou budoucností)
    s_pred = np.sign(df["pred_delta"].values)
    s_true = np.sign(df["true_delta"].values)
    mask_nonzero = (s_true != 0)

    sign_acc = safe_mean((s_pred[mask_nonzero] == s_true[mask_nonzero]).astype(float))

    up_mask = (s_true > 0)
    down_mask = (s_true < 0)
    recall_up = safe_mean((s_pred[up_mask] == 1).astype(float)) if np.any(up_mask) else float('nan')
    recall_dn = safe_mean((s_pred[down_mask] == -1).astype(float)) if np.any(down_mask) else float('nan')
    sign_bacc = np.nanmean([recall_up, recall_dn])
    pred_up_mask = (s_pred == 1)
    precision_up = safe_mean((s_true[pred_up_mask] == 1).astype(float)) if np.any(pred_up_mask) else float('nan')

    MAE_ret = float(np.mean(np.abs(df["true_delta"].values)))
    MSE_ret = float(np.mean((df["true_delta"].values) ** 2))

    # Weighted sign-accuracy (váhy dle velikosti pohybu)
    w_sign_acc = None
    if args.weighted_sign_acc:
        weights = np.abs(df["true_delta"].values) ** float(args.wexp)
        weights = np.where(mask_nonzero, weights, 0.0)
        w_den = float(np.sum(weights))
        if w_den > 0:
            w_num = float(np.sum(weights * (s_pred == s_true)))
            w_sign_acc = w_num / w_den

    # ----- Obchodní filtry -----
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

    # PnL (jednoduchý model: size=1, open t, close t+H), fee = dvě strany
    fee = (args.fee_bps / 10000.0) * 2.0
    signed_ret = np.sign(df["pred_delta"].values) * df["true_delta"].values
    fee_cost = fee * df["close"].values
    pnl = signed_ret - fee_cost
    pnl = pnl[trade_mask]
    trades = int(np.sum(trade_mask))
    pnl_total = float(np.sum(pnl)) if trades > 0 else 0.0
    pnl_avg = float(np.mean(pnl)) if trades > 0 else 0.0

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
        "w_sign_acc": (float(w_sign_acc) if (w_sign_acc is not None) else None),
        "trades": trades,
        "pnl_total": pnl_total,
        "pnl_avg": pnl_avg,
    }

    # Buckets (kvantily podle |true_delta|)
    buckets_df = None
    if args.buckets and args.buckets > 0:
        q = np.linspace(0, 1, args.buckets + 1)
        edges = np.quantile(np.abs(df["true_delta"].values), q)
        edges = np.unique(edges)  # ochrana proti duplicitním hranám
        if len(edges) >= 2:
            idx = np.digitize(np.abs(df["true_delta"].values), edges[1:-1], right=True)
            recs = []
            for b in range(len(edges) - 1):
                mask = (idx == b)
                if np.sum(mask) == 0:
                    continue
                sacc = safe_mean((s_pred[mask] == s_true[mask]).astype(float))
                mean_mag = float(np.mean(np.abs(df["true_delta"].values[mask])))
                recs.append({
                    "bucket": b,
                    "edge_low": float(edges[b]),
                    "edge_high": float(edges[b + 1]),
                    "count": int(np.sum(mask)),
                    "sign_acc": float(sacc),
                    "mean_abs_move": mean_mag
                })
            if recs:
                buckets_df = pd.DataFrame.from_records(recs)

    return row, buckets_df


# ----------------- main -----------------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    variants = extract_variants_and_horizons(df)

    base_minutes = infer_base_minutes_from_name(args.csv)
    sel_horizons = parse_multi_horizons(args.multi_horizons, base_minutes)

    rows = []
    buckets_all = []

    for var, horizons in variants.items():
        for H in horizons:
            if sel_horizons is not None and H not in sel_horizons:
                continue
            r, btab = compute_directional_rows(df, var, H, args)
            if r is not None:
                rows.append(r)
            if btab is not None and len(btab) > 0:
                btab.insert(0, "variant", var)
                btab.insert(1, "horizon_bars", H)
                buckets_all.append(btab)

    if not rows:
        print("Nenalezeny žádné metriky (zkontroluj --multi-horizons vs. CSV).")
        return

    out = pd.DataFrame(rows).sort_values(["horizon_bars", "variant"]).reset_index(drop=True)
    print("\n=== Fractal directional leaderboard — sign_acc (vyšší lepší) ===")
    cols = [
        "variant", "horizon_bars", "N",
        "sign_acc", "sign_bacc", "w_sign_acc",
        "precision_up", "recall_up",
        "MAE_ret", "MSE_ret",
        "trades", "pnl_total", "pnl_avg"
    ]
    def fmt_w(x):
        return f"{x:.6f}" if (x is not None and x == x) else "   n/a "
    print(out.to_string(
        index=False,
        columns=cols,
        justify="right",
        formatters={
            "sign_acc": "{:.6f}".format,
            "sign_bacc": "{:.6f}".format,
            "w_sign_acc": fmt_w,
            "precision_up": "{:.6f}".format,
            "recall_up": "{:.6f}".format,
            "MAE_ret": "{:.6f}".format,
            "MSE_ret": "{:.6f}".format,
            "pnl_total": "{:.6f}".format,
            "pnl_avg": "{:.6f}".format,
        }
    ))

    base = Path(args.csv).stem.replace("forecast_", "")
    out_csv = outdir / f"{base}_metrics_fractal_plus.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nUloženo: {out_csv}")

    if buckets_all:
        buckets_df = pd.concat(buckets_all, ignore_index=True)
        bcsv = outdir / f"{base}_metrics_fractal_buckets.csv"
        buckets_df.to_csv(bcsv, index=False)
        print(f"Uloženo (buckets): {bcsv}")


if __name__ == "__main__":
    main()

