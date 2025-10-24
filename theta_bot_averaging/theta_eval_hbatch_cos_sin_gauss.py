
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# Helpery
# -----------------------------

def _read_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # zkus různé názvy času
    tcol = None
    for c in ["time", "timestamp", "date", "datetime", "Time", "Timestamp"]:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        # fallback: index na pořadí
        df["time"] = np.arange(len(df))
        tcol = "time"
    # zkus různé názvy closu
    ccol = None
    for c in ["close", "Close", "price", "Price", "close_price"]:
        if c in df.columns:
            ccol = c
            break
    if ccol is None:
        raise ValueError(f"'{path}': nenašel jsem sloupec s close cenou.")
    # normalize
    out = df[[tcol, ccol]].copy()
    out.columns = ["time", "close"]
    return out


def _make_period_grid(minP: int, maxP: int, nP: int) -> np.ndarray:
    # log-space mřížka period
    return np.unique(np.round(np.geomspace(minP, maxP, nP)).astype(int))


def _build_features(closes: np.ndarray, window: int, periods: np.ndarray, sigma: float) -> np.ndarray:
    """
    Sin/Cos báze přes okno s jemným Gauss vážením (sigma).
    Výstup: X_all shape [N, 2*nP], kde pro každé P máme (cos, sin).
    """
    N = len(closes)
    nP = len(periods)
    X = np.zeros((N, 2 * nP), dtype=float)

    # předpočítat váhy okna (Gauss kolem konce okna)
    w_idx = np.arange(window, dtype=float)
    # centrováno na konec okna (větší váha pro čerstvé body)
    mu = window - 1.0
    std = max(1.0, sigma * window)
    gauss = np.exp(-0.5 * ((w_idx - mu) / std) ** 2)
    gauss /= gauss.sum()

    for i in range(window - 1, N):
        seg = closes[i - window + 1 : i + 1]
        if seg.shape[0] != window:
            continue
        seg = seg.astype(float)
        # z-normalizace okna (stabilita)
        seg = (seg - seg.mean()) / (seg.std() + 1e-9)

        col = 0
        for P in periods:
            # úhlová frekvence ~ 2π / P
            t = np.arange(window, dtype=float)
            omega = 2.0 * math.pi / float(P)
            cos_b = np.cos(omega * t)
            sin_b = np.sin(omega * t)
            # projekce s Gauss váhami
            X[i, col]   = float(np.sum(seg * cos_b * gauss))
            X[i, col+1] = float(np.sum(seg * sin_b * gauss))
            col += 2

    return X


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    # řešení ridge (X^T X + lam I)^-1 X^T y
    XtX = X.T @ X
    d = XtX.shape[0]
    beta = np.linalg.solve(XtX + lam * np.eye(d), X.T @ y)
    return beta


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _safe_stem_key(path: str) -> str:
    # pro jména eval souborů jako "eval_h_BTCUSDT_1HCSV.csv"
    base = Path(path).name
    stem = Path(path).stem
    return base.replace(".", "").upper(), stem.replace(".", "").upper()


# -----------------------------
# Jádro evaluace (bez leaků)
# -----------------------------

def evaluate_symbol_csv(path: str,
                        interval: str,
                        window: int,
                        horizon: int,
                        minP: int,
                        maxP: int,
                        nP: int,
                        sigma: float,
                        lam: float,
                        pred_ensemble: str,
                        max_by: str):
    df_in = _read_price_csv(path)
    times = df_in["time"].to_numpy()
    closes = df_in["close"].to_numpy().astype(float)

    periods = _make_period_grid(minP, maxP, nP)
    X_all = _build_features(closes, window, periods, sigma)

    rows = []
    preds, trues, lasts = [], [], []

    # walk-forward: pro každé 'hi' predikujeme delta v horizontu h
    # trénink NA MINULOSTI: hi_tr = hi - horizon
    for hi in range(window, len(closes) - horizon):
        lo = hi - window
        hi_tr = hi - horizon
        if hi_tr - lo < 8:  # minimální počet vzorků pro fit
            continue

        Xw = X_all[lo:hi_tr, :]
        yw = (closes[lo + horizon : hi_tr + horizon] - closes[lo:hi_tr]).astype(float)

        # vynech okna, kde nejsou hotové features
        if Xw.shape[0] != yw.shape[0] or Xw.shape[0] <= 2:
            continue

        beta = _ridge_fit(Xw, yw, lam)

        # rozhodnutí děláme na poslední UZAVŘENÉ svíčce
        x_now = X_all[hi - 1, :]
        if not np.any(x_now):
            continue

        last_price = float(closes[hi - 1])
        future_price = float(closes[hi - 1 + horizon])
        true_delta = future_price - last_price

        if pred_ensemble == "avg":
            pred_delta = float(x_now @ beta)
        else:
            # jednoduchý výběr "nejpřínosnějšího" páru (cos,sin) podle |beta|
            contrib = []
            for k in range(len(periods)):
                sl = slice(2 * k, 2 * k + 2)
                contrib.append(float(np.linalg.norm(beta[sl], ord=2)))
            k_best = int(np.argmax(contrib))
            sl = slice(2 * k_best, 2 * k_best + 2)
            pred_delta = float(x_now[sl] @ beta[sl])

        pred_dir = int(np.sign(pred_delta))
        true_dir = int(np.sign(true_delta))
        correct_pred_val = 1 if pred_dir == true_dir else 0

        rows.append({
            "time": str(times[hi - 1]),
            "entry_idx": int(hi - 1),
            "compare_idx": int(hi - 1 + horizon),
            "last_price": float(last_price),
            "pred_price": float(last_price + pred_delta),
            "future_price": float(future_price),
            "pred_delta": float(pred_delta),
            "true_delta": float(true_delta),
            "pred_dir": int(pred_dir),
            "true_dir": int(true_dir),
            "correct_pred": int(correct_pred_val),
        })

        preds.append(pred_delta)
        trues.append(true_delta)
        lasts.append(last_price)

    if len(rows) == 0:
        raise RuntimeError(f"No rows produced for {path}. Check data length vs. window/horizon.")

    df_eval = pd.DataFrame(rows)

    # výstupní CSV (eval_h_...)
    base_key, stem_key = _safe_stem_key(path)
    out_eval_csv = f"eval_h_{stem_key}.csv"
    wanted_cols = [
        "time","entry_idx","compare_idx",
        "last_price","pred_price","future_price",
        "pred_delta","true_delta","pred_dir","true_dir","correct_pred"
    ]
    df_eval[wanted_cols].to_csv(out_eval_csv, index=False)
    print(f"\nUloženo CSV: {out_eval_csv}")

    # metriky
    preds = np.array(preds, dtype=float)
    trues = np.array(trues, dtype=float)
    lasts = np.array(lasts, dtype=float)

    hit_rate_pred = float(np.mean(np.sign(preds) == np.sign(trues)))
    hit_rate_hold = float(np.mean((trues > 0).astype(int)))  # "hold" benchmark (směr > 0)
    corr_pred_true = _corr(preds, trues)
    mae_price = float(np.mean(np.abs(trues - preds)))
    mae_return = float(np.mean(np.abs((trues - preds) / (lasts + 1e-12))))
    count = int(len(trues))

    # summary JSON
    summary = {
        "symbol": path,
        "interval": interval,
        "window": window,
        "horizon": horizon,
        "minP": minP,
        "maxP": maxP,
        "nP": nP,
        "sigma": sigma,
        "lambda": lam,
        "pred_ensemble": pred_ensemble,
        "max_by": max_by,
        "hit_rate_pred": hit_rate_pred,
        "hit_rate_hold": hit_rate_hold,
        "corr_pred_true": corr_pred_true,
        "mae_price": mae_price,
        "mae_return": mae_return,
        "count": count,
    }
    out_json = f"sum_h_{stem_key}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Uloženo summary: {out_json}")

    # log výstup do konzole (jako dřív)
    print("\n\n--- HSTRATEGY vs HOLD ---")
    print(f"hit_rate_pred:  {hit_rate_pred:9.6f}")
    print(f"hit_rate_hold:  {hit_rate_hold:9.6f}")
    print(f"corr_pred_true: {corr_pred_true:9.6f}")
    print(f"mae_price:      {mae_price}")
    print(f"mae_return:     {mae_return}")
    print(f"count:          {count}\n")

    return df_eval, summary


def run_batch(symbol_paths: str,
              interval: str,
              window: int,
              horizon: int,
              minP: int,
              maxP: int,
              nP: int,
              sigma: float,
              lam: float,
              pred_ensemble: str,
              max_by: str):
    syms = [s.strip() for s in symbol_paths.split(",") if s.strip()]
    rows = []
    for sym in syms:
        print(f"\n=== Running {sym} ===\n")
        df_eval, summary = evaluate_symbol_csv(
            sym, interval, window, horizon,
            minP, maxP, nP, sigma, lam,
            pred_ensemble, max_by
        )
        rows.append(summary)

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True, help="CSV paths, comma-separated")
    p.add_argument("--interval", default="1h")
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--minP", type=int, default=24)
    p.add_argument("--maxP", type=int, default=480)
    p.add_argument("--nP", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.8)
    p.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    p.add_argument("--pred-ensemble", choices=["avg", "max"], default="avg")
    p.add_argument("--max-by", choices=["transform", "contrib"], default="transform")
    p.add_argument("--out", required=True, help="summary CSV path")
    return p.parse_args()


def main():
    args = parse_args()
    df = run_batch(
        args.symbols, args.interval, args.window, args.horizon,
        args.minP, args.maxP, args.nP, args.sigma, args.lam,
        args.pred_ensemble, args.max_by
    )
    # výstupní tabulka podobná předchozí
    cols = [
        "symbol","phase","pred_ensemble","max_by",
        "hit_rate_pred","hit_rate_hold","delta_hit",
        "corr_pred_true","mae_price","mae_return","count"
    ]
    out_df = pd.DataFrame({
        "symbol": df["symbol"],
        "phase": ["biquat"] * len(df),
        "pred_ensemble": df["pred_ensemble"],
        "max_by": df["max_by"],
        "hit_rate_pred": df["hit_rate_pred"],
        "hit_rate_hold": df["hit_rate_hold"],
        "delta_hit": df["hit_rate_pred"] - df["hit_rate_hold"],
        "corr_pred_true": df["corr_pred_true"],
        "mae_price": df["mae_price"],
        "mae_return": df["mae_return"],
        "count": df["count"].astype(int),
    }, columns=cols)
    out_path = args.out
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nUloženo: {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
