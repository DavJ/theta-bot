#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_eval_quaternion_ridge_v2.py
---------------------------------
Quaternionic theta + ridge (v2):
- Deterministic Fourier grid: ω_m = 2π m / window
- EMA weighting within each fit window (W^{1/2})
- Weighted ridge: β = (Xw^T Xw + λ I)^{-1} Xw^T yw
- Optional per-frequency block normalization (4-d blocks) before standardization
- Rich diagnostics (anti/zero rate, corr of cumulative price)

Usage example:
python theta_eval_quaternion_ridge_v2.py \
  --csv prices/BTCUSDT_1h.csv --price-col close \
  --horizon 1 --window 256 --q 0.65 --n-terms 16 --n-freq 6 \
  --lambda 0.3 --phase-scale 1.0 --ema-alpha 0.02 --block-norm \
  --outdir results_quat_v2_btc
"""

import argparse
import os
import numpy as np
import pandas as pd


# ------------- Jacobi theta (real-valued with real z) -------------

def theta1_real(z: float, q: float, n_terms: int = 16) -> float:
    s = 0.0
    for n in range(n_terms):
        w = ((-1.0) ** n) * (q ** ((n + 0.5) ** 2))
        s += w * np.sin((2*n + 1) * z)
    return 2.0 * s

def theta2_real(z: float, q: float, n_terms: int = 16) -> float:
    s = 0.0
    for n in range(n_terms):
        w = q ** ((n + 0.5) ** 2)
        s += w * np.cos((2*n + 1) * z)
    return 2.0 * s

def theta3_real(z: float, q: float, n_terms: int = 16) -> float:
    s = 1.0
    for n in range(1, n_terms + 1):
        w = q ** (n ** 2)
        s += 2.0 * w * np.cos(2 * n * z)
    return s

def theta4_real(z: float, q: float, n_terms: int = 16) -> float:
    s = 1.0
    for n in range(1, n_terms + 1):
        w = ((-1.0) ** n) * (q ** (n ** 2))
        s += 2.0 * w * np.cos(2 * n * z)
    return s


# ------------- Quaternionic basis (stacked 4-d blocks) -------------

def quaternion_theta_components(z: float, q: float, n_terms: int):
    a = theta3_real(z, q, n_terms)
    b = theta4_real(z, q, n_terms)
    c = theta2_real(z, q, n_terms)
    d = theta1_real(z, q, n_terms)
    return a, b, c, d

def build_feature_matrix_quat(T: int, freqs: np.ndarray, q: float, n_terms: int, phase_scale: float = 1.0):
    K = len(freqs); D = 4 * K
    X = np.zeros((T, D), dtype=np.float64)
    for t in range(T):
        col = 0
        for omega in freqs:
            z = (phase_scale * t) * float(omega)
            X[t, col:col+4] = quaternion_theta_components(z, q, n_terms)
            col += 4
    return X


# ------------- Helpers -------------

def hit_rate(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    s_true = np.sign(y_true); s_hat = np.sign(y_hat)
    mask = (s_true != 0)
    if mask.sum() == 0: return np.nan
    return float((s_true[mask] == s_hat[mask]).mean())

def anti_hit_rate(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    s_true = np.sign(y_true); s_hat = np.sign(y_hat)
    mask = (s_true != 0)
    if mask.sum() == 0: return np.nan
    return float((s_true[mask] == -s_hat[mask]).mean())

def zero_rate(y_hat: np.ndarray) -> float:
    return float((np.sign(y_hat) == 0).mean())

def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0: return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0); sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma

def block_normalize_inplace(X: np.ndarray, block_size: int = 4):
    # Normalize each 4-d frequency block to have unit average L2 norm across rows.
    T, D = X.shape; K = D // block_size
    for k in range(K):
        cols = slice(k*block_size, (k+1)*block_size)
        block = X[:, cols]
        # scale so that mean L2 norm over rows is 1
        norms = np.linalg.norm(block, axis=1)
        s = norms.mean()
        if s > 0:
            X[:, cols] = block / s


# ------------- Weighted ridge -------------

def fit_weighted_ridge(X_fit_std: np.ndarray, y_fit: np.ndarray, ema_alpha: float, lam: float):
    """
    Apply EMA weights within window (newest weight ~1, oldest ~(1-alpha)^{W-1}).
    Weighted features: Xw = W^{1/2} * X_fit_std
    """
    W = X_fit_std.shape[0]
    # weights oldest->newest: (1-alpha)^{W-1-i}
    if ema_alpha <= 0:
        w = np.ones(W, dtype=np.float64)
    else:
        base = (1.0 - ema_alpha)
        exps = np.arange(W-1, -1, -1, dtype=np.float64)  # newest first? we want oldest small, newest large
        w = base ** exps
    w_sqrt = np.sqrt(w).reshape(-1, 1)
    Xw = X_fit_std * w_sqrt
    yw = y_fit * w_sqrt[:, 0]

    XtX = Xw.T @ Xw
    reg = lam * np.eye(XtX.shape[0], dtype=np.float64)
    beta = np.linalg.solve(XtX + reg, Xw.T @ yw)
    return beta


# ------------- Main evaluation -------------

def evaluate(csv_path: str,
             price_col: str,
             horizon: int,
             window: int,
             q: float,
             n_terms: int,
             n_freq: int,
             lam: float,
             outdir: str,
             phase_scale: float = 1.0,
             ema_alpha: float = 0.02,
             block_norm: bool = False):

    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path} (cwd={os.getcwd()})")

    df = pd.read_csv(csv_path)
    if price_col not in df.columns:
        candidates = ["close", "Close", "close_price", "Close Price", "c"]
        found = [c for c in candidates if c in df.columns]
        if found:
            price_col = found[0]
            print(f"[warn] price_col not found, using autodetected '{price_col}'")
        else:
            raise ValueError(f"price_col '{price_col}' not in CSV columns: {list(df.columns)}")

    p = df[price_col].astype(float).values
    T = len(p)
    if T <= window + horizon + 2:
        raise ValueError(f"Not enough samples (T={T}) for window={window} and horizon={horizon}.")

    y = p[horizon:] - p[:-horizon]  # delta target
    T_y = len(y)

    # Fourier grid tied to window
    m = np.arange(1, n_freq + 1, dtype=np.float64)
    freqs = 2.0 * np.pi * m / float(window)

    # Build features
    X_full = build_feature_matrix_quat(T, freqs=freqs, q=q, n_terms=n_terms, phase_scale=phase_scale)
    if block_norm:
        block_normalize_inplace(X_full, block_size=4)

    X = X_full[:-horizon, :]

    y_hats = []; y_trues = []; t_indices = []
    diag_rows = []; beta_accum = []

    for t0 in range(window, T_y):
        lo, hi = t0 - window, t0
        X_fit = X[lo:hi, :]
        y_fit = y[lo:hi]
        x_now = X[t0, :]
        y_true = y[t0]

        # standardize by window
        mu, sigma = standardize_fit(X_fit)
        X_fit_std = standardize_apply(X_fit, mu, sigma)
        x_now_std = standardize_apply(x_now.reshape(1, -1), mu, sigma)[0]

        # conditioning
        try:
            cond = float(np.linalg.cond(X_fit_std))
        except Exception:
            cond = np.nan

        beta = fit_weighted_ridge(X_fit_std, y_fit, ema_alpha=ema_alpha, lam=lam)
        y_hat = float(x_now_std @ beta)

        # in-window diagnostics
        ins_corr = corr_safe(y_fit, X_fit_std @ beta)
        ins_hr = hit_rate(y_fit, X_fit_std @ beta)

        y_hats.append(y_hat); y_trues.append(y_true); t_indices.append(t0)
        diag_rows.append({
            "t_index": int(t0),
            "cond_X": cond,
            "beta_norm": float(np.linalg.norm(beta)),
            "x_now_norm": float(np.linalg.norm(x_now_std)),
            "y_fit_mean": float(np.mean(y_fit)),
            "y_fit_std": float(np.std(y_fit)),
            "insample_corr": ins_corr,
            "insample_hit_rate": ins_hr,
            "y_true": float(y_true),
            "y_hat": float(y_hat),
        })
        beta_accum.append(np.abs(beta))

    y_hats = np.array(y_hats); y_trues = np.array(y_trues); t_indices = np.array(t_indices, int)

    # extra metrics
    hr = hit_rate(y_trues, y_hats)
    anti = anti_hit_rate(y_trues, y_hats)
    zero = zero_rate(y_hats)

    # corr of cumulative price levels
    # rebuild price aligned with predictions indices: start at p[window] baseline
    base_price = p[window]
    y_hat_cum = base_price + np.cumsum(y_hats)
    y_true_cum = base_price + np.cumsum(y_trues)
    corr_price = corr_safe(y_true_cum, y_hat_cum)

    metrics = {
        "n_samples": int(len(y_trues)),
        "corr_pred_true": corr_safe(y_trues, y_hats),
        "hit_rate": hr,
        "anti_hit_rate": anti,
        "zero_rate": zero,
        "corr_price": corr_price,
        "mae": float(np.mean(np.abs(y_trues - y_hats))),
        "rmse": float(np.sqrt(np.mean((y_trues - y_hats) ** 2))),
        "window": int(window),
        "horizon": int(horizon),
        "q": float(q),
        "n_terms": int(n_terms),
        "n_freq": int(n_freq),
        "lambda": float(lam),
        "phase_scale": float(phase_scale),
        "ema_alpha": float(ema_alpha),
        "block_norm": bool(block_norm),
    }

    # Save outputs
    outdir = os.path.abspath(outdir)
    pd.DataFrame({"t_index": t_indices, "y_true": y_trues, "y_hat": y_hats}).to_csv(
        os.path.join(outdir, "predictions.csv"), index=False
    )
    pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "summary_quat_ridge_v2.csv"), index=False)
    pd.DataFrame(diag_rows).to_csv(os.path.join(outdir, "diagnostics.csv"), index=False)

    if len(beta_accum) > 0:
        avg_beta = np.mean(np.vstack(beta_accum), axis=0)
        pd.DataFrame({"feature_index": np.arange(len(avg_beta)), "avg_abs_beta": avg_beta}).to_csv(
            os.path.join(outdir, "feature_importances.csv"), index=False
        )

    print("[summary_v2]", metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--price-col", default="close")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--q", type=float, default=0.65)
    ap.add_argument("--n-terms", type=int, default=16)
    ap.add_argument("--n-freq", type=int, default=6)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.3)
    ap.add_argument("--phase-scale", type=float, default=1.0)
    ap.add_argument("--ema-alpha", type=float, default=0.02, help="EMA weight alpha in (0,1). 0 disables weighting.")
    ap.add_argument("--block-norm", action="store_true")
    ap.add_argument("--outdir", default="results_quat_v2")
    args = ap.parse_args()

    evaluate(csv_path=args.csv,
             price_col=args.price_col,
             horizon=args.horizon,
             window=args.window,
             q=args.q,
             n_terms=args.n_terms,
             n_freq=args.n_freq,
             lam=args.lam,
             outdir=args.outdir,
             phase_scale=args.phase_scale,
             ema_alpha=args.ema_alpha,
             block_norm=args.block_norm)


if __name__ == "__main__":
    main()
