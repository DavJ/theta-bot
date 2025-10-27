#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_eval_quaternion_ridge_v3.py
---------------------------------
Quaternionic theta + ridge (v3):
- Deterministic Fourier grid: ω_m = 2π m / window
- EMA weighting within the window (W^{1/2})
- Optional **weighted QR** orthogonalization before ridge (--weighted-qr)
- **Per-frequency block regularization** via --lambda-blocks "1,0.8,1.2" (scales λ per 4-d block)
- Block normalization (--block-norm)
- Rich diagnostics (anti/zero rate, corr of cumulative price)
- **Grid tuner** (--grid) that explores a small neighborhood around the provided hyperparameters
  and writes a CSV ranking of runs by corr_pred_true and hit_rate.

Usage example (single run):
python theta_eval_quaternion_ridge_v3.py \
  --csv prices/BTCUSDT_1h.csv --price-col close \
  --horizon 1 --window 256 --q 0.6 --n-terms 16 --n-freq 6 \
  --lambda 0.6 --phase-scale 1.1 --ema-alpha 0.03 \
  --block-norm --weighted-qr \
  --outdir theta_bot_averaging/results_quat_btc_v3

Usage example (grid tuner):
python theta_eval_quaternion_ridge_v3.py \
  --csv prices/BTCUSDT_1h.csv --price-col close \
  --horizon 1 --window 256 --q 0.6 --n-terms 16 --n-freq 6 \
  --lambda 0.6 --phase-scale 1.1 --ema-alpha 0.03 \
  --block-norm --weighted-qr --grid \
  --outdir theta_bot_averaging/results_quat_btc_v3_grid
"""

import argparse
import os
import itertools
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

def block_normalize_inplace(X: np.ndarray, block_size: int = 4):
    T, D = X.shape; K = D // block_size
    for k in range(K):
        cols = slice(k*block_size, (k+1)*block_size)
        block = X[:, cols]
        norms = np.linalg.norm(block, axis=1)
        s = norms.mean()
        if s > 0:
            X[:, cols] = block / s


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


# ------------- Weighted QR + Ridge -------------

def build_weights(W: int, ema_alpha: float) -> np.ndarray:
    if ema_alpha <= 0:
        return np.ones(W, dtype=np.float64)
    base = (1.0 - ema_alpha)
    exps = np.arange(W-1, -1, -1, dtype=np.float64)  # oldest -> newest
    w = base ** exps
    return w

def solve_ridge_weighted_qr(X_fit_std: np.ndarray, y_fit: np.ndarray,
                            lam: float, lam_blocks: np.ndarray | None,
                            weighted_qr: bool, ema_alpha: float) -> np.ndarray:
    Wn, D = X_fit_std.shape
    w = build_weights(Wn, ema_alpha)
    w_sqrt = np.sqrt(w).reshape(-1, 1)
    Xw = X_fit_std * w_sqrt
    yw = y_fit * w_sqrt[:, 0]

    if weighted_qr:
        # QR on weighted features
        Q, R = np.linalg.qr(Xw, mode='reduced')  # Xw = Q R
        # Ridge in R-space: (R^T R + Λ) β = R^T Q^T y_w
        RtR = R.T @ R
        if lam_blocks is not None:
            # lam_blocks repeats 4 times per frequency block
            Reg = (lam * np.diag(lam_blocks))
        else:
            Reg = lam * np.eye(D, dtype=np.float64)
        rhs = R.T @ (Q.T @ yw)
        beta = np.linalg.solve(RtR + Reg, rhs)
        return beta
    else:
        XtX = Xw.T @ Xw
        if lam_blocks is not None:
            Reg = (lam * np.diag(lam_blocks))
        else:
            Reg = lam * np.eye(D, dtype=np.float64)
        beta = np.linalg.solve(XtX + Reg, Xw.T @ yw)
        return beta


# ------------- One evaluation run -------------

def evaluate_once(p: np.ndarray,
                  window: int, horizon: int,
                  q: float, n_terms: int, n_freq: int, lam: float,
                  phase_scale: float, ema_alpha: float, block_norm: bool,
                  weighted_qr: bool, lam_blocks_vec: np.ndarray | None):

    T = len(p)
    if T <= window + horizon + 2:
        raise ValueError(f"Not enough samples (T={T}) for window={window} and horizon={horizon}.")

    y = p[horizon:] - p[:-horizon]
    T_y = len(y)

    # Fourier grid (tied to window)
    m = np.arange(1, n_freq + 1, dtype=np.float64)
    freqs = 2.0 * np.pi * m / float(window)

    # Build features
    X_full = build_feature_matrix_quat(T, freqs=freqs, q=q, n_terms=n_terms, phase_scale=phase_scale)
    if block_norm:
        block_normalize_inplace(X_full, block_size=4)
    X = X_full[:-horizon, :]

    # Walk-forward
    y_hats = []; y_trues = []; t_indices = []
    for t0 in range(window, T_y):
        lo, hi = t0 - window, t0
        X_fit = X[lo:hi, :]
        y_fit = y[lo:hi]
        x_now = X[t0, :]
        y_true = y[t0]

        mu, sigma = standardize_fit(X_fit)
        X_fit_std = standardize_apply(X_fit, mu, sigma)
        x_now_std = standardize_apply(x_now.reshape(1, -1), mu, sigma)[0]

        beta = solve_ridge_weighted_qr(
            X_fit_std, y_fit,
            lam=lam, lam_blocks=lam_blocks_vec,
            weighted_qr=weighted_qr, ema_alpha=ema_alpha
        )
        y_hat = float(x_now_std @ beta)

        y_hats.append(y_hat); y_trues.append(y_true); t_indices.append(t0)

    y_hats = np.array(y_hats); y_trues = np.array(y_trues); t_indices = np.array(t_indices, int)

    # Extra metrics
    hr = hit_rate(y_trues, y_hats)
    anti = anti_hit_rate(y_trues, y_hats)
    zero = zero_rate(y_hats)
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
        "weighted_qr": bool(weighted_qr),
    }
    return metrics, t_indices, y_trues, y_hats


# ------------- High-level evaluate (IO + grid) -------------

def parse_lambda_blocks(n_freq: int, lam_blocks_str: str | None) -> np.ndarray | None:
    if not lam_blocks_str: return None
    parts = [float(x.strip()) for x in lam_blocks_str.split(",") if x.strip()]
    if len(parts) != n_freq:
        raise ValueError(f"--lambda-blocks expects {n_freq} values (got {len(parts)}).")
    # expand to 4-d per block
    vec = []
    for s in parts:
        vec.extend([s, s, s, s])
    return np.array(vec, dtype=np.float64)

def save_run(outdir: str, name: str, metrics: dict, t_indices, y_trues, y_hats):
    os.makedirs(outdir, exist_ok=True)
    # summary
    summ = pd.DataFrame([metrics])
    summ["run_name"] = name
    summ_path = os.path.join(outdir, f"summary_{name}.csv")
    summ.to_csv(summ_path, index=False)
    # preds
    pred = pd.DataFrame({"t_index": t_indices, "y_true": y_trues, "y_hat": y_hats})
    pred_path = os.path.join(outdir, f"predictions_{name}.csv")
    pred.to_csv(pred_path, index=False)
    return summ_path, pred_path

def evaluate(csv_path: str, price_col: str, horizon: int, window: int,
             q: float, n_terms: int, n_freq: int, lam: float, outdir: str,
             phase_scale: float, ema_alpha: float, block_norm: bool,
             weighted_qr: bool, lambda_blocks: str | None,
             grid: bool):

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

    lam_blocks_vec = parse_lambda_blocks(n_freq, lambda_blocks)

    if not grid:
        # single run
        metrics, t_idx, y_true, y_hat = evaluate_once(
            p=p, window=window, horizon=horizon, q=q, n_terms=n_terms, n_freq=n_freq, lam=lam,
            phase_scale=phase_scale, ema_alpha=ema_alpha, block_norm=block_norm,
            weighted_qr=weighted_qr, lam_blocks_vec=lam_blocks_vec
        )
        name = "main"
        save_run(outdir, name, metrics, t_idx, y_true, y_hat)
        print("[summary_v3]", metrics)
        return

    # grid search
    # define small neighborhoods
    def clip_q(x): return float(np.clip(x, 0.4, 0.9))
    q_vals = sorted(set([clip_q(q*0.9), q, clip_q(q*1.1)]))
    lam_vals = sorted(set([max(1e-6, lam*0.5), lam, lam*1.5]))
    phase_vals = sorted(set([phase_scale*0.95, phase_scale, phase_scale*1.05]))
    ema_vals = sorted(set([max(0.0, ema_alpha-0.01), ema_alpha, min(0.2, ema_alpha+0.01)]))
    nfreq_vals = sorted(set([max(4, n_freq-2), n_freq, min(12, n_freq+2)]))
    window_vals = sorted(set([max(128, window-64), window, window+64]))

    grid_rows = []
    all_summaries = []
    best_corr = None; best_hr = None

    for (qv, lamv, phv, emav, nf, win) in itertools.product(q_vals, lam_vals, phase_vals, ema_vals, nfreq_vals, window_vals):
        try:
            lam_blocks_vec = parse_lambda_blocks(nf, lambda_blocks)
            metrics, t_idx, y_true, y_hat = evaluate_once(
                p=p, window=win, horizon=horizon, q=qv, n_terms=n_terms, n_freq=nf, lam=lamv,
                phase_scale=phv, ema_alpha=emav, block_norm=block_norm,
                weighted_qr=weighted_qr, lam_blocks_vec=lam_blocks_vec
            )
            run_name = f"q{qv:.3f}_lam{lamv:.3f}_ph{phv:.3f}_ema{emav:.3f}_nf{nf}_win{win}"
            save_run(outdir, run_name, metrics, t_idx, y_true, y_hat)
            row = dict(metrics); row["run_name"] = run_name
            grid_rows.append(row)
        except Exception as e:
            row = {"run_name": f"ERR_win{win}_nf{nf}", "error": str(e)}
            grid_rows.append(row)

    grid_df = pd.DataFrame(grid_rows)
    # rankings
    if "corr_pred_true" in grid_df.columns:
        best_by_corr = grid_df.dropna(subset=["corr_pred_true"]).sort_values("corr_pred_true", ascending=False).head(10)
    else:
        best_by_corr = pd.DataFrame()
    if "hit_rate" in grid_df.columns:
        best_by_hr = grid_df.dropna(subset=["hit_rate"]).sort_values("hit_rate", ascending=False).head(10)
    else:
        best_by_hr = pd.DataFrame()

    grid_path = os.path.join(outdir, "grid_results.csv")
    grid_df.to_csv(grid_path, index=False)

    top_path = os.path.join(outdir, "grid_top.csv")
    with open(top_path, "w", encoding="utf-8") as f:
        f.write("# Top by corr_pred_true\n")
        if not best_by_corr.empty:
            best_by_corr.to_csv(f, index=False)
        f.write("\n# Top by hit_rate\n")
        if not best_by_hr.empty:
            best_by_hr.to_csv(f, index=False)

    print(f"[grid] Saved {len(grid_rows)} runs to: {grid_path}")
    print(f"[grid] Top tables saved to: {top_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--price-col", default="close")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--q", type=float, default=0.6)
    ap.add_argument("--n-terms", type=int, default=16)
    ap.add_argument("--n-freq", type=int, default=6)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.6)
    ap.add_argument("--phase-scale", type=float, default=1.1)
    ap.add_argument("--ema-alpha", type=float, default=0.03)
    ap.add_argument("--block-norm", action="store_true")
    ap.add_argument("--weighted-qr", action="store_true")
    ap.add_argument("--lambda-blocks", type=str, default=None, help="Comma-separated multipliers per frequency (length = n_freq).")
    ap.add_argument("--grid", action="store_true", help="Run small grid search around the provided hyperparameters.")
    ap.add_argument("--outdir", default="results_quat_v3")
    args = ap.parse_args()

    # Load CSV prices
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
             block_norm=args.block_norm,
             weighted_qr=args.weighted_qr,
             lambda_blocks=args.lambda_blocks,
             grid=args.grid)


if __name__ == "__main__":
    main()
