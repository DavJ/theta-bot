#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_eval_quaternion_ridge.py
------------------------------
Truly quaternionic (Hamilton) theta + ridge evaluator with causal walk-forward.

- Quaternions q = a + b i + c j + d k with real components (Hamilton algebra).
- Theta basis is built from *real-valued* Jacobi theta series (θ1..θ4) using real z = ω * (scale * t).
- Each frequency ω_k yields one quaternion feature q_k(t); the feature vector stacks all 4 components per ω_k.
- Regression target is real (delta price). Prediction is the Euclidean inner product Σ_k <β_k, q_k(t)>,
  which equals Re(Σ_k conj(β_k) * q_k(t)) in quaternion notation.

Outputs:
- {outdir}/predictions.csv
- {outdir}/summary_quat_ridge.csv
- {outdir}/diagnostics.csv
- {outdir}/feature_importances.csv
"""

import argparse
import os
import numpy as np
import pandas as pd


# ------------------------------
# Jacobi theta (q-series, real-valued with real argument z)
# ------------------------------

def theta1_real(z: float, q: float, n_terms: int = 16) -> float:
    # θ1(z,q) = 2 * Σ_{n=0}^∞ (-1)^n q^{(n+1/2)^2} * sin((2n+1) z)
    s = 0.0
    for n in range(n_terms):
        w = ((-1.0) ** n) * (q ** ((n + 0.5) ** 2))
        s += w * np.sin((2*n + 1) * z)
    return 2.0 * s


def theta2_real(z: float, q: float, n_terms: int = 16) -> float:
    # θ2(z,q) = 2 * Σ_{n=0}^∞ q^{(n+1/2)^2} * cos((2n+1) z)
    s = 0.0
    for n in range(n_terms):
        w = q ** ((n + 0.5) ** 2)
        s += w * np.cos((2*n + 1) * z)
    return 2.0 * s


def theta3_real(z: float, q: float, n_terms: int = 16) -> float:
    # θ3(z,q) = 1 + 2 * Σ_{n=1}^∞ q^{n^2} * cos(2 n z)
    s = 1.0
    for n in range(1, n_terms + 1):
        w = q ** (n ** 2)
        s += 2.0 * w * np.cos(2 * n * z)
    return s


def theta4_real(z: float, q: float, n_terms: int = 16) -> float:
    # θ4(z,q) = 1 + 2 * Σ_{n=1}^∞ (-1)^n q^{n^2} * cos(2 n z)
    s = 1.0
    for n in range(1, n_terms + 1):
        w = ((-1.0) ** n) * (q ** (n ** 2))
        s += 2.0 * w * np.cos(2 * n * z)
    return s


# ------------------------------
# Quaternion basis builder
# ------------------------------

def quaternion_theta_components(z: float, q: float, n_terms: int) -> tuple[float, float, float, float]:
    """
    Build a quaternion from theta series at phase z:
      a ← θ3(z,q), b ← θ4(z,q), c ← θ2(z,q), d ← θ1(z,q)
    All are real with real z.
    """
    a = theta3_real(z, q, n_terms)
    b = theta4_real(z, q, n_terms)
    c = theta2_real(z, q, n_terms)
    d = theta1_real(z, q, n_terms)
    return a, b, c, d


def build_feature_matrix_quat(T: int,
                              freqs: np.ndarray,
                              q: float,
                              n_terms: int,
                              phase_scale: float = 1.0) -> np.ndarray:
    """
    Build X ∈ R^{T × (4*K)} with stacked quaternion components per frequency.
    """
    K = len(freqs)
    D = 4 * K
    X = np.zeros((T, D), dtype=np.float64)
    for t in range(T):
        col = 0
        for omega in freqs:
            z = (phase_scale * t) * float(omega)
            a, b, c, d = quaternion_theta_components(z, q, n_terms)
            X[t, col:col+4] = (a, b, c, d)
            col += 4
    # normalize per-frequency block to avoid scale explosion (optional; keep raw for now)
    return X


# ------------------------------
# Ridge regression helpers (real-expanded)
# ------------------------------

def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    D = X.shape[1]
    XtX = X.T @ X
    reg = lam * np.eye(D, dtype=np.float64)
    beta = np.linalg.solve(XtX + reg, X.T @ y)
    return beta


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma


def hit_rate(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    s_true = np.sign(y_true)
    s_hat = np.sign(y_hat)
    mask = (s_true != 0)
    if mask.sum() == 0:
        return np.nan
    return float((s_true[mask] == s_hat[mask]).mean())


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------
# Main evaluation
# ------------------------------

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
             seed: int = 42,
             save_diagnostics: bool = True,
             save_feature_importance: bool = True):

    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path} (cwd={os.getcwd()})")

    df = pd.read_csv(csv_path)
    if price_col not in df.columns:
        # tiny autodetect
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

    y = p[horizon:] - p[:-horizon]  # target delta
    T_y = len(y)

    # Frequencies (deterministic spread + small jitter)
    rng = np.random.default_rng(seed)
    base = np.linspace(0.2, 1.8, n_freq)  # radians/step
    jitter = rng.uniform(-0.05, 0.05, size=n_freq)
    freqs = base + jitter

    # Build features
    X_full = build_feature_matrix_quat(T, freqs=freqs, q=q, n_terms=n_terms, phase_scale=phase_scale)
    X = X_full[:-horizon, :]  # align with y

    # Walk-forward with diagnostics
    y_hats, y_trues, t_indices = [], [], []
    diag_rows = []
    beta_accum = []

    for t0 in range(window, T_y):
        lo, hi = t0 - window, t0
        X_fit = X[lo:hi, :]
        y_fit = y[lo:hi]
        x_now = X[t0, :]
        y_true = y[t0]

        mu, sigma = standardize_fit(X_fit)
        X_fit_std = standardize_apply(X_fit, mu, sigma)
        x_now_std = standardize_apply(x_now.reshape(1, -1), mu, sigma)[0]

        # conditioning
        try:
            cond = float(np.linalg.cond(X_fit_std))
        except Exception:
            cond = np.nan

        beta = fit_ridge_closed_form(X_fit_std, y_fit, lam=lam)
        y_hat = float(x_now_std @ beta)

        # in-window diagnostics
        try:
            ins_corr = corr_safe(y_fit, X_fit_std @ beta)
            ins_hr = hit_rate(y_fit, X_fit_std @ beta)
        except Exception:
            ins_corr, ins_hr = np.nan, np.nan

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

    metrics = {
        "n_samples": int(len(y_trues)),
        "corr_pred_true": corr_safe(y_trues, y_hats),
        "hit_rate": hit_rate(y_trues, y_hats),
        "mae": float(np.mean(np.abs(y_trues - y_hats))),
        "rmse": float(np.sqrt(np.mean((y_trues - y_hats) ** 2))),
        "window": int(window),
        "horizon": int(horizon),
        "q": float(q),
        "n_terms": int(n_terms),
        "n_freq": int(n_freq),
        "lambda": float(lam),
        "phase_scale": float(phase_scale),
    }

    # Save outputs
    pd.DataFrame({"t_index": t_indices, "y_true": y_trues, "y_hat": y_hats}).to_csv(
        os.path.join(outdir, "predictions.csv"), index=False
    )
    pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "summary_quat_ridge.csv"), index=False)
    pd.DataFrame(diag_rows).to_csv(os.path.join(outdir, "diagnostics.csv"), index=False)

    if len(beta_accum) > 0:
        avg_beta = np.mean(np.vstack(beta_accum), axis=0)
        pd.DataFrame({"feature_index": np.arange(len(avg_beta)), "avg_abs_beta": avg_beta}).to_csv(
            os.path.join(outdir, "feature_importances.csv"), index=False
        )

    print("[summary]", metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--price-col", default="close")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--q", type=float, default=0.6)
    ap.add_argument("--n-terms", type=int, default=24)
    ap.add_argument("--n-freq", type=int, default=8)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--phase-scale", type=float, default=1.0)
    ap.add_argument("--outdir", default="results_quat")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
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
             seed=args.seed)


if __name__ == "__main__":
    main()
