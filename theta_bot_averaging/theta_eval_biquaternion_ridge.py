#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_eval_biquaternion_ridge.py
--------------------------------
Prototype evaluator for a *biquaternion-style* Jacobi-theta basis with walk-forward ridge projection.

Notes
-----
- This is a *prototype* that encodes a biquaternion as a 4-component complex vector aligned with (1, i, j, k).
- We do not rely on an external quaternion library; instead, we represent biquaternions as np.ndarray(shape=(4,), dtype=complex).
- For regression, we project the 4 complex components into a real feature vector by concatenating real and imaginary parts.
- Jacobi theta functions are approximated using truncated q-series (no external deps).
- Evaluation is strictly causal (walk-forward).

CLI
---
Example:
    python theta_eval_biquaternion_ridge.py \
        --csv data.csv --price-col close --horizon 1 --window 512 \
        --q 0.6 --n-terms 24 --n-freq 8 \
        --lambda 1.0 --outdir results_biquat

Outputs:
- {outdir}/predictions.csv           (t_index, y_true, y_hat)
- {outdir}/summary_biquat_ridge.csv  (one-line metrics)
"""

import argparse
import os
import math
import numpy as np
import pandas as pd


# ------------------------------
# Biquaternion utilities
# ------------------------------

def biquat_from_components(a, b, c, d):
    """
    Construct a biquaternion as 4 complex components aligned with (1, i, j, k).
    Returns np.array shape (4,) dtype=complex.
    """
    return np.array([a, b, c, d], dtype=np.complex128)


def biquat_to_real_features(q4: np.ndarray) -> np.ndarray:
    """
    Map a biquaternion (4 complex numbers) to an 8D real vector [Re(a), Im(a), Re(b), Im(b), Re(c), Im(c), Re(d), Im(d)].
    """
    assert q4.shape == (4,), "biquat must be length-4 complex vector"
    feats = np.empty(8, dtype=np.float64)
    k = 0
    for comp in q4:
        feats[k] = comp.real
        feats[k+1] = comp.imag
        k += 2
    return feats


# ------------------------------
# Jacobi theta (q-series) approximations
# ------------------------------

def theta1_series(z: complex, q: float, n_terms: int = 16) -> complex:
    # θ1(z,q) ≈ 2 * sum_{n=0..N} (-1)^n * q^{(n+1/2)^2} * sin((2n+1) z)
    s = 0.0 + 0.0j
    for n in range(n_terms):
        w = ((-1.0)**n) * (q ** ((n + 0.5) ** 2))
        s += w * np.sin((2*n + 1) * z)
    return 2.0 * s


def theta2_series(z: complex, q: float, n_terms: int = 16) -> complex:
    # θ2(z,q) ≈ 2 * sum_{n=0..N} q^{(n+1/2)^2} * cos((2n+1) z)
    s = 0.0 + 0.0j
    for n in range(n_terms):
        w = (q ** ((n + 0.5) ** 2))
        s += w * np.cos((2*n + 1) * z)
    return 2.0 * s


def theta3_series(z: complex, q: float, n_terms: int = 16) -> complex:
    # θ3(z,q) ≈ 1 + 2 * sum_{n=1..N} q^{n^2} * cos(2 n z)
    s = 1.0 + 0.0j
    for n in range(1, n_terms + 1):
        w = q ** (n ** 2)
        s += 2.0 * w * np.cos(2 * n * z)
    return s


def theta4_series(z: complex, q: float, n_terms: int = 16) -> complex:
    # θ4(z,q) ≈ 1 + 2 * sum_{n=1..N} (-1)^n * q^{n^2} * cos(2 n z)
    s = 1.0 + 0.0j
    for n in range(1, n_terms + 1):
        w = ((-1.0)**n) * (q ** (n ** 2))
        s += 2.0 * w * np.cos(2 * n * z)
    return s


# ------------------------------
# Basis construction
# ------------------------------

def analytic_pair(cos_val: complex, sin_val: complex) -> complex:
    """
    Build an analytic signal-like complex value from real cos and sin "channels".
    """
    return cos_val + 1j * sin_val


def biquaternion_theta_basis(t: int,
                             freqs: np.ndarray,
                             q: float,
                             n_terms: int,
                             phase_scale: float = 1.0) -> np.ndarray:
    """
    Construct a biquaternion feature (4 complex components) at time index t
    by combining theta series across multiple frequencies.
    We create 4 components (a,b,c,d) as mixtures of θ1..θ4 analytic pairs.

    Parameters
    ----------
    t : int
        Time index (integer step).
    freqs : np.ndarray
        Angular frequencies ω_k.
    q : float
        Base for q-series (0<q<1).
    n_terms : int
        Truncation of series.
    phase_scale : float
        Multiplier on t to produce z=ω_k*(phase_scale*t).

    Returns
    -------
    np.ndarray shape (4,), complex
        Biquaternion components.
    """
    a = 0.0 + 0.0j
    b = 0.0 + 0.0j
    c = 0.0 + 0.0j
    d = 0.0 + 0.0j

    # Combine across frequencies to give richer structure
    for omega in freqs:
        z = (phase_scale * t) * omega

        # Construct "analytic" components for each theta using matching sin/cos series
        th1 = analytic_pair(theta1_series(z, q, n_terms), theta2_series(z, q, n_terms))  # pairing θ1 with θ2 as imag companion
        th3 = analytic_pair(theta3_series(z, q, n_terms), theta4_series(z, q, n_terms))  # pairing θ3 with θ4

        # Mix into 4 components (some linear mixing for diversity)
        a += th3
        b += th1
        c += (th3 - th1) * 0.5
        d += (th3 + th1) * 0.5

    # Normalize by number of freqs to keep scales reasonable
    K = max(1, len(freqs))
    return biquat_from_components(a / K, b / K, c / K, d / K)


def build_feature_matrix_biquat(T: int,
                                freqs: np.ndarray,
                                q: float,
                                n_terms: int,
                                phase_scale: float = 1.0) -> np.ndarray:
    """
    Build feature matrix X of shape (T, D), where each row is the flattened
    real feature vector derived from the biquaternion basis at time t.

    D = 8 (Re/Im for 4 components)
    """
    D = 8
    X = np.zeros((T, D), dtype=np.float64)
    for t in range(T):
        biq = biquaternion_theta_basis(t, freqs, q, n_terms, phase_scale=phase_scale)
        X[t, :] = biquat_to_real_features(biq)
    return X


# ------------------------------
# Ridge regression helpers
# ------------------------------

def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Closed-form ridge: beta = (X^T X + lam*I)^{-1} X^T y
    """
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


# ------------------------------
# Metrics
# ------------------------------

def hit_rate(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    s_true = np.sign(y_true)
    s_hat = np.sign(y_hat)
    mask = (s_true != 0)
    if mask.sum() == 0:
        return np.nan
    return (s_true[mask] == s_hat[mask]).mean()


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
             seed: int = 42):

    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if price_col not in df.columns:
        raise ValueError(f"price_col '{price_col}' not found in CSV columns: {list(df.columns)}")

    # Prepare price and target delta
    p = df[price_col].astype(float).values
    T = len(p)
    if T <= window + horizon + 2:
        raise ValueError("Not enough samples for the chosen window and horizon.")

    y = p[horizon:] - p[:-horizon]  # length T - horizon
    T_y = len(y)

    # Frequencies: use a small set of incommensurate angular freqs
    rng = np.random.default_rng(seed)
    base = np.linspace(0.2, 1.8, n_freq)  # in radians per step (heuristic)
    jitter = rng.uniform(-0.05, 0.05, size=n_freq)
    freqs = base + jitter

    # Build features for all t (full length)
    X_full = build_feature_matrix_biquat(T, freqs=freqs, q=q, n_terms=n_terms, phase_scale=phase_scale)

    # Align X with y (prediction at t uses features at t)
    X = X_full[:-horizon, :]  # length T - horizon

    # Walk-forward
    y_hats = []
    y_trues = []
    t_indices = []

    for t0 in range(window, T_y):
        lo = t0 - window
        hi = t0  # exclusive

        X_fit = X[lo:hi, :]
        y_fit = y[lo:hi]

        x_now = X[t0, :]
        y_true = y[t0]

        # standardize on the window
        mu, sigma = standardize_fit(X_fit)
        X_fit_std = standardize_apply(X_fit, mu, sigma)
        x_now_std = standardize_apply(x_now.reshape(1, -1), mu, sigma)[0]

        # fit ridge and predict
        beta = fit_ridge_closed_form(X_fit_std, y_fit, lam=lam)
        y_hat = float(x_now_std @ beta)

        y_hats.append(y_hat)
        y_trues.append(y_true)
        t_indices.append(t0)

    y_hats = np.array(y_hats, dtype=float)
    y_trues = np.array(y_trues, dtype=float)
    t_indices = np.array(t_indices, dtype=int)

    # Metrics
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

    # Save predictions
    pred_path = os.path.join(outdir, "predictions.csv")
    pd.DataFrame({
        "t_index": t_indices,
        "y_true": y_trues,
        "y_hat": y_hats,
    }).to_csv(pred_path, index=False)

    # Save summary
    summ_path = os.path.join(outdir, "summary_biquat_ridge.csv")
    pd.DataFrame([metrics]).to_csv(summ_path, index=False)

    return pred_path, summ_path, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with a price column.")
    ap.add_argument("--price-col", default="close", help="Name of price column in CSV (default: close).")
    ap.add_argument("--horizon", type=int, default=1, help="Prediction horizon in steps (default: 1).")
    ap.add_argument("--window", type=int, default=512, help="Training window length (default: 512).")
    ap.add_argument("--q", type=float, default=0.6, help="q parameter for theta series (0<q<1, default: 0.6).")
    ap.add_argument("--n-terms", type=int, default=24, help="Number of q-series terms (default: 24).")
    ap.add_argument("--n-freq", type=int, default=8, help="Number of angular frequencies (default: 8).")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0, help="Ridge regularization lambda (default: 1.0).")
    ap.add_argument("--phase-scale", type=float, default=1.0, help="Scale applied to t in z=omega*(scale*t).")
    ap.add_argument("--outdir", default="results_biquat", help="Output directory (default: results_biquat).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for frequency jitter.")
    args = ap.parse_args()

    pred_path, summ_path, metrics = evaluate(
        csv_path=args.csv,
        price_col=args.price_col,
        horizon=args.horizon,
        window=args.window,
        q=args.q,
        n_terms=args.n_terms,
        n_freq=args.n_freq,
        lam=args.lam,
        outdir=args.outdir,
        phase_scale=args.phase_scale,
        seed=args.seed
    )

    print("Saved predictions to:", pred_path)
    print("Saved summary to:", summ_path)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
