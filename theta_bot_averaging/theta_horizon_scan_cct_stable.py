#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_horizon_scan_cct_stable.py
Numerically stabilized version of the CCT horizon experiment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import argparse, os

# -------------------------------------------------------------------
def theta_basis_complex(t, q=0.5, psi=0.0, n_terms=16):
    """Complex-time theta basis with controlled q, psi."""
    n = np.arange(-n_terms, n_terms + 1)
    # Clamp q, psi to stable ranges
    q = np.clip(q, 0.1, 0.9)
    psi = np.clip(psi, -0.5, 0.5)
    tau = q + 1j * psi
    val = np.sum(np.exp(1j * np.pi * n ** 2 * tau) * np.exp(2j * np.pi * n * t))
    return val / (2 * n_terms + 1)

# -------------------------------------------------------------------
def generate_theta_matrix_cct(prices, window=512, q_base=0.5, n_terms=8):
    """Adaptive theta matrix with normalization."""
    t = np.linspace(0, 1, window)
    grad = np.gradient(prices[:window])
    drift = np.cumsum(grad) / (np.max(np.abs(np.cumsum(grad))) + 1e-8)
    psi_drift = 0.15 * drift
    sigma = np.std(grad)
    q_adaptive = q_base * (1 + 0.3 * np.sin(2 * np.pi * t)) + 0.05 * sigma

    basis = []
    for k in range(n_terms):
        phi = np.real([theta_basis_complex(ti, q_adaptive[int(i)], psi_drift[int(i)], n_terms=8) for i, ti in enumerate(t)])
        psi = np.imag([theta_basis_complex(ti, q_adaptive[int(i)], psi_drift[int(i)], n_terms=8) for i, ti in enumerate(t)])
        phi /= np.linalg.norm(phi) + 1e-8
        psi /= np.linalg.norm(psi) + 1e-8
        basis.append(phi)
        basis.append(psi)
    return np.array(basis).T

# -------------------------------------------------------------------
def predict_theta_cct(prices, window=512, horizon=16, q_base=0.5):
    X = generate_theta_matrix_cct(prices, window=window, q_base=q_base)
    XtX = X.T @ X
    # Ridge regularization for numerical stability
    XtX += np.eye(XtX.shape[0]) * 1e-6
    XTX_inv = np.linalg.pinv(XtX)
    preds, reals = [], []

    for i in range(window, len(prices) - horizon):
        y = prices[i - window:i]
        w = XTX_inv @ X.T @ y
        y_pred = X @ w
        preds.append(y_pred[-1])
        reals.append(prices[i + horizon - 1])
    return np.array(preds), np.array(reals)

# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--csv-close-col", default="close")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--q-base", type=float, default=0.5)
    parser.add_argument("--outdir", default="results_cct_stable/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.symbols)
    prices = df[args.csv_close_col].values

    horizons = [1, 2, 4, 8, 16, 32, 64, 128]
    corrs = []

    print(f"🌀 Running stabilized CCT scan on {args.symbols} ...")
    for h in horizons:
        preds, reals = predict_theta_cct(prices, window=args.window, horizon=h, q_base=args.q_base)
        if np.any(np.isnan(preds)) or np.any(np.isnan(reals)):
            print(f"⚠️ horizon={h} → NaN encountered, skipping")
            corrs.append(np.nan)
            continue
        r, _ = pearsonr(preds, reals)
        corrs.append(r)
        print(f"  horizon={h:3d} → r = {r:.4f}")

    results = pd.DataFrame({"horizon": horizons, "correlation": corrs})
    results.to_csv(os.path.join(args.outdir, "horizon_corr_cct.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(horizons, corrs, "o-", lw=2, color="purple")
    plt.xscale("log", base=2)
    plt.xlabel("Prediction horizon h")
    plt.ylabel("Correlation r")
    plt.title("Stabilized CCT correlation vs. horizon")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.savefig(os.path.join(args.outdir, "horizon_corr_cct.png"), dpi=200)
    plt.show()

    print(f"\n✅ Done! Results saved to {args.outdir}")

# -------------------------------------------------------------------
if __name__ == "__main__":
    main()

