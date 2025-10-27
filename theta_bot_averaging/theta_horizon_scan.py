#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_horizon_scan.py
Author: David Jaroš & GPT-5 (2025)
Description:
  This script measures the correlation (Pearson r) between predicted and actual
  values as a function of the prediction horizon h.

  It uses a pure theta basis projection without Ridge regularization
  (CCT/UBT style analysis).

Outputs:
  - CSV table: horizon_corr.csv
  - Plot: horizon_corr.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.special import ellipk
import argparse
import os

# --- Theta basis approximation (real-valued Jacobi theta-like form) ---
def theta_basis(t, q=0.5, n_terms=32):
    """Approximate real part of the theta function."""
    n = np.arange(-n_terms, n_terms + 1)
    return np.sum(np.exp(1j * np.pi * n ** 2 * q) * np.exp(2j * np.pi * n * t))

# --- Build theta projection matrix ---
def generate_theta_matrix(prices, window=512, q=0.5, n_terms=16):
    """
    Generate a projection matrix using real and imaginary components
    of the theta basis for a given window size.
    """
    t = np.linspace(0, 1, window)
    basis = []
    for k in range(n_terms):
        phi = np.real([theta_basis(ti, q ** (k + 1)) for ti in t])
        psi = np.imag([theta_basis(ti, q ** (k + 1)) for ti in t])
        basis.append(phi)
        basis.append(psi)
    return np.array(basis).T  # Shape: (window, 2 * n_terms)

# --- Theta-based prediction model (simple projection) ---
def predict_theta(prices, window=512, horizon=16, q=0.5):
    """
    Predicts future prices using theta projection.
    No learning – just least squares projection within the theta space.
    """
    X = generate_theta_matrix(prices, window=window, q=q)
    XTX_inv = np.linalg.pinv(X.T @ X)
    preds, reals = [], []

    for i in range(window, len(prices) - horizon):
        y = prices[i - window:i]
        w = XTX_inv @ X.T @ y
        y_pred = X @ w
        preds.append(y_pred[-1])  # last projected value
        reals.append(prices[i + horizon - 1])
    return np.array(preds), np.array(reals)

# --- Main procedure ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True, help="CSV file with price data (e.g. ../prices/BTCUSDT_1h.csv)")
    parser.add_argument("--csv-time-col", default="time")
    parser.add_argument("--csv-close-col", default="close")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--outdir", default="horizon_scan/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.symbols)
    prices = df[args.csv_close_col].values
    horizons = [1, 2, 4, 8, 16, 32, 64, 128]
    corrs = []

    print(f"🧭 Running theta horizon scan on {args.symbols} ...")
    for h in horizons:
        preds, reals = predict_theta(prices, window=args.window, horizon=h, q=args.q)
        r, _ = pearsonr(preds, reals)
        corrs.append(r)
        print(f"  h={h:3d}  →  r = {r:.4f}")

    # --- Save results ---
    results = pd.DataFrame({"horizon": horizons, "correlation": corrs})
    results.to_csv(os.path.join(args.outdir, "horizon_corr.csv"), index=False)

    # --- Plot correlation vs horizon ---
    plt.figure(figsize=(8, 5))
    plt.plot(horizons, corrs, "o-", lw=2)
    plt.xscale("log", base=2)
    plt.xlabel("Prediction horizon h")
    plt.ylabel("Correlation r")
    plt.title("Theta projection correlation vs. prediction horizon")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.axhline(1.0, color="gray", ls="--", lw=1)
    plt.savefig(os.path.join(args.outdir, "horizon_corr.png"), dpi=200)
    plt.show()

    print(f"\n✅ Done! Results saved in {args.outdir}")

# --- Entry point ---
if __name__ == "__main__":
    main()

