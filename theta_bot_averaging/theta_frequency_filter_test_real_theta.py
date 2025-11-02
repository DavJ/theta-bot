#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_frequency_filter_test_real_theta.py
-----------------------------------------
Empirical resonance test on BTCUSDT using explicit Jacobi theta series expansion.

Author: David Jaroš (Theta Lab)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# === CONFIG ===
CSV_PATH = "../prices/BTCUSDT_1h.csv"
CSV_CLOSE_COL = "close"
OUT_CSV = "results_theta_frequency_filter_real_theta.csv"
WINDOW = 200
Q_LIST = [0.1, 0.3, 0.5, 0.7, 0.9]
HARMONIC_LIST = [2, 4, 8, 16, 32, 64]

# === Custom Jacobi theta approximations ===
def theta2(z, q, n_terms=40):
    n = np.arange(0, n_terms)
    return 2 * np.sum(q ** ((n + 0.5) ** 2) * np.cos((2 * n + 1) * z[:, None]), axis=1)

def theta3(z, q, n_terms=40):
    n = np.arange(1, n_terms)
    return 1 + 2 * np.sum(q ** (n ** 2) * np.cos(2 * n * z[:, None]), axis=1)

def theta4(z, q, n_terms=40):
    n = np.arange(1, n_terms)
    return 1 + 2 * np.sum((-1) ** n * q ** (n ** 2) * np.cos(2 * n * z[:, None]), axis=1)

def theta_basis(x, n_harmonics=8, q=0.3):
    """
    Construct real-valued theta basis using θ2, θ3, θ4.
    Each harmonic uses scaled argument n * π * x.
    """
    X = []
    for n in range(1, n_harmonics + 1):
        z = np.pi * n * x
        X.append(theta2(z, q))
        X.append(theta3(z, q))
        X.append(theta4(z, q))
    return np.vstack(X).T

# === Load and normalize prices ===
print(f"📈 Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
if CSV_CLOSE_COL not in df.columns:
    raise ValueError(f"Column '{CSV_CLOSE_COL}' not found in CSV!")
y = np.array(df[CSV_CLOSE_COL].values, dtype=float)
y = (y - np.mean(y)) / np.std(y)
if len(y) > 10000:
    y = y[-10000:]
x = np.linspace(0, 1, len(y))

# === Scan over q and harmonics ===
print("🧠 Scanning theta resonance (q × harmonics)...")
results = []
for q in tqdm(Q_LIST):
    for n_harm in HARMONIC_LIST:
        X = theta_basis(x, n_harmonics=n_harm, q=q)
        beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        y_pred = X @ beta
        r = np.corrcoef(y, y_pred)[0, 1]
        results.append((q, n_harm, r))

# === Save results ===
res_df = pd.DataFrame(results, columns=["q", "n_harmonics", "correlation_r"])
res_df.to_csv(OUT_CSV, index=False)
print(f"✅ Results saved to {OUT_CSV}")

# === Plot 3D heatmap ===
import matplotlib.ticker as mticker
import matplotlib.cm as cm

pivot = res_df.pivot(index="q", columns="n_harmonics", values="correlation_r")
plt.figure(figsize=(7,5))
plt.imshow(pivot, aspect="auto", origin="lower", cmap=cm.inferno,
           extent=[min(HARMONIC_LIST), max(HARMONIC_LIST),
                   min(Q_LIST), max(Q_LIST)])
plt.colorbar(label="Correlation r")
plt.xscale("log", base=2)
plt.xlabel("Number of harmonics (log₂ scale)")
plt.ylabel("q parameter")
plt.title("Theta resonance map on BTCUSDT")
plt.tight_layout()
plt.savefig("theta_resonance_map.png", dpi=200)
plt.show()

print("Done.")

