#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_frequency_filter_test_real.py
-----------------------------------
Empirical test of theta-basis resonance on real BTCUSDT data.
Checks how predictive correlation varies with number of harmonics.

Author: David Jaroš (Theta Lab)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# === CONFIG ===
CSV_PATH = "../prices/BTCUSDT_1h.csv"
CSV_CLOSE_COL = "close"
N_MAX_HARMONICS = 128
WINDOW = 200  # local regression window
OUT_CSV = "results_theta_frequency_filter_real.csv"

# === Helper: generate theta basis ===
def theta_basis(x, n_harmonics=8):
    """
    Returns matrix of cosine/sine theta harmonics.
    """
    X = []
    for n in range(1, n_harmonics + 1):
        X.append(np.cos(2 * np.pi * n * x))
        X.append(np.sin(2 * np.pi * n * x))
    return np.vstack(X).T

# === Load data ===
print(f"📈 Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
if CSV_CLOSE_COL not in df.columns:
    raise ValueError(f"Column '{CSV_CLOSE_COL}' not found in CSV!")
y = np.array(df[CSV_CLOSE_COL].values, dtype=float)
y = (y - np.mean(y)) / np.std(y)  # normalize

# Use only last part (optional, for speed)
if len(y) > 10000:
    y = y[-10000:]
N = len(y)
x = np.linspace(0, 1, N)

# === Core test ===
results = []
print("🎚 Scanning harmonic count...")
for n_harm in tqdm([2, 4, 8, 16, 32, 64, 128]):
    X = theta_basis(x, n_harmonics=n_harm)
    # Simple linear regression (least squares)
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    y_pred = X @ beta
    r = np.corrcoef(y, y_pred)[0, 1]
    results.append((n_harm, r))

# === Save & plot ===
res_df = pd.DataFrame(results, columns=["n_harmonics", "correlation_r"])
res_df.to_csv(OUT_CSV, index=False)
print(f"✅ Saved results to {OUT_CSV}")

plt.figure(figsize=(8,5))
plt.plot(res_df["n_harmonics"], res_df["correlation_r"], "o-", lw=2)
plt.xscale("log", base=2)
plt.xlabel("Number of harmonics (log2 scale)")
plt.ylabel("Correlation r")
plt.title("Theta-basis resonance on real BTCUSDT data")
plt.grid(True)
plt.tight_layout()
plt.savefig("theta_frequency_filter_real.png", dpi=200)
plt.show()

print("Done.")

