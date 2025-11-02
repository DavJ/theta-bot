# theta_frequency_filter_test.py
# Author: David Jaros & GPT-5
# Purpose: Verify the effect of removing high-frequency theta components
# on predictive correlation in the CCT model.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# --- Parameters ---------------------------------------------------------------
N = 2048             # number of time points
n_terms_list = [2, 4, 8, 16, 32, 64, 128]   # number of theta harmonics
ridge_alpha = 1e-3
seed = 42
np.random.seed(seed)

# --- Synthetic signal (replace with real crypto data if desired) --------------
t = np.linspace(0, 4 * np.pi, N)
signal = np.sin(t) + 0.5 * np.sin(3*t + 0.5) + 0.2 * np.sin(7*t + 1.2)
signal += 0.05 * np.random.randn(N)
signal = (signal - np.mean(signal)) / np.std(signal)

# --- Helper: theta basis generator --------------------------------------------
def theta_basis(t, n_terms):
    """Generate real theta-like harmonic basis up to n_terms."""
    X = []
    for n in range(1, n_terms + 1):
        X.append(np.sin(n * t))
        X.append(np.cos(n * t))
    return np.column_stack(X)

# --- Correlation test ---------------------------------------------------------
corrs = []
for n_terms in n_terms_list:
    X = theta_basis(t, n_terms)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_scaled[:-1], signal[1:])  # predict next time step
    pred = model.predict(X_scaled[:-1])

    r = np.corrcoef(signal[1:], pred)[0, 1]
    corrs.append(r)
    print(f"Harmonics {n_terms:>3d}: correlation r = {r:.4f}")

# --- Plot ---------------------------------------------------------------------
plt.figure(figsize=(7,4))
plt.plot(n_terms_list, corrs, marker='o', color='purple', lw=2)
plt.xscale('log', base=2)
plt.xlabel("Number of harmonics n_max")
plt.ylabel("Correlation r")
plt.title("CCT frequency filtering test: correlation vs. n_max")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

