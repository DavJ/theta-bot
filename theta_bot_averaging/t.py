import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Load BTCUSDT price data ===
data = pd.read_csv("eval_h_BTCUSDT_1H.csv")
price = data["close"].values.astype(float)
price = (price - np.mean(price)) / np.std(price)  # normalize

# === Split into train and test ===
split = int(0.8 * len(price))
train = price[:split]
test = price[split:]
t_train = np.arange(len(train))
t_test = np.arange(len(test)) + len(train)

# === Define theta basis approximation ===
def theta_basis_matrix(N, q, n_terms=32):
    """
    Build design matrix for truncated theta_3-like series:
    f(t) = Σ q^(n^2) * cos(2π n t / N)
    """
    t = np.linspace(0, 2 * np.pi, N)
    X = np.column_stack([q ** (n * n) * np.cos(2 * n * t) for n in range(1, n_terms + 1)])
    return X

# === Fit theta model ===
def fit_theta_series(y, q, n_terms=32):
    X = theta_basis_matrix(len(y), q, n_terms)
    coeffs, _, _, _ = np.linalg.lstsq(X, y - np.mean(y), rcond=None)
    return coeffs

def predict_theta_series(q, coeffs, N, n_terms=32):
    X = theta_basis_matrix(N, q, n_terms)
    return X @ coeffs

# === Scan over q parameters and measure correlation ===
qs = np.linspace(0.1, 0.9, 9)
taus = np.arange(-30, 31)  # time shifts
corr_map = np.zeros((len(qs), len(taus)))

for i, q in enumerate(qs):
    coeffs = fit_theta_series(train, q)
    pred = predict_theta_series(q, coeffs, len(test))

    for j, tau in enumerate(taus):
        if tau >= 0:
            r = np.corrcoef(pred[:-tau or None], test[tau:])[0, 1]
        else:
            r = np.corrcoef(pred[-tau:], test[:tau or None])[0, 1]
        corr_map[i, j] = r

# === Plot correlation map ===
plt.figure(figsize=(8, 5))
plt.imshow(corr_map, extent=[taus[0], taus[-1], qs[0], qs[-1]],
           origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label="Correlation r")
plt.xlabel("Time shift τ (hours)")
plt.ylabel("q parameter")
plt.title("Theta temporal resonance map on BTCUSDT")
plt.tight_layout()
plt.show()

# === Plot example prediction ===
best_q_idx, best_tau_idx = np.unravel_index(np.nanargmax(corr_map), corr_map.shape)
best_q = qs[best_q_idx]
best_tau = taus[best_tau_idx]

coeffs = fit_theta_series(train, best_q)
pred = predict_theta_series(best_q, coeffs, len(test))

plt.figure(figsize=(10, 4))
plt.plot(t_test, test, label="Real BTCUSDT")
plt.plot(t_test, pred, label=f"Theta prediction q={best_q:.2f}")
plt.legend()
plt.title(f"Prediction vs Real Data (best τ={best_tau}, corr={np.nanmax(corr_map):.3f})")
plt.xlabel("Time index")
plt.ylabel("Normalized price")
plt.tight_layout()
plt.show()

