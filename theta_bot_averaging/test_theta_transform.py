import pytest

pytest.skip("Legacy script, not a pytest test (requires external CSV and mpmath)", allow_module_level=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from theta_bot_averaging.theta_transform import theta_basis, orthonormalize, theta_transform, theta_inverse

# Načti reálná data
data = pd.read_csv("eval_h_BTCUSDT_1H.csv")
x = data['last_price'].values.astype(float)  # ✅ opraveno podle skutečného názvu sloupce
t = np.linspace(0, 1, len(x))

# Mřížky parametrů (2D spektrum)
omega_grid = np.linspace(0, np.pi, 16)
phi_grid = np.linspace(0, 2*np.pi, 16)

# Báze Jacobiho theta funkcí
B = theta_basis(t, omega_grid, phi_grid)

# Ortonormalizace báze (stabilní pomocí SVD)
U, S, Vt = orthonormalize(B)
B_orth = U

# Transformace a rekonstrukce
c = theta_transform(x, B_orth)
x_rec = theta_inverse(c, B_orth)

# Vyhodnocení přesnosti
corr = np.corrcoef(x, x_rec)[0, 1]
rms = np.sqrt(np.mean((x - x_rec)**2))
print(f"Korelace rekonstrukce: {corr:.6f}")
print(f"Chyba RMS: {rms:.6e}")

# 2D mapa amplitud
n_omega = len(omega_grid)
n_phi = len(phi_grid)
Cmap = np.abs(c[:n_omega * n_phi]).reshape((n_omega, n_phi))

plt.imshow(Cmap, extent=[phi_grid[0], phi_grid[-1], omega_grid[0], omega_grid[-1]],
           origin='lower', aspect='auto', cmap='plasma')
plt.colorbar(label='|c(ω, φ)|')
plt.xlabel('φ')
plt.ylabel('ω')
plt.title('Theta Resonance Map')
plt.tight_layout()
plt.savefig("theta_resonance_map.png", dpi=200)
plt.show()

