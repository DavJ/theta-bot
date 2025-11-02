#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theta Transform – Complex Basis Orthonormalization (Eigenvalue Diagnostics)
Autor: Ing. David Jaroš
Verze: 1.2
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, pi

# -------------------------------
# Parametry
# -------------------------------
N = 200
q = np.linspace(-1, 1, N)
t = np.linspace(-1, 1, N)
Q, T = np.meshgrid(q, t)

ψ = 0.2
φ_r = 0.0
φ_i = 0.1
τ = T + 1j * ψ

# -------------------------------
# Definice Θ_k
# -------------------------------
def theta_component(k, Q, τ):
    return np.exp(1j * (2 * pi * k * Q + φ_r)) * np.exp(-pi * k**2 * τ.imag) * np.exp(1j * φ_i * k**2)

k_vals = [-2, -1, 0, 1, 2]
Theta_list = [theta_component(k, Q, τ) for k in k_vals]

# -------------------------------
# Korelační matice
# -------------------------------
def correlation_matrix(basis):
    n = len(basis)
    R = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            R[i, j] = np.vdot(basis[i], basis[j])
    return R

O_before = correlation_matrix(Theta_list)

# -------------------------------
# Gram–Schmidt
# -------------------------------
def gram_schmidt_complex(vectors):
    ortho = []
    for v in vectors:
        for u in ortho:
            v -= np.vdot(u, v) * u
        v = v / np.linalg.norm(v)
        ortho.append(v)
    return ortho

Theta_ortho = gram_schmidt_complex(Theta_list)
O_after = correlation_matrix(Theta_ortho)

# -------------------------------
# Diagnostika odchylek
# -------------------------------
dev_real = np.max(np.abs(np.real(O_after) - np.eye(len(O_after))))
dev_imag = np.max(np.abs(np.imag(O_after)))
print(f"🔍 Max |Re<O_after> - I| = {dev_real:.2e}")
print(f"🔍 Max |Im<O_after>|    = {dev_imag:.2e}")

# -------------------------------
# Eigenvalue analýza
# -------------------------------
eig_before_real = np.linalg.eigvals(np.real(O_before))
eig_after_real  = np.linalg.eigvals(np.real(O_after))
eig_before_imag = np.linalg.eigvals(np.imag(O_before))
eig_after_imag  = np.linalg.eigvals(np.imag(O_after))

print("\n🧭 Eigenvalues (Re part before):", np.round(eig_before_real, 4))
print("🧭 Eigenvalues (Re part after): ", np.round(eig_after_real, 4))
print("🌀 Eigenvalues (Im part before):", np.round(eig_before_imag, 4))
print("🌀 Eigenvalues (Im part after): ", np.round(eig_after_imag, 4))

# -------------------------------
# Vizualizace matic
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

im1 = axes[0, 0].imshow(np.real(O_before), cmap='coolwarm', vmin=-1, vmax=1)
axes[0, 0].set_title("Re⟨Θᵢ,Θⱼ⟩ před GS")
fig.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(np.imag(O_before), cmap='plasma', vmin=-0.1, vmax=0.1)
axes[0, 1].set_title("Im⟨Θᵢ,Θⱼ⟩ před GS")
fig.colorbar(im2, ax=axes[0, 1])

im3 = axes[1, 0].imshow(np.real(O_after), cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 0].set_title("Re⟨Θᵢ,Θⱼ⟩ po GS")
fig.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(np.imag(O_after), cmap='plasma', vmin=-0.1, vmax=0.1)
axes[1, 1].set_title("Im⟨Θᵢ,Θⱼ⟩ po GS")
fig.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("theta_orthonormality_eigenvalues.png", dpi=200)
plt.show()

# -------------------------------
# Vizualizace spektra
# -------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(eig_before_real, np.zeros_like(eig_before_real), c='red', label='Re před GS')
plt.scatter(eig_after_real, np.zeros_like(eig_after_real)+0.02, c='green', label='Re po GS')
plt.scatter(eig_before_imag, np.zeros_like(eig_before_imag)-0.02, c='blue', label='Im před GS')
plt.scatter(eig_after_imag, np.zeros_like(eig_after_imag)-0.04, c='purple', label='Im po GS')
plt.title("Spektrum vlastních čísel – reálná a imaginární část")
plt.xlabel("λ")
plt.legend()
plt.grid(True)
plt.savefig("theta_eigen_spectrum.png", dpi=200)
plt.show()

print("✅ Výstupy uloženy jako:")
print("   - theta_orthonormality_eigenvalues.png")
print("   - theta_eigen_spectrum.png")

