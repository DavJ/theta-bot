import numpy as np
from mpmath import jtheta

def theta_basis(t, omega_grid, phi_grid, q=0.95):
    """Vytvoří 2D matici theta funkcí Θ(t; ω, φ)."""
    basis = []
    for ω in omega_grid:
        for φ in phi_grid:
            z = ω * t + φ
            vals = np.array([float(jtheta(3, zi, q)) for zi in z])
            basis.append(vals)
    return np.array(basis).T  # t × (ω×φ)

def orthonormalize(B):
    """Ortonormalizuje bázi pomocí SVD (stabilnější než Gram-Schmidt)."""
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    return U, S, Vt

def theta_transform(x, B_orth):
    """Přímá transformace — projekce na ortonormální bázi."""
    return B_orth.T @ x

def theta_inverse(coeffs, B_orth):
    """Rekonstrukce signálu ze spektrálních koeficientů."""
    return B_orth @ coeffs

