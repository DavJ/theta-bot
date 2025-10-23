import numpy as np
import logging

logger = logging.getLogger(__name__)

# ==========================
#   THETA TRANSFORM HELPERS
# ==========================

def _theta3_grid(T, K, tau_im):
    q = np.exp(-np.pi * tau_im)
    t = np.arange(T) / float(T)
    n = np.arange(-20, 21)
    phases = np.linspace(0.0, 1.0, K, endpoint=False)
    PhiR = []
    PhiI = []
    for phi in phases:
        arg = 2*np.pi * np.outer(t + phi, n)
        w = q**(n**2)
        PhiR.append((w * np.cos(arg)).sum(axis=1))
        PhiI.append((w * np.sin(arg)).sum(axis=1))
    PhiR = np.stack(PhiR, axis=1)
    PhiI = np.stack(PhiI, axis=1)
    Phi = (PhiR + 1j*PhiI).astype(np.complex128)
    for k in range(Phi.shape[1]):
        nk = np.sqrt(np.vdot(Phi[:, k], Phi[:, k]).real) + 1e-12
        Phi[:, k] /= nk
    return Phi


def _gram_schmidt_complex(Phi):
    Q = np.zeros_like(Phi, dtype=np.complex128)
    for i in range(Phi.shape[1]):
        v = Phi[:, i].astype(np.complex128).copy()
        for j in range(i):
            r = np.vdot(Q[:, j], v)
            v = v - r * Q[:, j]
        nrm = np.sqrt(np.vdot(v, v).real) + 1e-12
        Q[:, i] = v / nrm
    return Q


def theta_complex(series, K=16, tau_im=0.15, ridge=1e-3, gram=False):
    s = np.asarray(series, dtype=float)
    s = (s - s.mean()) / (s.std() + 1e-12)
    T = len(s)
    Phi = _theta3_grid(T, K, tau_im)
    if gram:
        Phi = _gram_schmidt_complex(Phi)

    H = np.conjugate(Phi).T @ Phi
    b = np.conjugate(Phi).T @ s
    H = H + ridge * np.eye(H.shape[0])
    a = np.linalg.solve(H, b)

    recon = (Phi @ a).real
    resid = s - recon

    scale = np.max(np.abs(a)) + 1e-12
    a_scaled = (a / scale).astype(np.complex128)

    resid_stats = np.array([resid.mean(), resid.std(), float(np.sum(resid**2))], dtype=float)
    a_scaled = np.nan_to_num(a_scaled)
    resid_stats = np.nan_to_num(resid_stats)

    # debug
    if logger and logger.isEnabledFor(logging.INFO):
        mean_amp = float(np.mean(np.abs(a_scaled)))
        nz = float((np.abs(a_scaled) > 1e-12).mean())
        logger.info(f"[theta_complex] K={K} tau={tau_im} ridge={ridge} gram={gram} mean|a|={mean_amp:.6f} nonzero={nz:.3f}")
    return a_scaled, resid_stats


def theta_reim_features(series, K=16, tau_im=0.15, ridge=1e-3, gram=False):
    a, resid = theta_complex(series, K=K, tau_im=tau_im, ridge=ridge, gram=gram)
    feats = np.concatenate([a.real, a.imag, resid])
    feats = np.nan_to_num(feats, copy=False)
    return feats


def theta_magphase_features(series, K=16, tau_im=0.15, ridge=1e-3, gram=False):
    a, resid = theta_complex(series, K=K, tau_im=tau_im, ridge=ridge, gram=gram)
    ph = np.angle(a)
    feats = np.concatenate([np.abs(a), np.cos(ph), np.sin(ph), resid])
    feats = np.nan_to_num(feats, copy=False)
    return feats
