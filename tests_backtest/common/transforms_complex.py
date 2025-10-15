import numpy as np
import logging

logger = logging.getLogger(__name__)

def fft_complex(series, top_n=16):
    s = np.asarray(series, dtype=float)
    s = (s - s.mean()) / (s.std() + 1e-12)
    Z = np.fft.rfft(s)  # complex spectrum (length T//2+1)
    # keep top-N by magnitude (exclude DC if needed)
    mag = np.abs(Z)
    idx = np.argsort(mag)[::-1][:top_n]
    Z_top = Z[idx]
    # sort bins by frequency index to keep temporal consistency
    idx_sorted = np.sort(idx)
    Z_sel = Z[idx_sorted]
    # normalize coefficients
    scale = np.max(np.abs(Z_sel)) + 1e-12
    Z_norm = (Z_sel / scale).astype(np.complex128)

    resid_energy = float(np.sum((mag**2)) - np.sum(np.abs(Z_sel)**2))
    resid_stats = np.array([0.0, 0.0, resid_energy], dtype=float)

    if logger and logger.isEnabledFor(logging.INFO):
        mean_amp = float(np.mean(np.abs(Z_norm)))
        nz = float((np.abs(Z_norm) > 1e-12).mean())
        logger.info(f"[fft_complex] top_n={top_n} mean|Z|={mean_amp:.6f} nonzero={nz:.3f}")
    return Z_norm, resid_stats
