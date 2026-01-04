from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd


def frac(x: np.ndarray | float) -> np.ndarray:
    """Fractional part in [0,1) for real inputs."""
    arr = np.asarray(x, dtype=float)
    return arr - np.floor(arr)


def log_phase(x: np.ndarray | float, base: float = 10.0, eps: float = 1e-12) -> np.ndarray:
    """Phase on [0,1) using fractional part of log-base returns."""
    arr = np.asarray(x, dtype=float)
    arr = np.maximum(arr, eps)
    return frac(np.log(arr) / math.log(base))


def circ_dist(a: float, b: float) -> float:
    """Circular distance on S1."""
    d = abs(a - b)
    return min(d, 1.0 - d)


# Alias for readability in some call sites
circular_distance = circ_dist


def phase_embedding(phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map phase to unit circle coordinates (cos, sin)."""
    phi_arr = np.asarray(phi, dtype=float)
    angles = 2 * np.pi * phi_arr
    return np.cos(angles), np.sin(angles)


def rolling_phase_concentration(phi: np.ndarray, window: int = 256) -> np.ndarray:
    """Rolling mean resultant length |E[e^{i*2*pi*phi}]|."""
    phi_series = pd.Series(phi, dtype=float)
    angles = 2 * np.pi * phi_series
    cos_part = np.cos(angles)
    sin_part = np.sin(angles)
    mean_cos = cos_part.rolling(window=window, min_periods=window).mean()
    mean_sin = sin_part.rolling(window=window, min_periods=window).mean()
    return np.sqrt(mean_cos**2 + mean_sin**2).to_numpy()


# Convenience alias
rolling_concentration = rolling_phase_concentration


def rolling_internal_concentration(
    cos_phi: np.ndarray,
    sin_phi: np.ndarray,
    cos_psi: np.ndarray,
    sin_psi: np.ndarray,
    window: int = 256,
) -> np.ndarray:
    """
    Rolling mean resultant length for (phi, psi) on a torus embedded in R^4.

    C_int = ||rolling_mean([cos φ, sin φ, cos ψ, sin ψ])|| / 2
    which stays in [0, 1]. The 1/2 normalization follows the ψ integration
    spec.
    """
    n = len(cos_phi)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        has_nan = (
            np.isnan(cos_phi[lo : i + 1]).any()
            or np.isnan(sin_phi[lo : i + 1]).any()
            or np.isnan(cos_psi[lo : i + 1]).any()
            or np.isnan(sin_psi[lo : i + 1]).any()
        )
        if has_nan:
            continue
        m1 = np.mean(cos_phi[lo : i + 1])
        m2 = np.mean(sin_phi[lo : i + 1])
        m3 = np.mean(cos_psi[lo : i + 1])
        m4 = np.mean(sin_psi[lo : i + 1])
        out[i] = float(np.sqrt(m1 * m1 + m2 * m2 + m3 * m3 + m4 * m4) / 2.0)
    return out


def max_drawdown(equity: np.ndarray) -> float:
    equity_arr = np.asarray(equity, dtype=float)
    if equity_arr.size == 0:
        return math.nan
    running_max = np.maximum.accumulate(equity_arr)
    dd = equity_arr / running_max - 1.0
    return float(dd.min())
