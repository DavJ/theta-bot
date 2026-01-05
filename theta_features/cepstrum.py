from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

EPS_LOG = 1e-12


def frac01(x: float, eps: float = np.finfo(float).eps) -> float:
    """
    Wrap x into [0, 1) with a stable modulus (robust to negative inputs) and clamp values extremely
    close to 1.0 back to 0 so phase angles remain continuous on the unit circle.

    Args:
        x: Value to wrap into the unit interval.
        eps: Tolerance for mapping values extremely close to 1.0 back to 0.0.

    Returns:
        Value normalized into [0, 1).
    """
    y = ((float(x) % 1.0) + 1.0) % 1.0
    if y >= 1.0 - eps:
        return 0.0
    return y


def _logtime_resample(seg: np.ndarray) -> np.ndarray:
    w = len(seg)
    idx = np.unique(np.floor(np.exp(np.linspace(np.log(1.0), np.log(w), w))).astype(int) - 1)
    idx = np.clip(idx, 0, w - 1)
    warped = seg[idx]
    if len(warped) == 0:
        return np.zeros(w, dtype=float)
    if len(warped) == 1:
        return np.full(w, warped[0], dtype=float)
    if len(warped) < w:
        x_src = np.linspace(0.0, 1.0, num=len(warped))
        x_tgt = np.linspace(0.0, 1.0, num=w)
        warped = np.interp(x_tgt, x_src, warped)
    return warped


def cepstral_phase(
    x: np.ndarray,
    domain: Literal["linear", "logtime"] = "linear",
    min_bin: int = 2,
    max_frac: float = 0.25,
    topk: int | None = None,
    eps: float = EPS_LOG,
    phase_source: Literal["spectrum", "cepstrum"] = "spectrum",
) -> float:
    """
    Return psi in [0,1) from the dominant cepstral bin within band.
    Deterministic and safe for constant arrays.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return math.nan
    domain = (domain or "linear").lower()
    phase_source_normalized = phase_source.lower()
    if phase_source_normalized not in {"spectrum", "cepstrum"}:
        raise ValueError("phase_source must be 'spectrum' or 'cepstrum'.")
    phase_source = phase_source_normalized
    min_bin = max(1, int(min_bin))
    max_frac = float(max_frac)

    seg = arr
    if domain == "logtime":
        seg = _logtime_resample(seg)

    spectrum = np.fft.fft(seg)
    if phase_source == "cepstrum":
        log_mag = np.log(np.abs(spectrum) + eps)
        cepstrum = np.fft.ifft(log_mag)

        candidate_max = min(int(len(seg) * max_frac), len(seg) // 2, len(cepstrum))
        max_bin = max(candidate_max, min_bin + 1)
        max_bin = min(max_bin, len(cepstrum))
        if min_bin >= max_bin:
            return math.nan
        candidate_slice = cepstrum[min_bin:max_bin]
    else:
        # Avoid the DC component when selecting the dominant spectral bin.
        k_min = min_bin
        candidate_max = min(int(len(seg) * max_frac), len(spectrum) // 2)
        k_max = max(candidate_max, k_min + 1)
        k_max = min(k_max, len(spectrum))
        if k_min >= k_max:
            return math.nan
        candidate_slice = spectrum[k_min:k_max]

    mags = np.abs(candidate_slice)
    if candidate_slice.size == 0:
        return math.nan
    if topk is not None and topk >= 2:
        k = min(topk, len(candidate_slice))
        idxs = np.argpartition(mags, -k)[-k:]
        angles = np.angle(candidate_slice[idxs])
        weights = mags[idxs]
        combined = np.sum(weights * np.exp(1j * angles))
        ang = float(np.angle(combined))
    else:
        best_idx = int(np.argmax(mags))
        ang = float(np.angle(candidate_slice[best_idx]))
    return frac01(ang / (2 * np.pi))


def rolling_cepstral_phase(
    series: pd.Series,
    window: int,
    min_bin: int = 2,
    max_frac: float = 0.25,
    topk: int | None = None,
    domain: Literal["linear", "logtime"] = "linear",
    eps: float = EPS_LOG,
    phase_source: Literal["spectrum", "cepstrum"] = "spectrum",
) -> pd.Series:
    """
    Rolling cepstral phase over a sliding window.
    """
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        window_arr = arr[i - window + 1 : i + 1]
        if np.isnan(window_arr).any():
            continue
        out[i] = cepstral_phase(
            window_arr,
            domain=domain,
            min_bin=min_bin,
            max_frac=max_frac,
            topk=topk,
            phase_source=phase_source,
            eps=eps,
        )
    return pd.Series(out, index=series.index)
