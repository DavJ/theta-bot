from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

EPS_LOG = 1e-12


def complex_cepstrum(signal: np.ndarray, eps: float = EPS_LOG) -> np.ndarray:
    """
    Complex cepstrum of a real-valued signal.
    """
    x = np.asarray(signal, dtype=float)
    spectrum = np.fft.fft(x)
    mag = np.abs(spectrum)
    ang = np.unwrap(np.angle(spectrum))
    log_spectrum = np.log(mag + eps) + 1j * ang
    return np.fft.ifft(log_spectrum)


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
) -> float:
    """
    Return psi in [0,1) from the dominant cepstral bin within band.
    Deterministic and safe for constant arrays.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return math.nan
    if np.isnan(arr).any():
        return math.nan
    domain = (domain or "linear").lower()
    min_bin = max(1, int(min_bin))
    max_frac = float(max_frac)

    seg = arr
    if domain == "logtime":
        seg = _logtime_resample(seg)

    spectrum = np.fft.fft(seg)
    log_mag = np.log(np.abs(spectrum) + eps)
    cepstrum = np.fft.ifft(log_mag)

    candidate_max = min(int(len(seg) * max_frac), len(seg) // 2, len(cepstrum))
    max_bin = max(candidate_max, min_bin + 1)
    max_bin = min(max_bin, len(cepstrum))
    if min_bin >= max_bin:
        return math.nan

    candidate_slice = cepstrum[min_bin:max_bin]
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
    phi = (ang / (2 * np.pi)) % 1.0
    if phi >= 1.0 - np.finfo(float).eps:
        phi = math.nextafter(1.0, 0.0)
    return phi


def complex_cepstral_phase(
    x: np.ndarray,
    domain: Literal["linear", "logtime"] = "linear",
    min_bin: int = 2,
    max_frac: float = 0.25,
    eps: float = EPS_LOG,
    return_debug: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """
    Return psi in [0,1) from complex cepstrum with optional debug outputs.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or np.isnan(arr).any():
        if return_debug:
            return math.nan, {}
        return math.nan

    domain = (domain or "linear").lower()
    min_bin = max(1, int(min_bin))
    max_frac = float(max_frac)

    seg = arr
    if domain == "logtime":
        seg = _logtime_resample(seg)

    cepstrum = complex_cepstrum(seg, eps=eps)

    candidate_max = min(int(len(seg) * max_frac), len(seg) // 2, len(cepstrum))
    max_bin = max(candidate_max, min_bin + 1)
    max_bin = min(max_bin, len(cepstrum))
    if min_bin >= max_bin:
        if return_debug:
            return math.nan, {}
        return math.nan

    band = cepstrum[min_bin:max_bin]
    if band.size == 0:
        if return_debug:
            return math.nan, {}
        return math.nan

    local_idx = int(np.argmax(np.abs(band)))
    n_star = min_bin + local_idx
    c_star = cepstrum[n_star]
    psi_angle = float(np.angle(c_star))
    psi = (psi_angle / (2 * np.pi)) % 1.0
    if psi >= 1.0 - np.finfo(float).eps:
        psi = math.nextafter(1.0, 0.0)
    psi = min(max(psi, 0.0), 1.0 - 1e-12)

    if not return_debug:
        return psi

    debug = {
        "psi_n_star": float(n_star),
        "psi_c_real": float(np.real(c_star)),
        "psi_c_imag": float(np.imag(c_star)),
        "psi_c_abs": float(np.abs(c_star)),
        "psi_angle_rad": psi_angle,
        "psi": float(psi),
    }
    return psi, debug


def rolling_cepstral_phase(
    series: pd.Series,
    window: int,
    min_bin: int = 2,
    max_frac: float = 0.25,
    topk: int | None = None,
    domain: Literal["linear", "logtime"] = "linear",
    eps: float = EPS_LOG,
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
            eps=eps,
        )
    return pd.Series(out, index=series.index)


def rolling_complex_cepstral_phase(
    series: pd.Series,
    window: int,
    min_bin: int = 2,
    max_frac: float = 0.25,
    domain: Literal["linear", "logtime"] = "linear",
    eps: float = EPS_LOG,
    return_debug: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """
    Rolling complex cepstral phase over a sliding window with optional debug outputs.
    """
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    debug_data = None
    if return_debug:
        debug_data = {
            "psi_n_star": np.full(n, np.nan, dtype=float),
            "psi_c_real": np.full(n, np.nan, dtype=float),
            "psi_c_imag": np.full(n, np.nan, dtype=float),
            "psi_c_abs": np.full(n, np.nan, dtype=float),
            "psi_angle_rad": np.full(n, np.nan, dtype=float),
        }

    for i in range(window - 1, n):
        window_arr = arr[i - window + 1 : i + 1]
        if np.isnan(window_arr).any():
            continue
        if return_debug:
            psi_val, dbg = complex_cepstral_phase(
                window_arr,
                domain=domain,
                min_bin=min_bin,
                max_frac=max_frac,
                eps=eps,
                return_debug=True,
            )
            out[i] = psi_val
            if debug_data is not None and dbg is not None and len(dbg) > 0:
                for key in debug_data:
                    debug_data[key][i] = dbg.get(key, math.nan)
        else:
            out[i] = complex_cepstral_phase(
                window_arr,
                domain=domain,
                min_bin=min_bin,
                max_frac=max_frac,
                eps=eps,
                return_debug=False,
            )

    series_out = pd.Series(out, index=series.index)
    if not return_debug:
        return series_out
    debug_df = pd.DataFrame(debug_data, index=series.index)
    return series_out, debug_df
