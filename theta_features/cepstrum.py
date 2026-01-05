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

    candidate_max = min(int(len(seg) * max_frac), len(seg) // 2)
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
    eps_float = np.finfo(float).eps
    if psi >= 1.0 - eps_float:
        psi = math.nextafter(1.0, 0.0)
    psi = min(max(psi, 0.0), 1.0 - eps_float)

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
            if debug_data is not None and dbg:
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


def mellin_transform(
    x: np.ndarray,
    grid_n: int = 256,
    sigma: float = 0.0,
    eps: float = EPS_LOG,
) -> np.ndarray:
    """
    Compute Mellin transform via log-domain resampling.
    
    For a window x[k], k=1..W (exclude k=0):
    - u = log(k); resample x(u) onto uniform u-grid of length grid_n
    - optionally weight by exp(sigma*u)
    - compute FFT in u-domain to obtain X_M(omega)
    
    Args:
        x: Input signal array
        grid_n: Length of uniform u-grid for resampling
        sigma: Real part of s = sigma + i*omega (exponential weighting)
        eps: Small constant to avoid log(0)
        
    Returns:
        Mellin transform X_M as complex array
    """
    arr = np.asarray(x, dtype=float)
    w = len(arr)
    if w == 0:
        return np.zeros(grid_n, dtype=complex)
    
    # Exclude k=0, use k=1..W
    k_vals = np.arange(1, w + 1, dtype=float)
    u_src = np.log(k_vals + eps)
    
    # Create uniform u-grid
    u_min = u_src[0]
    u_max = u_src[-1]
    u_grid = np.linspace(u_min, u_max, grid_n)
    
    # Resample x onto uniform u-grid
    x_resampled = np.interp(u_grid, u_src, arr)
    
    # Apply exponential weighting if sigma != 0
    if abs(sigma) > eps:
        x_resampled = x_resampled * np.exp(sigma * u_grid)
    
    # Compute FFT to get Mellin transform
    x_m = np.fft.fft(x_resampled)
    return x_m


def mellin_cepstrum(
    x: np.ndarray,
    grid_n: int = 256,
    sigma: float = 0.0,
    eps: float = EPS_LOG,
) -> np.ndarray:
    """
    Real Mellin cepstrum: C = ifft(log(|X_M| + eps))
    
    Args:
        x: Input signal array
        grid_n: Length of uniform u-grid for Mellin transform
        sigma: Real part of s for exponential weighting
        eps: Small constant for log stability
        
    Returns:
        Real Mellin cepstrum as complex array
    """
    x_m = mellin_transform(x, grid_n=grid_n, sigma=sigma, eps=eps)
    mag = np.abs(x_m)
    log_mag = np.log(mag + eps)
    cepstrum = np.fft.ifft(log_mag)
    return cepstrum


def mellin_complex_cepstrum(
    x: np.ndarray,
    grid_n: int = 256,
    sigma: float = 0.0,
    detrend_phase: bool = True,
    eps: float = EPS_LOG,
) -> np.ndarray:
    """
    Complex Mellin cepstrum with unwrapped phase and optional detrending.
    
    - phase = unwrap(angle(X_M))
    - if detrend_phase: remove linear trend from phase vs bin index
    - logX = log(|X_M| + eps) + 1j*phase
    - C = ifft(logX)
    
    Args:
        x: Input signal array
        grid_n: Length of uniform u-grid for Mellin transform
        sigma: Real part of s for exponential weighting
        detrend_phase: Whether to remove linear trend from unwrapped phase
        eps: Small constant for log stability
        
    Returns:
        Complex Mellin cepstrum as complex array
    """
    x_m = mellin_transform(x, grid_n=grid_n, sigma=sigma, eps=eps)
    mag = np.abs(x_m)
    phase = np.unwrap(np.angle(x_m))
    
    # Optional phase detrending
    if detrend_phase:
        n_bins = len(phase)
        bin_indices = np.arange(n_bins, dtype=float)
        # Remove linear trend using least-squares fit
        if n_bins > 1:
            # Compute slope and intercept
            mean_idx = np.mean(bin_indices)
            mean_phase = np.mean(phase)
            slope = np.sum((bin_indices - mean_idx) * (phase - mean_phase)) / (np.sum((bin_indices - mean_idx) ** 2) + eps)
            intercept = mean_phase - slope * mean_idx
            trend = slope * bin_indices + intercept
            phase = phase - trend
    
    log_spectrum = np.log(mag + eps) + 1j * phase
    cepstrum = np.fft.ifft(log_spectrum)
    return cepstrum


def _extract_psi_from_cepstrum(
    cepstrum: np.ndarray,
    min_bin: int = 2,
    max_frac: float = 0.25,
    phase_agg: Literal["peak", "cmean"] = "peak",
    phase_power: float = 1.0,
) -> float:
    """
    Extract psi from cepstrum using configurable aggregation.
    
    - band indices: [min_bin : floor(max_frac*len(C))]
    - phase_agg="peak": choose k* with max |C[k]| in band, psi = angle(C[k*])
    - phase_agg="cmean": circular mean of angles in band with weights |C|**phase_power
    
    Args:
        cepstrum: Complex cepstrum array
        min_bin: Minimum bin index for band
        max_frac: Maximum fraction of cepstrum length for band
        phase_agg: Aggregation method ("peak" or "cmean")
        phase_power: Power for weighting magnitudes in circular mean
        
    Returns:
        psi in [0, 1)
    """
    candidate_max = min(int(len(cepstrum) * max_frac), len(cepstrum) // 2)
    max_bin = max(candidate_max, min_bin + 1)
    max_bin = min(max_bin, len(cepstrum))
    
    if min_bin >= max_bin:
        return math.nan
    
    band = cepstrum[min_bin:max_bin]
    if band.size == 0:
        return math.nan
    
    mags = np.abs(band)
    
    if phase_agg == "cmean":
        # Circular mean with weighted magnitudes
        weights = np.power(mags + EPS_LOG, phase_power)
        angles = np.angle(band)
        combined = np.sum(weights * np.exp(1j * angles))
        ang = float(np.angle(combined))
    else:  # peak
        best_idx = int(np.argmax(mags))
        ang = float(np.angle(band[best_idx]))
    
    psi = (ang / (2 * np.pi)) % 1.0
    eps_float = np.finfo(float).eps
    if psi >= 1.0 - eps_float:
        psi = math.nextafter(1.0, 0.0)
    psi = min(max(psi, 0.0), 1.0 - eps_float)
    return psi


def mellin_cepstral_phase(
    x: np.ndarray,
    grid_n: int = 256,
    sigma: float = 0.0,
    min_bin: int = 2,
    max_frac: float = 0.25,
    phase_agg: Literal["peak", "cmean"] = "peak",
    phase_power: float = 1.0,
    eps: float = EPS_LOG,
    return_debug: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """
    Return psi in [0,1) from real Mellin cepstrum with optional debug outputs.
    
    Args:
        x: Input signal array
        grid_n: Length of uniform u-grid for Mellin transform
        sigma: Real part of s for exponential weighting
        min_bin: Minimum bin index for band
        max_frac: Maximum fraction of cepstrum length for band
        phase_agg: Aggregation method ("peak" or "cmean")
        phase_power: Power for weighting magnitudes in circular mean
        eps: Small constant for log stability
        return_debug: Whether to return debug information
        
    Returns:
        psi value or (psi, debug_dict) if return_debug=True
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or np.isnan(arr).any():
        if return_debug:
            return math.nan, {}
        return math.nan
    
    cepstrum = mellin_cepstrum(arr, grid_n=grid_n, sigma=sigma, eps=eps)
    psi = _extract_psi_from_cepstrum(
        cepstrum,
        min_bin=min_bin,
        max_frac=max_frac,
        phase_agg=phase_agg,
        phase_power=phase_power,
    )
    
    if not return_debug:
        return psi
    
    # Find the peak bin for debug info
    candidate_max = min(int(len(cepstrum) * max_frac), len(cepstrum) // 2)
    max_bin = max(candidate_max, min_bin + 1)
    max_bin = min(max_bin, len(cepstrum))
    
    if min_bin < max_bin:
        band = cepstrum[min_bin:max_bin]
        local_idx = int(np.argmax(np.abs(band)))
        n_star = min_bin + local_idx
        c_star = cepstrum[n_star]
    else:
        n_star = min_bin
        c_star = 0.0 + 0.0j
    
    debug = {
        "psi_n_star": float(n_star),
        "psi_c_real": float(np.real(c_star)),
        "psi_c_imag": float(np.imag(c_star)),
        "psi_c_abs": float(np.abs(c_star)),
        "psi_angle_rad": float(np.angle(c_star)),
        "psi": float(psi),
    }
    return psi, debug


def mellin_complex_cepstral_phase(
    x: np.ndarray,
    grid_n: int = 256,
    sigma: float = 0.0,
    detrend_phase: bool = True,
    min_bin: int = 2,
    max_frac: float = 0.25,
    phase_agg: Literal["peak", "cmean"] = "peak",
    phase_power: float = 1.0,
    eps: float = EPS_LOG,
    return_debug: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """
    Return psi in [0,1) from complex Mellin cepstrum with optional debug outputs.
    
    Args:
        x: Input signal array
        grid_n: Length of uniform u-grid for Mellin transform
        sigma: Real part of s for exponential weighting
        detrend_phase: Whether to remove linear trend from unwrapped phase
        min_bin: Minimum bin index for band
        max_frac: Maximum fraction of cepstrum length for band
        phase_agg: Aggregation method ("peak" or "cmean")
        phase_power: Power for weighting magnitudes in circular mean
        eps: Small constant for log stability
        return_debug: Whether to return debug information
        
    Returns:
        psi value or (psi, debug_dict) if return_debug=True
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or np.isnan(arr).any():
        if return_debug:
            return math.nan, {}
        return math.nan
    
    cepstrum = mellin_complex_cepstrum(
        arr, grid_n=grid_n, sigma=sigma, detrend_phase=detrend_phase, eps=eps
    )
    psi = _extract_psi_from_cepstrum(
        cepstrum,
        min_bin=min_bin,
        max_frac=max_frac,
        phase_agg=phase_agg,
        phase_power=phase_power,
    )
    
    if not return_debug:
        return psi
    
    # Find the peak bin for debug info
    candidate_max = min(int(len(cepstrum) * max_frac), len(cepstrum) // 2)
    max_bin = max(candidate_max, min_bin + 1)
    max_bin = min(max_bin, len(cepstrum))
    
    if min_bin < max_bin:
        band = cepstrum[min_bin:max_bin]
        local_idx = int(np.argmax(np.abs(band)))
        n_star = min_bin + local_idx
        c_star = cepstrum[n_star]
    else:
        n_star = min_bin
        c_star = 0.0 + 0.0j
    
    debug = {
        "psi_n_star": float(n_star),
        "psi_c_real": float(np.real(c_star)),
        "psi_c_imag": float(np.imag(c_star)),
        "psi_c_abs": float(np.abs(c_star)),
        "psi_angle_rad": float(np.angle(c_star)),
        "psi": float(psi),
    }
    return psi, debug


def rolling_mellin_cepstral_phase(
    series: pd.Series,
    window: int,
    grid_n: int = 256,
    sigma: float = 0.0,
    min_bin: int = 2,
    max_frac: float = 0.25,
    phase_agg: Literal["peak", "cmean"] = "peak",
    phase_power: float = 1.0,
    eps: float = EPS_LOG,
    return_debug: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """
    Rolling Mellin cepstral phase over a sliding window with optional debug outputs.
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
            psi_val, dbg = mellin_cepstral_phase(
                window_arr,
                grid_n=grid_n,
                sigma=sigma,
                min_bin=min_bin,
                max_frac=max_frac,
                phase_agg=phase_agg,
                phase_power=phase_power,
                eps=eps,
                return_debug=True,
            )
            out[i] = psi_val
            if debug_data is not None and dbg:
                for key in debug_data:
                    debug_data[key][i] = dbg.get(key, math.nan)
        else:
            out[i] = mellin_cepstral_phase(
                window_arr,
                grid_n=grid_n,
                sigma=sigma,
                min_bin=min_bin,
                max_frac=max_frac,
                phase_agg=phase_agg,
                phase_power=phase_power,
                eps=eps,
                return_debug=False,
            )
    
    series_out = pd.Series(out, index=series.index)
    if not return_debug:
        return series_out
    debug_df = pd.DataFrame(debug_data, index=series.index)
    return series_out, debug_df


def rolling_mellin_complex_cepstral_phase(
    series: pd.Series,
    window: int,
    grid_n: int = 256,
    sigma: float = 0.0,
    detrend_phase: bool = True,
    min_bin: int = 2,
    max_frac: float = 0.25,
    phase_agg: Literal["peak", "cmean"] = "peak",
    phase_power: float = 1.0,
    eps: float = EPS_LOG,
    return_debug: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """
    Rolling Mellin complex cepstral phase over a sliding window with optional debug outputs.
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
            psi_val, dbg = mellin_complex_cepstral_phase(
                window_arr,
                grid_n=grid_n,
                sigma=sigma,
                detrend_phase=detrend_phase,
                min_bin=min_bin,
                max_frac=max_frac,
                phase_agg=phase_agg,
                phase_power=phase_power,
                eps=eps,
                return_debug=True,
            )
            out[i] = psi_val
            if debug_data is not None and dbg:
                for key in debug_data:
                    debug_data[key][i] = dbg.get(key, math.nan)
        else:
            out[i] = mellin_complex_cepstral_phase(
                window_arr,
                grid_n=grid_n,
                sigma=sigma,
                detrend_phase=detrend_phase,
                min_bin=min_bin,
                max_frac=max_frac,
                phase_agg=phase_agg,
                phase_power=phase_power,
                eps=eps,
                return_debug=False,
            )
    
    series_out = pd.Series(out, index=series.index)
    if not return_debug:
        return series_out
    debug_df = pd.DataFrame(debug_data, index=series.index)
    return series_out, debug_df
