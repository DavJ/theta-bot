"""
Theta and Mellin transform feature extraction for dual-stream model.

This module provides functions to compute:
1. Theta basis projection coefficients from rolling windows of log-returns
2. Mellin transform features from theta signals
3. Combined dual-stream inputs for the DualStreamModel
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd


def build_theta_embedding(
    close: pd.Series,
    window: int,
    q: float,
    n_terms: int,
    mode: Literal["coeffs", "recon", "both"] = "both",
) -> Dict[str, np.ndarray]:
    """
    Build theta basis embedding from rolling windows of log-returns.

    Uses a truncated theta_3-like basis to fit rolling windows of detrended
    log-returns via least squares. This produces time-aligned theta coefficients
    and optionally reconstructed signals.

    Parameters
    ----------
    close : pd.Series
        Close price series with DatetimeIndex
    window : int
        Rolling window size for theta basis fitting
    q : float
        Theta basis parameter (typically 0 < q < 1, e.g., 0.9)
        Controls the decay of higher-order terms
    n_terms : int
        Number of theta basis terms to compute (positive integers)
    mode : {"coeffs", "recon", "both"}
        - "coeffs": Return only theta coefficients
        - "recon": Return only reconstructed signal
        - "both": Return both coefficients and reconstruction

    Returns
    -------
    dict with keys:
        - "theta_coeffs": np.ndarray shape (T, n_terms)
          Theta coefficients aligned to timestamps (NaN for first window-1 rows)
        - "theta_recon": np.ndarray shape (T, window)
          Reconstructed signal per window (only if mode includes "recon")

    Notes
    -----
    The theta basis is defined as:
        X[t, n] = q^(n^2) * cos(2*n*phi_t)
    where phi_t spans [0, 2π] uniformly across the window.

    Implementation ensures causality: each timestamp uses only data up to that time.
    """
    T = len(close)

    # Compute log-returns
    log_price = np.log(close.values)
    log_returns = np.diff(log_price, prepend=log_price[0])

    # Prepare outputs
    theta_coeffs = np.full((T, n_terms), np.nan)
    theta_recon = np.full((T, window), np.nan) if mode in ["recon", "both"] else None

    # Build theta basis matrix once (window x n_terms)
    # phi uniformly spans [0, 2π] across window
    phi = np.linspace(0, 2 * np.pi, window)
    basis = np.zeros((window, n_terms))
    for n in range(n_terms):
        # n+1 to start from 1st harmonic (0th term would be constant)
        basis[:, n] = (q ** ((n + 1) ** 2)) * np.cos(2 * (n + 1) * phi)

    # Rolling window computation (causal)
    for i in range(window - 1, T):
        # Extract window of log-returns ending at position i
        window_data = log_returns[i - window + 1 : i + 1]

        # Detrend: subtract mean to remove DC component
        window_data = window_data - window_data.mean()

        # Fit coefficients using least squares
        # Solve: basis @ coeffs = window_data
        coeffs, _, _, _ = np.linalg.lstsq(basis, window_data, rcond=None)
        theta_coeffs[i, :] = coeffs

        # Reconstruct if needed
        if theta_recon is not None:
            recon = basis @ coeffs
            theta_recon[i, :] = recon

    result = {"theta_coeffs": theta_coeffs}
    if mode in ["recon", "both"]:
        result["theta_recon"] = theta_recon

    return result


def mellin_transform_features(
    theta_signal_or_coeffs: np.ndarray,
    mellin_k: int,
    alpha: float = 0.5,
    omega_max: float = 1.0,
    use_log_t: bool = True,
    output: Literal["reim", "magphase", "mag"] = "mag",
) -> np.ndarray:
    """
    Compute Mellin transform features from theta signals or coefficients.

    The Mellin transform is a frequency-domain representation that captures
    scale-invariant features. For each row (timestamp), we compute:
        M(s) = Σ_{t=1}^{N} x[t] * t^(s-1)
    where s = alpha + i*omega, with omega sampled uniformly.

    Parameters
    ----------
    theta_signal_or_coeffs : np.ndarray
        Input array shape (T, N) where T is number of timestamps,
        N is either n_terms (coefficients) or window length (reconstructed signal)
    mellin_k : int
        Number of frequency samples for Mellin transform
    alpha : float
        Real part of Mellin transform parameter s = alpha + i*omega
        Typically 0.5 for good numerical stability
    omega_max : float
        Maximum imaginary frequency to sample (omega in [0, omega_max])
    use_log_t : bool
        If True, use log(t+1) instead of t for better numerical stability
    output : {"reim", "magphase", "mag"}
        Output format:
        - "reim": Real and imaginary parts (2*K features)
        - "magphase": Magnitude and phase (2*K features)
        - "mag": Magnitude only (K features)

    Returns
    -------
    np.ndarray
        Feature array shape (T, F) where F depends on output format:
        - "mag": F = mellin_k
        - "reim" or "magphase": F = 2*mellin_k

    Notes
    -----
    The implementation uses vectorized operations for efficiency.
    Normalization is applied to ensure numerical stability.
    """
    T, N = theta_signal_or_coeffs.shape
    omegas = np.linspace(0, omega_max, mellin_k)

    # Prepare index array t (1, 2, ..., N)
    t_vals = np.arange(1, N + 1, dtype=float)
    if use_log_t:
        t_vals = np.log(t_vals + 1)

    # Compute Mellin transform for each timestamp
    mellin_features = []

    for row_idx in range(T):
        x = theta_signal_or_coeffs[row_idx, :]

        # Skip if all NaN (early timestamps before window fills)
        if np.all(np.isnan(x)):
            if output == "mag":
                mellin_features.append(np.full(mellin_k, np.nan))
            else:
                mellin_features.append(np.full(2 * mellin_k, np.nan))
            continue

        # Replace NaN with 0 for computation
        x = np.nan_to_num(x, nan=0.0)

        # Normalize to unit L1 norm for stability
        norm = np.sum(np.abs(x)) + 1e-9
        x = x / norm

        # Compute M(s) for each omega
        # s = alpha + i*omega
        # t^(s-1) = t^(alpha-1) * exp(i*omega*log(t))
        mellin_vals = np.zeros(mellin_k, dtype=complex)

        for k, omega in enumerate(omegas):
            # Compute t^(alpha-1) * exp(i*omega*log(t_vals))
            if use_log_t:
                # If using log_t, t_vals is already log(t+1)
                exponent = (alpha - 1) * t_vals + 1j * omega * t_vals
            else:
                log_t = np.log(t_vals)
                exponent = (alpha - 1) * log_t + 1j * omega * log_t

            weights = np.exp(exponent)
            mellin_vals[k] = np.sum(x * weights)

        # Convert to output format
        if output == "mag":
            features = np.abs(mellin_vals)
        elif output == "reim":
            features = np.concatenate([mellin_vals.real, mellin_vals.imag])
        elif output == "magphase":
            features = np.concatenate([np.abs(mellin_vals), np.angle(mellin_vals)])
        else:
            raise ValueError(f"Unknown output format: {output}")

        mellin_features.append(features)

    return np.array(mellin_features)


def build_dual_stream_inputs(
    df: pd.DataFrame,
    window: int = 48,
    q: float = 0.9,
    n_terms: int = 8,
    mellin_k: int = 16,
    alpha: float = 0.5,
    omega_max: float = 1.0,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Build dual-stream inputs (theta + Mellin) from OHLCV dataframe.

    This is the main entry point for preparing inputs to DualStreamModel.
    Computes theta embeddings and Mellin features, then aligns and drops
    initial NaN rows to produce clean input arrays.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe with DatetimeIndex and 'close' column
    window : int
        Rolling window size for theta basis (default: 48)
    q : float
        Theta basis decay parameter (default: 0.9)
    n_terms : int
        Number of theta coefficients (default: 8)
    mellin_k : int
        Number of Mellin frequency samples (default: 16)
    alpha : float
        Mellin transform real parameter (default: 0.5)
    omega_max : float
        Mellin transform max frequency (default: 1.0)

    Returns
    -------
    tuple of (index, X_theta, X_mellin):
        - index : pd.DatetimeIndex
          Valid timestamps (after dropping NaN rows)
        - X_theta : np.ndarray shape (N, window)
          Theta reconstructed signals for each timestamp
        - X_mellin : np.ndarray shape (N, mellin_k)
          Mellin features computed from theta coefficients

    Notes
    -----
    The returned arrays have aligned indices and no NaN values.
    First (window-1) timestamps are dropped due to causality constraints.
    """
    close = df["close"]

    # Build theta embedding
    theta_dict = build_theta_embedding(
        close=close, window=window, q=q, n_terms=n_terms, mode="both"
    )

    theta_coeffs = theta_dict["theta_coeffs"]  # (T, n_terms)
    theta_recon = theta_dict["theta_recon"]  # (T, window)

    # Compute Mellin features from theta coefficients
    mellin_feats = mellin_transform_features(
        theta_signal_or_coeffs=theta_coeffs,
        mellin_k=mellin_k,
        alpha=alpha,
        omega_max=omega_max,
        output="mag",
    )

    # Create dataframe for alignment and NaN dropping
    result_df = pd.DataFrame(index=df.index)

    # Add theta recon features (use as X_theta input)
    for i in range(window):
        result_df[f"theta_recon_{i}"] = theta_recon[:, i]

    # Add Mellin features
    for i in range(mellin_k):
        result_df[f"mellin_{i}"] = mellin_feats[:, i]

    # Drop rows with any NaN
    result_df = result_df.dropna()

    # Extract arrays
    index = result_df.index
    theta_cols = [f"theta_recon_{i}" for i in range(window)]
    mellin_cols = [f"mellin_{i}" for i in range(mellin_k)]

    X_theta = result_df[theta_cols].values  # (N, window)
    X_mellin = result_df[mellin_cols].values  # (N, mellin_k)

    return index, X_theta, X_mellin
