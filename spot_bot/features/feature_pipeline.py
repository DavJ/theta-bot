from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from theta_features.log_phase_core import (
    log_phase,
    phase_embedding,
    rolling_internal_concentration,
    rolling_phase_concentration,
)
from theta_features.scale_phase import compute_scale_phase


@dataclass(frozen=True)
class FeatureConfig:
    base: float = 10.0
    rv_window: int = 24
    conc_window: int = 256
    psi_mode: str = "scale_phase"
    psi_window: int = 256
    # Legacy spectral parameters retained for backward compatibility (ignored by scale-phase psi)
    cepstrum_domain: str = "logtime"
    cepstrum_min_bin: int = 4
    cepstrum_max_frac: float = 0.2
    cepstrum_topk: Optional[int] = None
    # Mellin transform parameters
    mellin_grid_n: int = 256
    mellin_sigma: float = 0.0
    mellin_eps: float = 1e-12
    mellin_detrend_phase: bool = True
    psi_min_bin: int = 2
    psi_max_frac: float = 0.25
    psi_phase_agg: str = "peak"
    psi_phase_power: float = 1.0


def _resolve_timestamp(ohlcv_df: pd.DataFrame) -> pd.Series:
    """
    Resolve timestamps from either a 'timestamp' column or the DataFrame index.

    Supports both ISO timestamps and Unix epoch in milliseconds.
    """
    if "timestamp" in ohlcv_df.columns:
        s = ohlcv_df["timestamp"]
        if pd.api.types.is_numeric_dtype(s):
            ts = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(s, utc=True, errors="coerce")
    else:
        idx = ohlcv_df.index
        if pd.api.types.is_numeric_dtype(idx):
            ts = pd.to_datetime(idx, unit="ms", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(idx, utc=True, errors="coerce")
    return ts


def _compute_rv(log_returns: pd.Series, window: int) -> pd.Series:
    r2_sum = log_returns.pow(2).rolling(window=window, min_periods=window).sum()
    return r2_sum.pow(0.5)


def _normalize_psi_mode(mode: str | None) -> str:
    mode_norm = str(mode or "scale_phase").lower()
    if mode_norm in ("none",):
        return "none"
    if mode_norm in ("scale", "scale_phase"):
        return "scale_phase"
    raise ValueError(f"Unsupported psi_mode: {mode}")


def _compute_psi(rv: pd.Series, cfg: FeatureConfig) -> tuple[pd.Series, str]:
    mode = _normalize_psi_mode(cfg.psi_mode)
    if mode == "none":
        return pd.Series(np.nan, index=rv.index), mode
    psi = compute_scale_phase(rv, window=int(cfg.psi_window), base=float(cfg.base))
    return psi, mode


def compute_features(ohlcv_df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Build canonical regime features aligned with research implementations.

    Required columns: ['open', 'high', 'low', 'close', 'volume'] with either a
    'timestamp' column or a datetime index.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return pd.DataFrame()

    ts = _resolve_timestamp(ohlcv_df)
    close = pd.to_numeric(ohlcv_df["close"], errors="coerce")
    log_returns = np.log(close / close.shift(1))

    rv = _compute_rv(log_returns, window=int(cfg.rv_window))
    phi = log_phase(rv.to_numpy(), base=cfg.base)
    cos_phi, sin_phi = phase_embedding(phi)
    concentration = rolling_phase_concentration(phi, window=int(cfg.conc_window))

    feature_data = {
        "timestamp": ts,
        "close": close,
        "rv": rv,
        "phi": phi,
        "cos_phi": cos_phi,
        "sin_phi": sin_phi,
        "C": concentration,
    }

    psi = None
    psi_mode_value = _normalize_psi_mode(cfg.psi_mode)
    if cfg.psi_window and int(cfg.psi_window) > 0:
        psi, psi_mode_value = _compute_psi(rv, cfg)
        feature_data["psi"] = psi

    df_feat = pd.DataFrame(feature_data, index=close.index)
    df_feat["psi_mode"] = psi_mode_value

    if psi is not None:
        psi_vals = psi.to_numpy(dtype=float)
        cos_psi, sin_psi = phase_embedding(psi_vals)
        c_int = rolling_internal_concentration(
            np.asarray(cos_phi, dtype=float),
            np.asarray(sin_phi, dtype=float),
            cos_psi,
            sin_psi,
            window=int(cfg.conc_window),
        )
        df_feat["C_int"] = c_int
    else:
        df_feat["C_int"] = np.nan

    rank_c = df_feat["C"].rank(pct=True, method="average")
    rank_c_int = df_feat["C_int"].rank(pct=True, method="average")
    if df_feat["C_int"].notna().any():
        df_feat["S"] = ((rank_c + rank_c_int) / 2.0).where(df_feat["C_int"].notna())
    else:
        df_feat["S"] = rank_c

    return df_feat
