from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from theta_features.cepstrum import EPS_LOG, rolling_cepstral_phase, rolling_complex_cepstral_phase
from theta_features.log_phase_core import (
    log_phase,
    phase_embedding,
    rolling_internal_concentration,
    rolling_phase_concentration,
)


@dataclass(frozen=True)
class FeatureConfig:
    base: float = 10.0
    rv_window: int = 24
    conc_window: int = 256
    psi_mode: str = "cepstrum"
    psi_window: int = 256
    cepstrum_domain: str = "logtime"
    cepstrum_min_bin: int = 4
    cepstrum_max_frac: float = 0.2
    cepstrum_topk: Optional[int] = None


def _resolve_timestamp(ohlcv_df: pd.DataFrame) -> pd.Series:
    if "timestamp" in ohlcv_df.columns:
        ts = pd.to_datetime(ohlcv_df["timestamp"], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(ohlcv_df.index, utc=True, errors="coerce")
    return ts


def _compute_rv(log_returns: pd.Series, window: int) -> pd.Series:
    r2_sum = log_returns.pow(2).rolling(window=window, min_periods=window).sum()
    return r2_sum.pow(0.5)


def _compute_psi(log_rv: pd.Series, cfg: FeatureConfig) -> tuple[pd.Series, Optional[pd.DataFrame]]:
    mode = str(cfg.psi_mode or "none").lower()
    if mode == "none":
        return pd.Series(np.nan, index=log_rv.index), None
    if mode not in ("cepstrum", "complex_cepstrum"):
        raise ValueError(f"Unsupported psi_mode: {cfg.psi_mode}")
    domain = (cfg.cepstrum_domain or "linear").lower()
    if mode == "complex_cepstrum":
        return rolling_complex_cepstral_phase(
            log_rv,
            window=int(cfg.psi_window),
            min_bin=int(cfg.cepstrum_min_bin),
            max_frac=float(cfg.cepstrum_max_frac),
            domain=domain,
            eps=EPS_LOG,
            return_debug=True,
        )
    return (
        rolling_cepstral_phase(
            log_rv,
            window=int(cfg.psi_window),
            min_bin=int(cfg.cepstrum_min_bin),
            max_frac=float(cfg.cepstrum_max_frac),
            topk=cfg.cepstrum_topk,
            domain=domain,
            eps=EPS_LOG,
        ),
        None,
    )


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
    psi_debug = None
    if cfg.psi_window and int(cfg.psi_window) > 0:
        log_rv = pd.Series(np.log(np.abs(rv) + EPS_LOG), index=close.index)
        psi, psi_debug = _compute_psi(log_rv, cfg)
        feature_data["psi"] = psi

    df_feat = pd.DataFrame(feature_data, index=close.index)
    df_feat["psi_mode"] = str(cfg.psi_mode)
    debug_cols = ("psi_n_star", "psi_c_real", "psi_c_imag", "psi_c_abs", "psi_angle_rad")
    if psi_debug is not None:
        for col in debug_cols:
            df_feat[col] = psi_debug[col]
    else:
        for col in debug_cols:
            if col not in df_feat.columns:
                df_feat[col] = np.nan

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
