from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


DEFAULT_FEATURES: List[str] = [
    "log_return",
    "volatility_24",
    "volatility_72",
    "momentum_24",
    "momentum_72",
    "rsi_14",
    "volume_zscore",
    "phase_sin",
    "phase_cos",
]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(
    df: pd.DataFrame,
    feature_list: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Build a compact set of engineered features (classical + simple phase proxies).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least columns: open, high, low, close, volume.
    feature_list : Iterable[str], optional
        Subset or custom list of features to compute. Defaults to DEFAULT_FEATURES.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered feature columns aligned to df index.
    """
    feats = feature_list or DEFAULT_FEATURES
    out = pd.DataFrame(index=df.index)

    price = df["close"]
    log_price = np.log(price)
    log_return = log_price.diff()

    if "log_return" in feats:
        out["log_return"] = log_return
    if "volatility_24" in feats:
        out["volatility_24"] = log_return.rolling(24).std()
    if "volatility_72" in feats:
        out["volatility_72"] = log_return.rolling(72).std()
    if "momentum_24" in feats:
        out["momentum_24"] = price.pct_change(24)
    if "momentum_72" in feats:
        out["momentum_72"] = price.pct_change(72)
    if "rsi_14" in feats:
        out["rsi_14"] = _rsi(price, window=14)
    if "volume_zscore" in feats:
        vol = df["volume"]
        out["volume_zscore"] = (vol - vol.rolling(48).mean()) / vol.rolling(48).std()

    # Simple complex-time phase proxies (sin/cos of cumulative angle)
    phase = log_return.cumsum().fillna(0.0)
    if "phase_sin" in feats:
        out["phase_sin"] = np.sin(phase)
    if "phase_cos" in feats:
        out["phase_cos"] = np.cos(phase)

    return out
