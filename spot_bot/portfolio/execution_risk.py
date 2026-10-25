"""Completed-day covariance publication for optional offline 4h risk checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from spot_bot.portfolio.risk import PortfolioRisk


def completed_risk_covariances(markets, risk: PortfolioRisk) -> np.ndarray:
    """Publish a complete daily covariance at 20h; replay must lag it once.

    A final partial day never publishes its incomplete close. Missing history is
    represented by a whole NaN matrix and requests zero exposure in the guard.
    """
    if not markets or not isinstance(risk, PortfolioRisk):
        raise ValueError("Spot markets and PortfolioRisk required")
    first = next(iter(markets.values())).index
    if (not isinstance(first, pd.DatetimeIndex) or first.tz is None or first.empty
            or not first.is_unique or not first.is_monotonic_increasing
            or not first.to_series().diff().dropna().eq(pd.Timedelta('4h')).all()
            or any(not f.index.equals(first) for f in markets.values())):
        raise ValueError("Aligned complete timezone-aware 4h grid required")
    index = first.tz_convert('UTC')
    if ((index.minute != 0).any() or (index.second != 0).any()
            or (index.microsecond != 0).any() or (index.nanosecond != 0).any()
            or (index.hour % 4 != 0).any()):
        raise ValueError("UTC four-hour open labels required")
    closes = pd.DataFrame({s: f.close.to_numpy(dtype=float) for s, f in markets.items()}, index=index)
    if not np.isfinite(closes.to_numpy()).all() or (closes <= 0).any().any():
        raise ValueError("Finite positive spot closes required")
    counts = closes.resample('1D').size()
    daily = closes.resample('1D').last().loc[counts.eq(6)]
    covariances = risk.covariances(daily)
    result = np.full((len(index), len(markets), len(markets)), np.nan)
    if not daily.empty:
        publication = daily.index + pd.Timedelta('20h')
        positions = publication.searchsorted(index, side='right') - 1
        ready = positions >= 0
        result[ready] = covariances[positions[ready]]
    return result
