"""Fixed allocation references for offline comparison, never live defaults."""
from __future__ import annotations

import numpy as np
import pandas as pd

from spot_bot.portfolio.momentum_refinement import _redistribute_capped


REFERENCES = ('cash', 'equal_25', 'equal_50', 'inverse_vol_50')


def allocation_targets(markets):
    """Publish only completed UTC days; the replay must lag these targets.

    Equal allocations are weekly rebalanced portfolios under the replay's risk
    guard, not literal buy-and-hold. Inverse volatility removes only momentum
    eligibility from the existing capped strategy's daily allocation rule.
    """
    if set(markets) != {'BTCUSDT', 'ETHUSDT', 'BNBUSDT'}:
        raise ValueError('The three frozen spot markets are required')
    original = next(iter(markets.values())).index
    if (not isinstance(original, pd.DatetimeIndex) or original.tz is None or original.empty
            or not original.is_unique or not original.is_monotonic_increasing
            or not original.to_series().diff().dropna().eq(pd.Timedelta('4h')).all()
            or any(not f.index.equals(original) for f in markets.values())):
        raise ValueError('Aligned complete timezone-aware 4h grid required')
    index = original.tz_convert('UTC')
    if ((index.hour % 4 != 0).any() or (index.minute != 0).any()
            or (index.second != 0).any() or (index.microsecond != 0).any()
            or (index.nanosecond != 0).any()):
        raise ValueError('UTC four-hour open labels required')
    close = pd.DataFrame({s: f.close.to_numpy(dtype=float) for s, f in markets.items()}, index=index)
    if not np.isfinite(close.to_numpy()).all() or (close <= 0).any().any():
        raise ValueError('Finite positive spot closes required')
    daily = close.resample('1D').last().loc[close.resample('1D').size().eq(6)]
    volatility = daily.pct_change(fill_method=None).rolling(60, min_periods=60).std(ddof=0).clip(lower=.0001)
    score = (1. / volatility).fillna(0.)
    inverse = np.array([_redistribute_capped(row, .5, .25) for row in score.to_numpy()])
    inverse = inverse.reshape(len(daily), len(markets))
    daily_targets = {'cash': pd.DataFrame(0., index=daily.index, columns=close.columns),
        'equal_25': pd.DataFrame(.25 / 3, index=daily.index, columns=close.columns),
        'equal_50': pd.DataFrame(.5 / 3, index=daily.index, columns=close.columns),
        'inverse_vol_50': pd.DataFrame(inverse, index=daily.index, columns=close.columns)}
    targets = {}
    for name, target in daily_targets.items():
        target.index = target.index + pd.Timedelta('20h')
        published = target.reindex(index, method='ffill').fillna(0.)
        published.index = original
        targets[name] = published
    return targets


def realized_daily_volatility(equity, capital):
    """Close-to-close daily realized standard deviation, including initial NAV."""
    values = equity.set_index('timestamp').equity.resample('1D').last().to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any() or capital <= 0:
        raise ValueError('Positive funded daily equity required')
    returns = values / np.r_[capital, values[:-1]] - 1.
    return float(returns.std(ddof=0) * np.sqrt(365))
