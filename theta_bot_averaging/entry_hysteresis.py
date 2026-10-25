"""Fixed causal momentum-entry hysteresis for optional offline spot research."""
from __future__ import annotations

import numpy as np
import pandas as pd

from spot_bot.portfolio.momentum_refinement import _entry_state, _redistribute_capped

ENTRY_HURDLE = .0064  # Twice the approximate nominal 0.32% round-trip allowance.
VARIANTS = ('guarded_edge','entry_hysteresis','symmetric_hurdle',
            'two_day_confirmation','hysteresis_confirmed')


def eligibility_states(momentum, ready, variant):
    if variant not in VARIANTS or not momentum.index.equals(ready.index) or list(momentum)!=list(ready):
        raise ValueError('Aligned fixed momentum eligibility inputs required')
    if not ready.dtypes.eq(bool).all():
        raise ValueError('Explicit nonnullable warmup flags required')
    positive = momentum.gt(0) & ready
    if variant=='guarded_edge':
        return positive
    strong = momentum.gt(ENTRY_HURDLE) & ready
    if variant=='symmetric_hurdle':
        return strong
    entry = strong if variant in ('entry_hysteresis','hysteresis_confirmed') else positive
    if variant in ('two_day_confirmation','hysteresis_confirmed'):
        entry = entry.rolling(2,min_periods=2).sum().eq(2)
    return pd.DataFrame({s:_entry_state(entry[s].to_numpy(),momentum[s].to_numpy()).astype(bool)
                         for s in momentum},index=momentum.index)


def entry_weights(close, variant):
    """Eligibility state is a signal, not a record of filled positions."""
    if (not isinstance(close,pd.DataFrame) or close.empty or close.shape[1]!=3
            or not isinstance(close.index,pd.DatetimeIndex) or close.index.tz is None
            or not close.index.is_unique or not close.index.is_monotonic_increasing
            or not close.index.to_series().diff().dropna().eq(pd.Timedelta('1D')).all()
            or not np.isfinite(close.to_numpy()).all() or (close<=0).any().any()):
        raise ValueError('Complete positive timezone-aware daily prices required')
    momentum = close.pct_change(30,fill_method=None)
    volatility = close.pct_change(fill_method=None).rolling(60,min_periods=60).std(ddof=0).clip(lower=.0001)
    active = eligibility_states(momentum,volatility.notna(),variant)
    scores = active.astype(float).div(volatility).fillna(0.)
    values = np.array([_redistribute_capped(row,.5,.25) for row in scores.to_numpy()])
    return pd.DataFrame(values,index=close.index,columns=close.columns)


def completed_entry_targets(markets):
    if set(markets) != {'BTCUSDT','ETHUSDT','BNBUSDT'}:
        raise ValueError('The three fixed spot markets are required')
    first = next(iter(markets.values())).index
    if (not isinstance(first,pd.DatetimeIndex) or first.empty or first.tz is None
            or not first.is_unique or not first.is_monotonic_increasing
            or not first.to_series().diff().dropna().eq(pd.Timedelta('4h')).all()
            or any(not f.index.equals(first) for f in markets.values())):
        raise ValueError('Aligned complete timezone-aware 4h markets required')
    index = first.tz_convert('UTC')
    if ((index.hour%4!=0).any() or (index.minute!=0).any() or (index.second!=0).any()
            or (index.microsecond!=0).any() or (index.nanosecond!=0).any()):
        raise ValueError('UTC four-hour labels required')
    close = pd.DataFrame({s:f.close.to_numpy(dtype=float) for s,f in markets.items()},index=index)
    if not np.isfinite(close.to_numpy()).all() or (close<=0).any().any():
        raise ValueError('Finite positive source closes required')
    daily = close.resample('1D').last().loc[close.resample('1D').size().eq(6)]
    result = {}
    for variant in VARIANTS:
        if daily.empty:
            target = pd.DataFrame(0.,index=index,columns=close.columns)
        else:
            weights = entry_weights(daily,variant)
            weights.index = weights.index+pd.Timedelta('20h')
            target = weights.reindex(index,method='ffill').fillna(0.)
        target.index = first
        result[variant] = target
    return result
