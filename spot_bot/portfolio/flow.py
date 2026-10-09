"""Fixed completed-4h-bar spot signals; no loss-based sizing or exchange access."""
from __future__ import annotations

import numpy as np
import pandas as pd

from spot_bot.portfolio.trend import TrendPortfolio

FLOW_VARIANTS = ("price_momentum", "flow_momentum", "price_breakout", "flow_breakout",
                 "flow_reversion", "flow_blend")


def flow_features(frame):
    """Executed taker flow; trailing standardization excludes the current bar."""
    close = frame.close
    pressure = ((2 * frame.taker_buy_base - frame.volume).rolling(3, min_periods=3).sum()
                / frame.volume.rolling(3, min_periods=3).sum().replace(0, np.nan))
    mean = pressure.shift(1).rolling(180, min_periods=84).mean()
    scale = pressure.shift(1).rolling(180, min_periods=84).std(ddof=0).clip(lower=.001)
    log_price = np.log(close)
    price_mean = log_price.rolling(12, min_periods=12).mean()
    price_scale = log_price.rolling(12, min_periods=12).std(ddof=0).clip(lower=1e-6)
    return pd.DataFrame({"flow_z": (pressure - mean) / scale,
        "price_z": (log_price - price_mean) / price_scale,
        "momentum_24h": close.pct_change(6, fill_method=None),
        "momentum_7d": close.pct_change(42, fill_method=None),
        "channel_high": close.rolling(12, min_periods=12).max().shift(1),
        "channel_low": close.rolling(3, min_periods=3).min().shift(1),
        "volatility": close.pct_change(fill_method=None).rolling(84, min_periods=84).std(ddof=0).clip(lower=.0001)},
        index=frame.index)


def _held(entry, exit_, maximum_bars=None):
    values = np.zeros(len(entry), dtype=float)
    active, age = False, 0
    for i, (enter, leave) in enumerate(zip(entry, exit_)):
        if active:
            age += 1
            if leave or (maximum_bars is not None and age >= maximum_bars):
                active = False
                # No simultaneous re-entry on a timed/risk exit.
                values[i] = 0
                continue
        if not active and enter:
            active, age = True, 0
        values[i] = float(active)
    return values


def spot_flow_targets(markets):
    """Return targets at the completed close; replay shifts them to the next open."""
    symbols = list(markets)
    index = next(iter(markets.values())).index
    states = {name: pd.DataFrame(0., index=index, columns=symbols) for name in FLOW_VARIANTS[:-1]}
    volatility = states["price_momentum"].copy()
    for symbol, frame in markets.items():
        f = flow_features(frame)
        volatility[symbol] = f.volatility
        trend = f.momentum_7d > 0
        momentum = trend & (f.momentum_24h > 0)
        states["price_momentum"][symbol] = momentum.astype(float)
        states["flow_momentum"][symbol] = (momentum & (f.flow_z > .5)).astype(float)
        leave = frame.close < f.channel_low
        enter = frame.close > f.channel_high
        states["price_breakout"][symbol] = _held(enter.to_numpy(), leave.to_numpy())
        states["flow_breakout"][symbol] = _held((enter & (f.flow_z > .5)).to_numpy(), leave.to_numpy())
        states["flow_reversion"][symbol] = _held(
            (trend & (f.price_z < -2) & (f.flow_z < -1.5)).to_numpy(),
            ((f.price_z >= 0) | ~trend).to_numpy(), maximum_bars=6)
    result = {}
    for name, active in states.items():
        score = active.div(volatility)
        result[name] = (score.div(score.sum(axis=1).replace(0, np.nan), axis=0)
                        .mul(.5).fillna(0).clip(lower=0, upper=.25))
    result["flow_blend"] = sum(result[n] for n in ("flow_momentum", "flow_breakout", "flow_reversion")) / 3
    return result


def daily_control_targets(markets):
    """Publish the old daily target at each day's final 4h close (bar label 20h)."""
    daily = {s: f.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                     "close": "last", "volume": "sum"}) for s, f in markets.items()}
    close = pd.DataFrame({s: f.close for s, f in daily.items()})
    target = TrendPortfolio("momentum_vol", max_exposure=.5, asset_cap=.25).weights(close)
    target.index = target.index + pd.Timedelta("20h")
    index = next(iter(markets.values())).index
    return target.reindex(index, method="ffill").fillna(0), daily
