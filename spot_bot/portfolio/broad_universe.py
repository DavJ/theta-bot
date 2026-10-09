"""Fixed historical-cohort spot momentum policies, with causal liquidity and gaps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spot_bot.portfolio.momentum_refinement import MomentumRefinement, _redistribute_capped
from spot_bot.portfolio.risk import PortfolioRisk

BROAD_VARIANTS = ("spot_control", "capped_control", "broad_inverse_vol", "broad_top3",
                  "broad_top3_vol", "broad_top5_vol", "broad_top5_consensus", "broad_top5_vol20")
CONTROL_ASSETS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
POLYGON_NOTICE = pd.Timestamp("2024-08-28 09:00", tz="UTC")


def complete_daily(markets):
    """Six actual 4h quotes required; no last-price or partial-day price filling."""
    daily = {}
    for symbol, frame in markets.items():
        aggregate = frame.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                              "close": "last", "volume": "sum", "quote_volume": "sum"})
        count = frame.close.resample("1D").count()
        daily[symbol] = aggregate.where(count.eq(6), np.nan, axis=0)
    return daily


def liquidity_eligibility(close, quote_volume):
    """90 contiguous completed days, $10m/day, historical top ten; ties use cohort order."""
    if (not close.index.equals(quote_volume.index) or list(close.columns) != list(quote_volume.columns)
            or not isinstance(close.index, pd.DatetimeIndex) or close.index.tz is None
            or close.empty or not close.index.is_unique or not close.index.is_monotonic_increasing
            or not close.index.to_series().diff().dropna().eq(pd.Timedelta("1D")).all()
            or np.isinf(close.to_numpy()).any() or np.isinf(quote_volume.to_numpy()).any()
            or (close <= 0).any().any() or (quote_volume < 0).any().any()
            or not close.isna().equals(quote_volume.isna())):
        raise ValueError("Aligned complete-day prices/volumes or explicit missing days required")
    ready = close.notna().rolling(90, min_periods=90).sum().eq(90)
    liquidity = quote_volume.rolling(90, min_periods=90).mean()
    liquid = ready & liquidity.ge(10_000_000.)
    ranked = liquidity.where(liquid, -np.inf).rank(axis=1, ascending=False, method="first")
    return liquid & ranked.le(10)


def weekly_top(score, eligible, count):
    """Sunday selection; daily disqualification exits, replacement waits until Sunday."""
    selected = score.where(eligible, -np.inf).rank(axis=1, ascending=False, method="first").le(count) & eligible
    sundays = pd.Series(score.index.dayofweek == 6, index=score.index)
    membership = selected.astype(float).where(sundays, np.nan, axis=0).ffill().fillna(0.).gt(0)
    return membership & eligible


def reduce_volatility(weights, close):
    """20% annual target, no scaling up; only selected covariance is required."""
    covariance = PortfolioRisk(volatility_window=60, covariance_shrinkage=.25).covariances(close)
    values = weights.to_numpy().copy()
    for i, row in enumerate(values):
        selected = np.flatnonzero(row > 0)
        if not len(selected):
            continue
        matrix = covariance[i][np.ix_(selected, selected)]
        if not np.isfinite(matrix).all():
            values[i] = 0.
            continue
        w = row[selected]
        annual_vol = np.sqrt(max(0., float(w @ matrix @ w)) * 365)
        if annual_vol > .20:
            values[i] *= .20 / annual_vol
    return pd.DataFrame(values, index=weights.index, columns=weights.columns)


@dataclass(frozen=True)
class BroadSpot:
    """Offline only, fixed 50% account and 25% asset limits; no fitted parameters."""
    variant: str

    def __post_init__(self):
        if self.variant not in BROAD_VARIANTS:
            raise ValueError("Unknown fixed broad-universe policy")

    def weights(self, close, quote_volume):
        eligible = liquidity_eligibility(close, quote_volume)
        if self.variant in ("spot_control", "capped_control"):
            result = pd.DataFrame(0., index=close.index, columns=close.columns)
            old_name = "spot_control" if self.variant == "spot_control" else "capped_allocation"
            completed = close.loc[:, list(CONTROL_ASSETS)].dropna()
            if not completed.empty:
                control = MomentumRefinement(old_name).weights(completed)
                result.loc[:, list(CONTROL_ASSETS)] = control.reindex(close.index, fill_value=0.)
            return result
        # The notice can affect the target only at that day's completed close.
        if "POLYGON" in eligible:
            pre_notice = close.index < POLYGON_NOTICE.normalize()
            pol_resumed = close.index >= pd.Timestamp("2024-09-13", tz="UTC")
            eligible["POLYGON"] &= pre_notice | pol_resumed
        momentum = {h: close.pct_change(h, fill_method=None) for h in (14, 30, 60)}
        volatility = close.pct_change(fill_method=None).rolling(60, min_periods=60).std(ddof=0).clip(lower=.0001)
        if self.variant == "broad_top5_consensus":
            active = eligible & (sum(momentum[h].gt(0).astype(int) for h in (14, 30, 60)) >= 2)
            z = np.stack([momentum[h].div(volatility * np.sqrt(h)).to_numpy() for h in (14, 30, 60)], axis=-1)
            score = pd.DataFrame(np.median(z, axis=-1), index=close.index, columns=close.columns)
        else:
            active = eligible & momentum[30].gt(0) & volatility.notna()
            score = momentum[30] if self.variant == "broad_top3" else momentum[30].div(volatility * np.sqrt(30))
        if self.variant == "broad_inverse_vol":
            scores = active.astype(float).div(volatility).fillna(0.)
            values = np.array([_redistribute_capped(row, .5, .25) for row in scores.to_numpy()])
            return pd.DataFrame(values, index=close.index, columns=close.columns)
        count = 3 if self.variant in ("broad_top3", "broad_top3_vol") else 5
        weights = weekly_top(score, active, count).astype(float) * (.5 / count)
        return reduce_volatility(weights, close) if self.variant == "broad_top5_vol20" else weights


def broad_targets(markets):
    """Publish with completed 20h candle; replay lags to next open, keeping NaNs in features."""
    daily = complete_daily(markets)
    close = pd.DataFrame({s: f.close for s, f in daily.items()})
    volume = pd.DataFrame({s: f.quote_volume for s, f in daily.items()})
    index = next(iter(markets.values())).index
    targets = {}
    for name in BROAD_VARIANTS:
        target = BroadSpot(name).weights(close, volume)
        target.index = target.index + pd.Timedelta("20h")
        targets[name] = target.reindex(index, method="ffill", fill_value=0.)
    quoted = pd.DataFrame({s: f.open.notna() for s, f in markets.items()})
    # Sentinels exist only in the replay view; source/indicator prices are still absent.
    executable = {s: f[["open", "high", "low", "close", "volume"]].fillna(0.) for s, f in markets.items()}
    return targets, executable, quoted, daily
