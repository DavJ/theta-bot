"""Completed-day classical signals, exposure ensembles and forecast fusion.

The hourly execution path sees a daily observation only after that UTC day is
complete. No daily close is broadcast backwards into its unfinished day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spot_bot.strategies.base import Intent, Strategy
from spot_bot.strategies.forecast_fusion import calibrate_and_fuse
from spot_bot.strategies.meanrev_dual_kalman import MeanRevDualKalmanStrategy


MULTI_APPROACHES = ("momentum", "ema_trend", "breakout", "range_reversion",
                    "ensemble_equal", "ensemble_regime", "forecast_mean", "kalman_fusion")


def _positions(entry, exit_):
    active = False
    result = []
    for enter, leave in zip(entry.fillna(False), exit_.fillna(False)):
        if active and leave:
            active = False
        elif not active and enter:
            active = True
        result.append(float(active))
    return pd.Series(result, index=entry.index)


class MultiStrategy(Strategy):
    owns_regime = True
    allow_loss_exits = True

    def __init__(self, approach="kalman_fusion", max_exposure=0.3,
                 fee_rate=0.001, slippage_bps=5.0, spread_bps=2.0):
        if approach not in MULTI_APPROACHES:
            raise ValueError(f"Unsupported multi-strategy approach: {approach}")
        if not 0 < max_exposure <= 1 or min(fee_rate, slippage_bps, spread_bps) < 0:
            raise ValueError("Positive bounded exposure and nonnegative costs required")
        self.approach = approach
        self.max_exposure = float(max_exposure)
        self.round_trip_cost = 2 * fee_rate + (2 * slippage_bps + spread_bps) / 10000

    def _daily_inputs(self, features):
        if "timestamp" in features:
            timestamps = pd.to_datetime(features.timestamp, utc=True)
        elif isinstance(features.index, pd.DatetimeIndex):
            timestamps = pd.to_datetime(features.index, utc=True)
        else:
            raise ValueError("Multi-strategy signals require real candle timestamps")
        close = pd.Series(pd.to_numeric(features.close, errors="coerce").to_numpy(), index=timestamps)
        if not close.index.is_unique or not close.index.is_monotonic_increasing:
            raise ValueError("Candle timestamps must be ordered and unique")
        if not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
            raise ValueError("Close prices must be finite and positive")
        # A day's last observed close is labelled at the NEXT midnight. The
        # incomplete final bucket lies in the future and never maps to past bars.
        daily = close.resample("1D", label="right", closed="left").last().dropna()
        return close, daily

    def _daily_models(self, features):
        hourly, daily = self._daily_inputs(features)
        vol = daily.pct_change(fill_method=None).rolling(20, min_periods=20).std(ddof=0).clip(lower=0.0001)
        fast = daily.ewm(span=20, adjust=False, min_periods=20).mean()
        slow = daily.ewm(span=60, adjust=False, min_periods=60).mean()
        gap = (fast - slow) / daily
        momentum = daily.pct_change(30, fill_method=None)
        high = daily.rolling(20, min_periods=20).max().shift(1)
        low = daily.rolling(20, min_periods=20).min().shift(1)
        exit_low = daily.rolling(10, min_periods=10).min().shift(1)
        center = daily.rolling(20, min_periods=20).mean()
        sigma = daily.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
        z = (daily - center) / sigma
        range_regime = gap.between(-0.01, 0.03)
        range_active = False
        range_entry = 0.0
        range_position = []
        for price, z_val, allowed in zip(daily, z, range_regime):
            if range_active and (not allowed or z_val >= 0 or price <= range_entry * 0.95):
                range_active = False
            elif not range_active and allowed and z_val < -2:
                range_active = True
                range_entry = float(price)
            range_position.append(float(range_active))

        components = pd.DataFrame({
            "momentum": _positions(momentum > self.round_trip_cost, momentum < -self.round_trip_cost),
            "ema_trend": _positions(gap > self.round_trip_cost, gap < -self.round_trip_cost),
            "breakout": _positions(daily > high, daily < exit_low),
            "range_reversion": range_position,
        }, index=daily.index) * self.max_exposure
        scores = pd.DataFrame({
            "momentum": np.tanh(momentum / (vol * np.sqrt(30))),
            "ema_trend": np.tanh(gap / (vol * np.sqrt(20))),
            "breakout": (2 * (daily - low) / (high - low).replace(0.0, np.nan) - 1).clip(-1, 1),
            "range_reversion": (-np.tanh(z / 2)).where(range_regime, 0.0),
        }, index=daily.index)
        needs_theta = self.approach not in {"momentum", "ema_trend", "breakout", "range_reversion"}
        if needs_theta:
            theta = MeanRevDualKalmanStrategy(price_space="log_vol").generate_series(
                features, risk_budgets=pd.Series(1.0, index=features.index))
            # Theta's own regime gate belongs to its source, not to every model.
            state = pd.to_numeric(features.get("S", pd.Series(1.0, index=features.index)), errors="coerce")
            theta = theta.where(state >= 0.2, 0.0)
            theta_hourly = pd.Series(theta.to_numpy(), index=hourly.index)
            theta_daily = theta_hourly.resample("1D", label="right", closed="left").last().reindex(daily.index)
            scores.insert(0, "theta", theta_daily)
            components.insert(0, "theta", theta_daily.clip(0, self.max_exposure))
        return hourly, daily, components, scores, gap, vol

    def generate_frame(self, features):
        if features is None or features.empty:
            return pd.DataFrame(index=getattr(features, "index", None))
        hourly, daily, components, scores, gap, vol = self._daily_models(features)
        diagnostics = pd.DataFrame(index=daily.index)
        if self.approach in components:
            target = components[self.approach]
        elif self.approach == "ensemble_equal":
            target = components.mean(axis=1)
        elif self.approach == "ensemble_regime":
            strength = (gap.abs() / (vol * np.sqrt(20))).clip(0, 1).fillna(0.0)
            trend_mean = components[["momentum", "ema_trend", "breakout"]].mean(axis=1)
            target = 0.2 * components.theta + (0.6 + 0.2 * strength) * trend_mean + 0.2 * (1 - strength) * components.range_reversion
            diagnostics["trend_strength"] = strength
        else:
            diagnostics = calibrate_and_fuse(scores, daily)
            forecast = diagnostics.get("fused_return" if self.approach == "kalman_fusion" else "mean_return",
                                       pd.Series(np.nan, index=daily.index))
            hurdle = self.round_trip_cost / 20
            target = _positions(forecast > hurdle, forecast < -hurdle) * self.max_exposure
            target = target.where(forecast.notna(), 0.0)
        diagnostics["desired_exposure"] = target.fillna(0.0).clip(0.0, self.max_exposure)
        mapped = diagnostics.reindex(hourly.index, method="ffill")
        mapped.index = features.index
        mapped["desired_exposure"] = mapped.desired_exposure.fillna(0.0)
        return mapped

    def generate_series(self, features):
        frame = self.generate_frame(features)
        return frame.get("desired_exposure", pd.Series(0.0, index=features.index))

    def generate_intent(self, features):
        frame = self.generate_frame(features)
        if frame.empty:
            return Intent(0.0, reason="No closed history", diagnostics={})
        last = frame.iloc[-1]
        diagnostics = {k: float(v) for k, v in last.items() if pd.notna(v)}
        return Intent(float(last.desired_exposure), reason=self.approach, diagnostics=diagnostics)
