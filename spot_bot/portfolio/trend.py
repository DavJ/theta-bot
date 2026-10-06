"""Fixed, completed-day trend allocations for offline spot portfolio research."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from spot_bot.strategies.multi_strategy import _positions


PORTFOLIO_APPROACHES = ("btc_breakout", "breakout_equal", "momentum_vol",
                        "horizons_equal", "horizons_vol", "rotation")


@dataclass(frozen=True)
class TrendPortfolio:
    approach: str
    max_exposure: float = 0.3
    asset_cap: float = 0.15
    rebalance_weekday: int = 0
    rebalance_band: float = 0.01

    def __post_init__(self):
        if self.approach not in PORTFOLIO_APPROACHES:
            raise ValueError("Unknown portfolio approach")
        if not (0 < self.asset_cap <= self.max_exposure <= 1
                and 0 <= self.rebalance_band <= self.max_exposure
                and self.rebalance_weekday in range(7)):
            raise ValueError("Invalid spot allocation limits or schedule")

    def weights(self, close: pd.DataFrame) -> pd.DataFrame:
        """Targets as of each completed close; execution MUST lag these by a day."""
        if (close.empty or not isinstance(close.index, pd.DatetimeIndex)
                or not close.index.is_unique or not close.index.is_monotonic_increasing
                or not close.columns.is_unique
                or not np.isfinite(close.to_numpy()).all() or (close <= 0).any().any()):
            raise ValueError("Ordered, unique daily timestamps and positive finite prices required")
        zero = close * 0.0
        if "breakout" in self.approach:
            high = close.rolling(20, min_periods=20).max().shift(1)
            low = close.rolling(10, min_periods=10).min().shift(1)
            active = pd.DataFrame({symbol: _positions(close[symbol] > high[symbol],
                                                      close[symbol] < low[symbol])
                                   for symbol in close}, index=close.index)
            if self.approach == "btc_breakout":
                if "BTCUSDT" not in close:
                    raise ValueError("BTC-only control requires BTCUSDT")
                zero["BTCUSDT"] = active.BTCUSDT * self.max_exposure
                return zero
            targets = active.div(active.sum(axis=1).replace(0, np.nan), axis=0) * self.max_exposure
        elif self.approach == "rotation":
            momentum = close.pct_change(90, fill_method=None)
            # Stable column order resolves ties without a look-ahead rank.
            eligible = (momentum > 0) & (momentum.rank(axis=1, ascending=False, method="first") <= 2)
            targets = eligible.astype(float) * (self.max_exposure / 2)
        else:
            volatility = close.pct_change(fill_method=None).rolling(60, min_periods=60).std(ddof=0).clip(lower=0.0001)
            if self.approach == "momentum_vol":
                active = close.pct_change(30, fill_method=None).gt(0)
                score = active.astype(float).div(volatility)
                targets = score.div(score.sum(axis=1).replace(0, np.nan), axis=0) * self.max_exposure
            else:
                returns = [close.pct_change(days, fill_method=None) for days in (30, 90, 180, 365)]
                ready = np.logical_and.reduce([r.notna().to_numpy() for r in returns])
                conviction = sum(r.gt(0).astype(float) for r in returns) / len(returns)
                conviction = conviction.where(ready, 0.0)
                base = close * 0 + 1.0 if self.approach == "horizons_equal" else 1.0 / volatility
                targets = base.div(base.sum(axis=1), axis=0) * conviction * self.max_exposure
        return targets.fillna(0.0).clip(lower=0.0, upper=self.asset_cap)
