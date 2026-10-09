"""Fixed causal refinements of the daily spot momentum research control."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spot_bot.portfolio.flow import daily_control_targets
from spot_bot.portfolio.trend import TrendPortfolio


MOMENTUM_VARIANTS = ("spot_control", "wide_band", "confirmed_entry", "strong_entry",
                     "strong_entry_wide", "btc_regime", "capped_allocation")


def _entry_state(entry, momentum):
    """Signal eligibility persists until nonpositive momentum, independently of fills."""
    result = np.zeros(len(momentum), dtype=float)
    active = False
    for i, (enter, value) in enumerate(zip(entry, momentum)):
        if not np.isfinite(value) or value <= 0:
            active = False
        elif enter:
            active = True
        result[i] = float(active)
    return result


def _redistribute_capped(scores, budget, asset_cap):
    """Capped proportional allocation; uninvestable weight remains cash."""
    result = np.zeros_like(scores, dtype=float)
    available = scores > 0
    remaining = budget
    while available.any() and remaining > 1e-12:
        proposal = remaining * scores[available] / scores[available].sum()
        saturated = proposal >= asset_cap
        locations = np.flatnonzero(available)
        if not saturated.any():
            result[locations] = proposal
            break
        capped = locations[saturated]
        result[capped] = asset_cap
        available[capped] = False
        remaining -= len(capped) * asset_cap
    return result


@dataclass(frozen=True)
class MomentumRefinement:
    """Offline daily policy. Caller must execute after this completed close."""
    variant: str
    max_exposure: float = .5
    asset_cap: float = .25

    def __post_init__(self):
        if self.variant not in MOMENTUM_VARIANTS:
            raise ValueError("Unknown fixed momentum refinement")
        if (not np.isfinite([self.max_exposure, self.asset_cap]).all()
                or not 0 < self.asset_cap <= self.max_exposure <= 1):
            raise ValueError("Invalid owned-funds spot allocation limits")

    @property
    def approach(self):
        return self.variant

    @property
    def rebalance_weekday(self):
        return 0

    @property
    def rebalance_band(self):
        return .03 if self.variant in ("wide_band", "strong_entry_wide") else .01

    def weights(self, close: pd.DataFrame) -> pd.DataFrame:
        control = TrendPortfolio("momentum_vol", max_exposure=self.max_exposure,
                                 asset_cap=self.asset_cap).weights(close)
        if (close.index.tz is None or not close.index.to_series().diff().dropna()
                .eq(pd.Timedelta("1D")).all()):
            raise ValueError("Complete timezone-aware daily prices required")
        if self.variant in ("spot_control", "wide_band"):
            return control
        momentum = close.pct_change(30, fill_method=None)
        volatility = (close.pct_change(fill_method=None).rolling(60, min_periods=60)
                      .std(ddof=0).clip(lower=.0001))
        active = momentum.gt(0)
        if self.variant in ("confirmed_entry", "strong_entry", "strong_entry_wide"):
            if self.variant == "confirmed_entry":
                entry = active.rolling(3, min_periods=3).sum().eq(3) & volatility.notna()
            else:
                # A past strength heuristic, not an estimated future net edge or p-value.
                entry = momentum.gt(.25 * volatility * np.sqrt(30))
            active = pd.DataFrame({s: _entry_state(entry[s].to_numpy(), momentum[s].to_numpy())
                                   for s in close}, index=close.index)
        elif self.variant == "btc_regime":
            if "BTCUSDT" not in close:
                raise ValueError("BTC regime filter requires BTCUSDT")
            active = active.mul(momentum.BTCUSDT.gt(0), axis=0)
        score = active.astype(float).div(volatility).fillna(0.)
        if self.variant == "capped_allocation":
            values = np.array([_redistribute_capped(row, self.max_exposure, self.asset_cap)
                               for row in score.to_numpy()])
            return pd.DataFrame(values, index=close.index, columns=close.columns)
        return (score.div(score.sum(axis=1).replace(0, np.nan), axis=0)
                .mul(self.max_exposure).fillna(0.).clip(lower=0., upper=self.asset_cap))


def momentum_refinement_targets(markets):
    """Publish each completed-day target with the final 4h candle (20h label)."""
    _, daily = daily_control_targets(markets)
    close = pd.DataFrame({s: f.close for s, f in daily.items()})
    index = next(iter(markets.values())).index
    result = {}
    for name in MOMENTUM_VARIANTS:
        target = MomentumRefinement(name).weights(close)
        target.index = target.index + pd.Timedelta("20h")
        result[name] = target.reindex(index, method="ffill").fillna(0.)
    return result, daily
