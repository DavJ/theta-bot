"""Fixed daily momentum horizons, weekly ranks and actual-buy permission signals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spot_bot.portfolio.flow import daily_control_targets
from spot_bot.portfolio.momentum_refinement import MomentumRefinement, _redistribute_capped

MULTISCALE_VARIANTS = ("spot_control", "capped_control", "consensus", "dual_horizon",
                       "ranked_m30", "ranked_consensus", "capped_no_chase", "consensus_no_chase")


def _weekly_top_two(score, eligible):
    """Choose at completed Sunday closes; lost eligibility exits without replacing."""
    selected = score.where(eligible, -np.inf).rank(axis=1, ascending=False, method="first").le(2) & eligible
    sundays = pd.Series(score.index.dayofweek == 6, index=score.index)
    members = selected.astype(float).where(sundays, np.nan, axis=0).ffill().fillna(0.).gt(0)
    return members & eligible


@dataclass(frozen=True)
class MultiscaleSpot:
    """Offline policy; all inputs are completed daily prices, with no exchange access."""
    variant: str
    max_exposure: float = .5
    asset_cap: float = .25
    rebalance_weekday: int = 0
    rebalance_band: float = .01

    def __post_init__(self):
        if self.variant not in MULTISCALE_VARIANTS:
            raise ValueError("Unknown fixed multiscale variant")
        if (not np.isfinite([self.max_exposure, self.asset_cap, self.rebalance_band]).all()
                or not 0 < self.asset_cap <= self.max_exposure <= 1
                or not 0 <= self.rebalance_band <= self.max_exposure or self.rebalance_weekday != 0):
            raise ValueError("Invalid owned-funds spot caps or fixed Monday schedule")

    def weights(self, close):
        original = MomentumRefinement("spot_control", self.max_exposure, self.asset_cap).weights(close)
        if self.variant == "spot_control":
            return original
        if self.variant in ("capped_control", "capped_no_chase"):
            return MomentumRefinement("capped_allocation", self.max_exposure, self.asset_cap).weights(close)
        volatility = close.pct_change(fill_method=None).rolling(60, min_periods=60).std(ddof=0).clip(lower=.0001)
        momentum = {h: close.pct_change(h, fill_method=None) for h in (14, 30, 60, 90)}
        consensus = (sum(momentum[h].gt(0).astype(int) for h in (14, 30, 60)) >= 2) & volatility.notna()
        if self.variant == "dual_horizon":
            active = momentum[30].gt(0) & momentum[90].gt(0) & volatility.notna()
        elif self.variant == "ranked_m30":
            active = momentum[30].gt(0) & volatility.notna()
        else:
            active = consensus
        if self.variant.startswith("ranked_"):
            if self.variant == "ranked_m30":
                strength = momentum[30].div(volatility * np.sqrt(30))
            else:
                z = np.stack([momentum[h].div(volatility * np.sqrt(h)).to_numpy()
                              for h in (14, 30, 60)], axis=-1)
                strength = pd.DataFrame(np.median(z, axis=-1), index=close.index, columns=close.columns)
            membership = _weekly_top_two(strength, active)
            return membership.astype(float) * min(self.asset_cap, self.max_exposure / 2)
        score = active.astype(float).div(volatility).fillna(0.)
        values = np.array([_redistribute_capped(row, self.max_exposure, self.asset_cap)
                           for row in score.to_numpy()])
        return pd.DataFrame(values, index=close.index, columns=close.columns)

    def buy_permissions(self, close):
        if self.variant not in ("capped_no_chase", "consensus_no_chase"):
            return None
        # Reuse the input/grid validation even when this method is called directly.
        MomentumRefinement("spot_control", self.max_exposure, self.asset_cap).weights(close)
        volatility = close.pct_change(fill_method=None).rolling(60, min_periods=60).std(ddof=0).clip(lower=.0001)
        return close.pct_change(7, fill_method=None).le(2 * volatility * np.sqrt(7)) & volatility.notna()


def multiscale_targets(markets):
    """Return targets and buy masks published at the completed final 4h daily close."""
    _, daily = daily_control_targets(markets)
    close = pd.DataFrame({s: f.close for s, f in daily.items()})
    index = next(iter(markets.values())).index
    targets, masks = {}, {}
    for name in MULTISCALE_VARIANTS:
        policy = MultiscaleSpot(name)
        target, mask = policy.weights(close), policy.buy_permissions(close)
        target.index = target.index + pd.Timedelta("20h")
        targets[name] = target.reindex(index, method="ffill", fill_value=0.)
        if mask is not None:
            mask.index = mask.index + pd.Timedelta("20h")
            masks[name] = mask.reindex(index, method="ffill", fill_value=False)
        else:
            masks[name] = None
    return targets, masks, daily
