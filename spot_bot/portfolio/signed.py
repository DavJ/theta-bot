"""Completed-close long/short allocations for offline derivatives research."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from spot_bot.portfolio.trend import TrendPortfolio

SIGNED_APPROACHES = ("signed_m30", "signed_horizons", "ema", "breakout", "relative", "pairs", "blend")


def _caps(weights, gross=1., asset=.5):
    maximum = weights.abs().max(axis=1)
    total = weights.abs().sum(axis=1)
    scale = pd.concat([pd.Series(1., index=weights.index), gross / total.replace(0, np.nan),
                       asset / maximum.replace(0, np.nan)], axis=1).min(axis=1)
    return weights.mul(scale, axis=0).fillna(0.)


@dataclass(frozen=True)
class SignedPortfolio:
    approach: str

    def __post_init__(self):
        if self.approach not in (*SIGNED_APPROACHES, "perp_long_control"):
            raise ValueError("Unknown signed research approach")

    def weights(self, close):
        if (close.empty or not close.index.is_unique or not close.index.is_monotonic_increasing
                or not isinstance(close.index, pd.DatetimeIndex) or close.index.tz is None
                or not close.columns.is_unique or not np.isfinite(close.to_numpy()).all()
                or (close <= 0).any().any()):
            raise ValueError("Ordered positive completed prices required")
        if self.approach == "perp_long_control":
            return TrendPortfolio("momentum_vol", max_exposure=.5, asset_cap=.25).weights(close)
        vol = close.pct_change(fill_method=None).rolling(60, min_periods=60).std(ddof=0).clip(lower=.0001)
        inv = 1 / vol
        base = inv.div(inv.sum(axis=1), axis=0)
        if self.approach == "signed_m30":
            result = np.sign(close.pct_change(30, fill_method=None)) * base
        elif self.approach == "signed_horizons":
            returns = [close.pct_change(n, fill_method=None) for n in (7, 30, 90)]
            result = sum(np.sign(r) for r in returns) / 3 * base
        elif self.approach == "ema":
            result = np.sign(close.ewm(span=12, adjust=False, min_periods=48).mean()
                             - close.ewm(span=48, adjust=False, min_periods=48).mean()) * base
        elif self.approach == "breakout":
            upper, lower = close.rolling(20).max().shift(1), close.rolling(20).min().shift(1)
            exit_high, exit_low = close.rolling(10).max().shift(1), close.rolling(10).min().shift(1)
            result = close * 0.
            for s in close:
                state, states = 0, []
                for p, hi, lo, ehi, elo in zip(close[s], upper[s], lower[s], exit_high[s], exit_low[s]):
                    if p > hi:
                        state = 1
                    elif p < lo:
                        state = -1
                    elif (state == 1 and p < elo) or (state == -1 and p > ehi):
                        state = 0
                    states.append(state)
                result[s] = states
            result *= base
        elif self.approach == "relative":
            momentum = close.pct_change(30, fill_method=None)
            result = close * 0.
            for ts, row in momentum.iterrows():
                if row.notna().all() and row.max() > row.min():
                    result.loc[ts, row.idxmax()] = .5
                    result.loc[ts, row.idxmin()] = -.5
            result = result.where(vol.notna(), 0.)
        elif self.approach == "pairs":
            if not {"BTCUSDT", "ETHUSDT", "BNBUSDT"}.issubset(close):
                raise ValueError("Fixed pairs require BTC, ETH and BNB")
            x = np.log(close.BTCUSDT)
            xp, mx, vx = x.shift(1), x.shift(1).rolling(90).mean(), x.shift(1).rolling(90).var(ddof=0)
            result = close * 0.
            for s in ("ETHUSDT", "BNBUSDT"):
                y = np.log(close[s])
                yp = y.shift(1)
                my, vy = yp.rolling(90).mean(), yp.rolling(90).var(ddof=0)
                cov = xp.rolling(90).cov(yp, ddof=0)
                beta = (cov / vx.replace(0, np.nan)).clip(.25, 3.)
                sigma = (vy + beta.pow(2) * vx - 2 * beta * cov).clip(lower=1e-12).pow(.5)
                z = (y - (my - beta * mx) - beta * x) / sigma
                state, states = 0, []
                for value in z:
                    if not np.isfinite(value) or abs(value) >= 4:
                        state = 0
                    elif state and (abs(value) <= .5 or state * value >= 0):
                        state = 0
                    elif not state and abs(value) >= 2:
                        state = -int(np.sign(value))
                    states.append(state)
                w = pd.Series(states, index=close.index) / (1 + beta)
                result[s] += .5 * w.fillna(0.)
                result["BTCUSDT"] -= .5 * (beta * w).fillna(0.)
        else:
            result = sum(SignedPortfolio(s).weights(close)
                         for s in ("signed_horizons", "ema", "relative", "pairs")) / 4
        return _caps(result)
