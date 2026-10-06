"""Causal exposure budgets for offline, unlevered daily portfolio research."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioRisk:
    volatility_target: float | None = None
    volatility_window: int = 60
    covariance_shrinkage: float = 0.25
    drawdown_limit: float | None = None
    drawdown_buffer: float = 0.02
    cushion_multiplier: float = 5.0

    def __post_init__(self):
        if (not isinstance(self.volatility_window, int) or self.volatility_window < 2
                or not np.isfinite([self.covariance_shrinkage, self.drawdown_buffer,
                                    self.cushion_multiplier]).all()
                or not 0 <= self.covariance_shrinkage <= 1
                or not 0 <= self.drawdown_buffer < 1 or self.cushion_multiplier <= 0):
            raise ValueError("Invalid covariance or cushion parameters")
        if self.volatility_target is not None and (
                not np.isfinite(self.volatility_target) or self.volatility_target <= 0):
            raise ValueError("Volatility target must be positive and finite")
        if self.drawdown_limit is not None and (
                not np.isfinite(self.drawdown_limit)
                or not self.drawdown_buffer < self.drawdown_limit < 1):
            raise ValueError("Drawdown limit must exceed the reserve and be below one")

    def covariances(self, close: pd.DataFrame) -> np.ndarray:
        """Completed-close covariance matrices; caller must use the PRIOR day."""
        returns = close.pct_change(fill_method=None)
        size = close.shape[1]
        covariance = returns.rolling(self.volatility_window,
                                     min_periods=self.volatility_window).cov(ddof=0)
        values = covariance.to_numpy().reshape(len(close), size, size)
        diagonal = values * np.eye(size)
        return ((1 - self.covariance_shrinkage) * values
                + self.covariance_shrinkage * diagonal)

    def constrain(self, weights: pd.Series, covariance: np.ndarray | None,
                  nav: float, peak: float) -> tuple[pd.Series, dict]:
        """Reduce proposed weights using known NAV and an unreplenished peak."""
        if (not np.isfinite([nav, peak]).all() or nav <= 0 or peak < nav - 1e-8
                or not np.isfinite(weights.to_numpy()).all() or (weights < 0).any()):
            raise ValueError("Positive NAV, historical peak and nonnegative weights required")
        exposure = float(weights.sum())
        floor = (peak * (1 - self.drawdown_limit + self.drawdown_buffer)
                 if self.drawdown_limit is not None else 0.0)
        budget = (min(1.0, self.cushion_multiplier * max(0.0, nav - floor) / nav)
                  if self.drawdown_limit is not None else 1.0)
        annual_vol = None
        scale = min(1.0, budget / exposure) if exposure > 0 else 1.0
        if self.volatility_target is not None:
            if covariance is None or not np.isfinite(covariance).all():
                scale = 0.0
            else:
                w = weights.to_numpy()
                annual_vol = float(np.sqrt(max(0.0, w @ covariance @ w) * 365))
                if annual_vol > 0:
                    scale = min(scale, self.volatility_target / annual_vol)
        return weights * scale, {"risk_scale": scale, "risk_floor": floor,
                                 "risk_budget_exposure": budget,
                                 "estimated_annual_volatility": annual_vol}
