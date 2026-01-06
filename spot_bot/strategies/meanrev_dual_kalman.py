from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from spot_bot.utils.normalization import clip01
from theta_bot_averaging.filters.dual_kalman import (
    AdaptiveLevelTrendKalman,
    RegimeKalman1D,
    circle_dist,
    robust_zscore,
)

from .base import Intent, Strategy


MIN_VARIANCE = 1e-12


@dataclass
class DualKalmanParams:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    q_r: float = 1e-3
    r_z: float = 1e-2
    q_level: float = 1e-4
    q_slope: float = 1e-6
    r: float = 1e-3
    s_min: float = 0.3
    s_max: float = 5.0
    k_u: float = 1.0
    sigma_window: int = 32
    emax: float = 1.0


class MeanRevDualKalmanStrategy(Strategy):
    """
    Mean-reversion intent using dual Kalman filters (regime + adaptive level/trend).
    """

    def __init__(self, **kwargs) -> None:
        params = DualKalmanParams(**kwargs)
        self.params = params

    def _build_regime_signal(self, features_df: pd.DataFrame) -> pd.Series:
        c = pd.to_numeric(features_df.get("C"), errors="coerce")
        psi = pd.to_numeric(features_df.get("psi"), errors="coerce")
        rv = pd.to_numeric(features_df.get("rv"), errors="coerce")

        rv_norm = robust_zscore(rv, window=self.params.sigma_window)
        conc_term = 1.0 - c

        psi_prev = psi.shift(1)
        psi_dist = psi.combine(psi_prev, circle_dist)
        psi_dist = psi_dist.clip(lower=0.0, upper=0.5)

        z = (
            self.params.alpha * conc_term.fillna(0.0)
            + self.params.beta * psi_dist.fillna(0.0)
            + self.params.gamma * rv_norm.fillna(0.0)
        )
        return z

    def _run_filters(self, close: pd.Series, z: pd.Series) -> tuple[float, float, float, float, float]:
        regime = RegimeKalman1D(q=self.params.q_r, r=self.params.r_z, mean=0.0, var=1.0)
        scale_vals = []
        for val in z:
            r_hat = regime.step(val)
            scale = float(np.clip(np.exp(r_hat), self.params.s_min, self.params.s_max))
            scale_vals.append(scale)

        price_series = close.astype(float)
        main = AdaptiveLevelTrendKalman(
            q_level=self.params.q_level,
            q_slope=self.params.q_slope,
            r=self.params.r,
            level=float(price_series.iloc[0]) if not price_series.empty else None,
        )

        residual = 0.0
        sigma2 = self.params.r
        level = float(price_series.iloc[0]) if not price_series.empty else 0.0
        slope = 0.0
        for y, scale in zip(price_series, scale_vals):
            level, slope, residual, sigma2 = main.step(y, scale=scale)
        sigma = float(np.sqrt(max(sigma2, MIN_VARIANCE)))
        last_scale = float(scale_vals[-1]) if scale_vals else 1.0
        return level, slope, residual, sigma, last_scale

    def _target_exposure(self, u_t: float, risk_budget: float, *, apply_budget: bool = True) -> float:
        raw = float(np.clip(-self.params.k_u * u_t, -self.params.emax, self.params.emax))
        # Long-only clamp for compatibility with existing sizing.
        target = float(np.clip(raw, 0.0, self.params.emax))
        if apply_budget:
            target *= clip01(risk_budget)
        return float(np.clip(target, 0.0, self.params.emax))

    def generate_intent(self, features_df: pd.DataFrame) -> Intent:
        if features_df is None or features_df.empty:
            return Intent(desired_exposure=0.0, reason="No features available", diagnostics={})

        close = pd.to_numeric(features_df.get("close"), errors="coerce").dropna()
        if close.empty:
            return Intent(desired_exposure=0.0, reason="No price data", diagnostics={})

        z = self._build_regime_signal(features_df.loc[close.index])
        level, slope, residual, sigma, last_scale = self._run_filters(close, z)

        if sigma <= 0.0 or np.isnan(sigma):
            sigma = np.sqrt(MIN_VARIANCE)
        u_t = residual / sigma
        risk_budget = float(features_df.iloc[-1].get("risk_budget", 1.0))
        desired_exposure = self._target_exposure(u_t, risk_budget=risk_budget, apply_budget=True)

        diagnostics = {
            "level": level,
            "slope": slope,
            "residual": residual,
            "sigma": sigma,
            "u_t": u_t,
            "risk_budget": risk_budget,
            "desired_raw": float(np.clip(-self.params.k_u * u_t, -self.params.emax, self.params.emax)),
            "scale": last_scale,
        }
        return Intent(desired_exposure=desired_exposure, reason="Dual Kalman mean reversion", diagnostics=diagnostics)

    def generate_series(
        self, features_df: pd.DataFrame, risk_budgets: Optional[pd.Series] = None, apply_budget: bool = True
    ) -> pd.Series:
        if features_df is None or features_df.empty:
            return pd.Series(dtype=float)

        close = pd.to_numeric(features_df.get("close"), errors="coerce").dropna()
        if close.empty:
            return pd.Series(dtype=float, index=features_df.index)

        z = self._build_regime_signal(features_df.loc[close.index])
        budgets = risk_budgets
        if budgets is None:
            budgets = (
                pd.to_numeric(features_df.get("risk_budget"), errors="coerce").reindex(close.index).fillna(1.0)
                if "risk_budget" in features_df
                else pd.Series(1.0, index=close.index)
            )
        budgets = budgets.reindex(close.index).fillna(1.0)

        regime = RegimeKalman1D(q=self.params.q_r, r=self.params.r_z, mean=0.0, var=1.0)
        main = AdaptiveLevelTrendKalman(
            q_level=self.params.q_level,
            q_slope=self.params.q_slope,
            r=self.params.r,
            level=float(close.iloc[0]),
        )

        exposures = []
        sigma2 = self.params.r
        for price, z_val, budget in zip(close, z, budgets):
            r_hat = regime.step(z_val)
            scale = float(np.clip(np.exp(r_hat), self.params.s_min, self.params.s_max))
            _, _, residual, sigma2 = main.step(price, scale=scale)
            sigma = float(np.sqrt(max(sigma2, MIN_VARIANCE)))
            u_t = residual / sigma if sigma > 0 else 0.0
            exposures.append(self._target_exposure(u_t, float(budget), apply_budget=apply_budget))
        return pd.Series(exposures, index=close.index, dtype=float)

    # Compatibility wrapper for backtests using StrategyOutput-like interface
    def generate(self, prices: pd.Series, features_df: Optional[pd.DataFrame] = None):
        feats = features_df
        if feats is None:
            feats = pd.DataFrame({"close": prices})
        intent = self.generate_intent(feats)

        class _Out:
            def __init__(self, exposure: float, diag: dict) -> None:
                self.desired_exposure = exposure
                self.diagnostics = diag

        return _Out(intent.desired_exposure, intent.diagnostics)
