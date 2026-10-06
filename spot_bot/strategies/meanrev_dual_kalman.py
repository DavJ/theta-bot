from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, NamedTuple, Optional

import numpy as np
import pandas as pd

from spot_bot.strategies.risk import StrategyOutput
from theta_bot_averaging.filters.dual_kalman import (
    MIN_VARIANCE,
    AdaptiveLevelTrendKalman,
    RegimeKalman1D,
    circle_dist,
    robust_zscore,
)

from .base import Intent, Strategy


class FilterOutput(NamedTuple):
    level: float
    slope: float
    residual: float
    sigma: float
    scale: float
    innovation_var: float


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
    r_max: float = 8.0  # Maximum allowed value for r_hat to prevent exp overflow
    conf_floor: float = 0.05  # Minimum confidence value
    conf_power: float = 1.0  # Power to apply to confidence when scaling risk budget
    snr_s0: float = 0.02  # SNR normalization constant (default 0.02, range 0.01-0.05 for crypto)
    snr_enabled: bool = False  # Enable SNR-based confidence component
    price_space: str = "dollars"  # Legacy control, or scale-invariant log_vol research mode
    volatility_window: int = 120
    volatility_floor: float = 0.0001  # One basis point in hourly log-return units


class MeanRevDualKalmanStrategy(Strategy):
    """
    Mean-reversion intent using dual Kalman filters (regime + adaptive level/trend).
    """

    def __init__(self, **kwargs) -> None:
        params = DualKalmanParams(**kwargs)
        if params.price_space not in {"dollars", "log_vol"}:
            raise ValueError("price_space must be dollars or log_vol")
        if params.volatility_window < 2 or not np.isfinite(params.volatility_floor) or params.volatility_floor <= 0:
            raise ValueError("Volatility window must be >= 2 and floor must be finite and positive")
        if params.r <= 0 or params.q_level < 0 or params.q_slope < 0:
            raise ValueError("Filter covariance must be positive and process noise nonnegative")
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

    def _filter_steps(self, close: pd.Series, z: pd.Series) -> Iterator[FilterOutput]:
        """Use identical filtering in single-intent and vectorized paths.

        log_vol uses dimensionless prices and trailing return variance. The
        variance is lagged: today's surprise cannot normalize itself away.
        The dollar mode is retained as a reproducible control.
        """
        prices = close.astype(float)
        variances = None
        if self.params.price_space == "log_vol":
            if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any():
                raise ValueError("log_vol requires finite positive close prices")
            prices = np.log(prices) - np.log(prices.iloc[0])
            floor = self.params.volatility_floor ** 2
            variances = (
                prices.diff().rolling(self.params.volatility_window, min_periods=2)
                .var(ddof=0).shift(1).fillna(floor).clip(lower=floor)
            )

        regime = RegimeKalman1D(q=self.params.q_r, r=self.params.r_z, mean=0.0, var=1.0)
        main = AdaptiveLevelTrendKalman(
            q_level=self.params.q_level,
            q_slope=self.params.q_slope,
            r=self.params.r,
            level=float(prices.iloc[0]) if not prices.empty else None,
        )
        if variances is not None:
            main.P = np.eye(2, dtype=float) * float(variances.iloc[0])

        for i, (price, z_val) in enumerate(zip(prices, z)):
            r_hat = np.clip(regime.step(z_val), -self.params.r_max, self.params.r_max)
            scale = float(np.clip(np.exp(r_hat), self.params.s_min, self.params.s_max))
            if variances is not None:
                variance = float(variances.iloc[i])
                main.r = variance
                main.q_level = (self.params.q_level / self.params.r) * variance
                main.q_slope = (self.params.q_slope / self.params.r) * variance
            level, slope, residual, sigma2 = main.step(price, scale=scale)
            yield FilterOutput(level, slope, residual, float(np.sqrt(max(sigma2, MIN_VARIANCE))), scale, sigma2)

    def _run_filters(self, close: pd.Series, z: pd.Series) -> FilterOutput:
        last = FilterOutput(0.0, 0.0, 0.0, np.sqrt(self.params.r), 1.0, self.params.r)
        for last in self._filter_steps(close, z):
            pass
        return last

    def _raw_signal(self, u_t: float) -> float:
        return -self.params.k_u * u_t

    def _compute_snr_confidence(self, slope: float, price: float, rv: float) -> tuple[float, float]:
        """
        Compute SNR-based confidence component.
        
        Args:
            slope: Kalman filter slope estimate
            price: Current price (for normalization)
            rv: Realized volatility
            
        Returns:
            (snr_raw, snr_conf) tuple
        """
        eps = 1e-12  # Small constant for numerical stability
        
        # Normalize slope to returns units if needed
        # In log space the slope is already a relative return.
        slope_rel = slope if self.params.price_space == "log_vol" else slope / max(price, eps)
        
        # SNR = signal strength / noise strength
        snr_raw = abs(slope_rel) / (rv + eps)
        
        # Convert to [0, 1] confidence
        snr_conf = snr_raw / (snr_raw + self.params.snr_s0)
        
        return float(snr_raw), float(snr_conf)

    def _target_exposure(self, u_t: float, risk_budget: float, *, apply_budget: bool = True) -> float:
        raw = self._raw_signal(u_t)
        target = float(np.clip(raw, -self.params.emax, self.params.emax))
        if apply_budget:
            budget = float(risk_budget)
            if not np.isfinite(budget):
                budget = 1.0
            budget = float(np.clip(budget, 0.0, 1.0))
            target *= budget
        return float(target)

    def generate_intent(self, features_df: pd.DataFrame) -> Intent:
        if features_df is None or features_df.empty:
            return Intent(desired_exposure=0.0, reason="No features available", diagnostics={})

        close = pd.to_numeric(features_df.get("close"), errors="coerce").dropna()
        if close.empty:
            return Intent(desired_exposure=0.0, reason="No price data", diagnostics={})

        z = self._build_regime_signal(features_df.loc[close.index]).reindex(close.index).fillna(0.0)
        filter_out = self._run_filters(close, z)
        level, slope, residual, sigma, last_scale, innovation_var = filter_out

        if sigma <= 0.0 or np.isnan(sigma):
            sigma = np.sqrt(MIN_VARIANCE)
        u_t = residual / sigma

        # Compute NIS-based confidence
        nis = (residual * residual) / max(innovation_var, MIN_VARIANCE)
        conf_nis = float(np.exp(-0.5 * nis))
        conf_nis = float(np.clip(conf_nis, self.params.conf_floor, 1.0))

        # Compute SNR-based confidence if enabled
        snr_raw = 0.0
        snr_conf = 1.0
        if self.params.snr_enabled:
            # Get RV from features
            last_row = features_df.iloc[-1]
            rv = float(last_row.get("rv", 0.02)) if isinstance(last_row, pd.Series) else 0.02
            price = float(close.iloc[-1])
            snr_raw, snr_conf = self._compute_snr_confidence(slope, price, rv)
        
        # Combine confidences
        conf_eff = conf_nis * snr_conf

        last_row = features_df.iloc[-1]
        if isinstance(last_row, pd.Series) and "risk_budget" in last_row:
            risk_budget = float(last_row.get("risk_budget", 1.0))
        else:
            risk_budget = 1.0

        # Apply effective confidence to risk budget
        risk_budget_eff = risk_budget * (conf_eff ** self.params.conf_power)
        desired_exposure = self._target_exposure(u_t, risk_budget=risk_budget_eff, apply_budget=True)

        raw_signal = self._raw_signal(u_t)
        diagnostics = {
            "level": level,
            "slope": slope,
            "residual": residual,
            "sigma": sigma,
            "u_t": u_t,
            "risk_budget": risk_budget,
            "desired_raw": float(np.clip(raw_signal, -self.params.emax, self.params.emax)),
            "scale": last_scale,
            "innovation_var": innovation_var,
            "nis": nis,
            "confidence": conf_nis,
            "snr_raw": snr_raw,
            "snr_conf": snr_conf,
            "conf_eff": conf_eff,
            "price_space": self.params.price_space,
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

        z = self._build_regime_signal(features_df.loc[close.index]).reindex(close.index).fillna(0.0)
        budgets = risk_budgets
        if budgets is None:
            budgets = (
                pd.to_numeric(features_df.get("risk_budget"), errors="coerce").reindex(close.index).fillna(1.0)
                if "risk_budget" in features_df
                else pd.Series(1.0, index=close.index)
            )
        budgets = budgets.reindex(close.index).fillna(1.0)
        
        # Get RV series if SNR is enabled
        rv_series = None
        if self.params.snr_enabled and "rv" in features_df:
            rv_series = pd.to_numeric(features_df.get("rv"), errors="coerce").reindex(close.index).fillna(0.02)

        exposures = []
        for i, (price, budget, filtered) in enumerate(zip(close, budgets, self._filter_steps(close, z))):
            level, slope, residual, sigma, scale, sigma2 = filtered
            u_t = residual / sigma if sigma > 0 else 0.0
            
            # Compute NIS-based confidence per bar
            nis = (residual * residual) / max(sigma2, MIN_VARIANCE)
            conf_nis = float(np.exp(-0.5 * nis))
            conf_nis = float(np.clip(conf_nis, self.params.conf_floor, 1.0))
            
            # Compute SNR-based confidence if enabled
            snr_conf = 1.0
            if self.params.snr_enabled and rv_series is not None:
                rv = float(rv_series.iloc[i])
                _, snr_conf = self._compute_snr_confidence(slope, price, rv)
            
            # Combine confidences
            conf_eff = conf_nis * snr_conf
            
            # Apply effective confidence to budget
            budget_eff = float(budget) * (conf_eff ** self.params.conf_power)
            exposures.append(self._target_exposure(u_t, budget_eff, apply_budget=apply_budget))
        return pd.Series(exposures, index=close.index, dtype=float)

    # Compatibility wrapper for backtests returning StrategyOutput
    def generate(self, prices: pd.Series, features_df: Optional[pd.DataFrame] = None):
        feats = features_df
        if feats is None:
            feats = pd.DataFrame({"close": prices})
        intent = self.generate_intent(feats)

        return StrategyOutput(desired_exposure=intent.desired_exposure, diagnostics=intent.diagnostics)
