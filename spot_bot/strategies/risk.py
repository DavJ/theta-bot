from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from spot_bot.utils.normalization import clip01

MIN_VARIANCE = 1e-8


def _hash_params(params: dict) -> str:
    key = "|".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


@dataclass
class StrategyOutput:
    desired_exposure: float
    diagnostics: dict


class MeanRevGatedStrategy:
    """
    Mean reversion intent using price vs EMA/std, mapped to signed exposure.
    """

    def __init__(
        self,
        ema_span: int = 20,
        std_lookback: int = 30,
        max_exposure: float = 1.0,
        entry_z: float = 0.5,
        full_z: float = 2.0,
    ) -> None:
        self.ema_span = int(ema_span)
        self.std_lookback = int(std_lookback)
        self.max_exposure = float(max_exposure)
        self.entry_z = float(entry_z)
        self.full_z = float(full_z)

    def _compute_z(self, prices: pd.Series) -> Tuple[float, float, float, float]:
        ema = prices.ewm(span=self.ema_span, adjust=False).mean()
        latest_price = float(prices.iloc[-1])
        latest_ema = float(ema.iloc[-1])
        recent = prices.tail(self.std_lookback)
        std = float(recent.std(ddof=0) if len(recent) > 1 else 1e-8)
        std = std if std > 0 else 1e-8
        z = (latest_price - latest_ema) / std
        return z, latest_price, latest_ema, std

    def generate(self, prices: pd.Series) -> StrategyOutput:
        prices = prices.dropna()
        if len(prices) < max(self.ema_span, self.std_lookback):
            return StrategyOutput(desired_exposure=0.0, diagnostics={"reason": "insufficient history"})
        z, latest_price, latest_ema, std = self._compute_z(prices)
        strength = -z
        if abs(strength) < self.entry_z:
            exposure = 0.0
            reason = "entry not met"
        else:
            clipped = np.clip(strength, -self.full_z, self.full_z)
            scale = clipped / max(self.full_z, 1e-8)
            exposure = float(np.clip(scale, -1.0, 1.0) * self.max_exposure)
            reason = "mean reversion signal"
        diag = {
            "z": z,
            "latest_price": latest_price,
            "latest_ema": latest_ema,
            "std": std,
            "reason": reason,
        }
        return StrategyOutput(desired_exposure=exposure, diagnostics=diag)


class KalmanRiskStrategy:
    """
    Kalman filter on log price; supports mean-reversion (residual) or trend (slope) modes.
    Exposure scaled by innovation variance and capped to max_exposure.
    """

    def __init__(
        self,
        mode: Literal["meanrev", "trend"] = "meanrev",
        q_level: float = 1e-4,
        q_trend: float = 1e-6,
        r: float = 1e-3,
        max_exposure: float = 1.0,
        min_bars: int = 10,
    ) -> None:
        self.mode = mode
        self.q_level = float(q_level)
        self.q_trend = float(q_trend)
        self.r = float(r)
        self.max_exposure = float(max_exposure)
        self.min_bars = int(min_bars)

    def _filter(self, prices: pd.Series) -> Tuple[np.ndarray, float]:
        level0 = float(prices.iloc[0])
        x = np.array([level0, 0.0], dtype=float)
        P = np.eye(2, dtype=float)
        F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        Q = np.array([[self.q_level, 0.0], [0.0, self.q_trend]], dtype=float)
        H = np.array([1.0, 0.0], dtype=float)
        innovation_var = float(self.r)
        for price in prices:
            y = float(price)
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            innovation = y - H @ x_pred
            innovation_var = float(H @ P_pred @ H.T + self.r)
            if np.isnan(innovation_var) or innovation_var <= 0.0:
                innovation_var = MIN_VARIANCE
            K = (P_pred @ H) / innovation_var
            x = x_pred + K * innovation
            P = (np.eye(2) - np.outer(K, H)) @ P_pred
        return x, innovation_var

    def generate(self, prices: pd.Series) -> StrategyOutput:
        prices = prices.dropna()
        if len(prices) < self.min_bars:
            return StrategyOutput(desired_exposure=0.0, diagnostics={"reason": "insufficient history"})
        if (prices <= 0).any():
            raise ValueError("Prices must be positive for Kalman filter.")
        log_prices = np.log(prices)
        state, innov_var = self._filter(log_prices)
        level, trend = float(state[0]), float(state[1])
        latest_lp = float(log_prices.iloc[-1])
        resid = latest_lp - level
        signal = -resid if self.mode == "meanrev" else trend
        scale = signal / max(np.sqrt(innov_var), np.sqrt(MIN_VARIANCE))
        exposure = float(np.clip(scale, -1.0, 1.0) * self.max_exposure)
        diag = {"level": level, "trend": trend, "resid": resid, "innovation_var": innov_var, "mode": self.mode}
        return StrategyOutput(desired_exposure=exposure, diagnostics=diag)


def apply_risk_gating(desired: float, risk_state: str, risk_budget: float) -> float:
    if risk_state == "OFF":
        return 0.0
    if risk_state in ("REDUCE", "ON"):
        return desired * clip01(risk_budget)
    return 0.0


def params_hash(params: dict) -> str:
    """Public helper to produce a short, deterministic hash for strategy params."""
    return _hash_params(params)
