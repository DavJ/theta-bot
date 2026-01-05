from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from spot_bot.utils.normalization import clip01

from .base import Intent, Strategy


@dataclass
class KalmanParams:
    q_level: float = 1e-4
    q_trend: float = 1e-6
    r: float = 1e-3
    k: float = 1.5
    min_bars: int = 10


class KalmanStrategy(Strategy):
    """
    Long/flat strategy using a local linear trend Kalman filter on prices.

    The state is [level, trend]. Exposure increases when price is below the
    filtered level (negative z) and decreases when price is above it.
    """

    def __init__(
        self,
        q_level: float = KalmanParams.q_level,
        q_trend: float = KalmanParams.q_trend,
        r: float = KalmanParams.r,
        k: float = KalmanParams.k,
        min_bars: int = KalmanParams.min_bars,
    ) -> None:
        self.params = KalmanParams(
            q_level=float(q_level),
            q_trend=float(q_trend),
            r=float(r),
            k=float(k),
            min_bars=int(min_bars),
        )

    def _extract_price(self, features_df: pd.DataFrame) -> pd.Series:
        for col in ("close", "Close", "price"):
            if col in features_df.columns:
                return features_df[col].astype(float)
        raise ValueError("features_df must contain a 'close' or 'price' column.")

    def _run_filter(self, prices: pd.Series) -> Tuple[np.ndarray, float]:
        """Return final state and innovation variance."""
        level0 = float(prices.iloc[0])
        x = np.array([level0, 0.0], dtype=float)
        P = np.eye(2, dtype=float)

        F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        Q = np.array([[self.params.q_level, 0.0], [0.0, self.params.q_trend]], dtype=float)
        H = np.array([1.0, 0.0], dtype=float)
        R = float(self.params.r)
        innovation_var = float(R)

        for price in prices:
            y = float(price)

            # Predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # Update
            innovation = y - H @ x_pred
            innovation_var = float(H @ P_pred @ H.T + R)
            if innovation_var <= 0.0 or np.isnan(innovation_var):
                innovation_var = 1e-8
            K = (P_pred @ H) / innovation_var
            x = x_pred + K * innovation
            P = (np.eye(2) - np.outer(K, H)) @ P_pred

        return x, innovation_var

    def generate_intent(self, features_df: pd.DataFrame) -> Intent:
        if features_df is None or features_df.empty:
            return Intent(desired_exposure=0.0, reason="No features available", diagnostics={})

        prices = self._extract_price(features_df).dropna()
        if prices.empty or len(prices) < self.params.min_bars:
            return Intent(desired_exposure=0.0, reason="Insufficient history", diagnostics={})

        state, innovation_var = self._run_filter(prices)
        level_est = float(state[0])
        latest_price = float(prices.iloc[-1])
        z = (latest_price - level_est) / float(np.sqrt(innovation_var) if innovation_var > 0 else 1e-8)

        exposure_raw = 1.0 / (1.0 + float(np.exp(self.params.k * z)))
        desired_exposure = clip01(exposure_raw)
        diagnostics = {
            "level": level_est,
            "trend": float(state[1]),
            "innovation_var": innovation_var,
            "z": z,
            "k": self.params.k,
        }

        reason = "Kalman long bias" if desired_exposure > 0 else "No signal"
        return Intent(desired_exposure=desired_exposure, reason=reason, diagnostics=diagnostics)
