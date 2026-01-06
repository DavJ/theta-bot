from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd


MIN_VARIANCE = 1e-12


def circle_dist(a: float, b: float) -> float:
    """
    Circular distance on [0, 1).

    Returns NaN if either input is NaN.
    """
    if pd.isna(a) or pd.isna(b):
        return np.nan
    diff = abs(a - b)
    wrapped = diff % 1.0
    return float(min(wrapped, 1.0 - wrapped))


def robust_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Robust z-score using rolling median and MAD (scaled by 1.4826).
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    window = max(int(window), 1)
    med = series.rolling(window=window, min_periods=1).median()
    mad = (series - med).abs().rolling(window=window, min_periods=1).median()
    denom = mad * 1.4826
    denom = denom.where(denom > MIN_VARIANCE, MIN_VARIANCE)
    z = (series - med) / denom
    return z.fillna(0.0)


@dataclass
class RegimeKalman1D:
    q: float
    r: float
    mean: float = 0.0
    var: float = 1.0

    def step(self, z: Optional[float]) -> float:
        """
        One-dimensional Kalman update r_{t+1} = r_t + noise, z_t = r_t + noise.
        """
        self.var = float(self.var + self.q)
        if z is None or pd.isna(z):
            return float(self.mean)
        innovation = float(z) - float(self.mean)
        S = self.var + self.r
        if np.isnan(S) or S <= 0.0:
            S = MIN_VARIANCE
        K = self.var / S
        self.mean = float(self.mean + K * innovation)
        self.var = float((1.0 - K) * self.var)
        return float(self.mean)


@dataclass
class AdaptiveLevelTrendKalman:
    q_level: float
    q_slope: float
    r: float
    level: Optional[float] = None
    slope: float = 0.0
    P: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=float))

    def step(self, y: float, scale: float = 1.0) -> Tuple[float, float, float, float]:
        """
        Local level + trend Kalman step with adaptive process noise scale.

        Returns:
            tuple: (level estimate, slope estimate, innovation, innovation variance)
        """
        if pd.isna(y):
            return float(self.level or 0.0), float(self.slope), 0.0, float(self.r)

        if self.level is None:
            self.level = float(y)
            self.slope = 0.0
            self.P = np.eye(2, dtype=float)

        F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        H = np.array([1.0, 0.0], dtype=float)
        q_scale = max(float(scale), 0.0)
        Q = np.diag([self.q_level * q_scale, self.q_slope * q_scale]).astype(float)

        x = np.array([self.level, self.slope], dtype=float)
        x_pred = F @ x
        P_pred = F @ self.P @ F.T + Q

        innovation = float(y) - float(x_pred[0])
        S = float(P_pred[0, 0] + self.r)
        if np.isnan(S) or S <= 0.0:
            S = MIN_VARIANCE
        K = (P_pred @ H) / S
        x_upd = x_pred + K * innovation
        P_upd = (np.eye(2) - np.outer(K, H)) @ P_pred

        self.level = float(x_upd[0])
        self.slope = float(x_upd[1])
        self.P = P_upd
        return self.level, self.slope, innovation, S
