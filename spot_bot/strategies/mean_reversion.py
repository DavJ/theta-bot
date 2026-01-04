from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from spot_bot.utils.normalization import clip01

from .base import Intent, Strategy


class MeanReversionStrategy(Strategy):
    """
    Simple long/flat mean-reversion intent generator.

    Uses z-score of price relative to an EMA to size long exposure.
    """

    def __init__(
        self,
        ema_span: int = 20,
        std_lookback: int = 30,
        entry_z: float = 0.5,
        full_z: float = 2.0,
        min_exposure: float = 0.2,
        max_exposure: float = 1.0,
    ) -> None:
        self.ema_span = int(ema_span)
        self.std_lookback = int(std_lookback)
        self.entry_z = float(entry_z)
        self.full_z = float(full_z)
        self.min_exposure = float(min_exposure)
        self.max_exposure = float(max_exposure)

    def _safe_std(self, prices: pd.Series) -> float:
        fallback_std = float(np.std(prices.values))
        return fallback_std if fallback_std > 0.0 else 1e-8

    def _extract_price(self, features_df: pd.DataFrame) -> pd.Series:
        for col in ("close", "Close", "price"):
            if col in features_df.columns:
                return features_df[col].astype(float)
        raise ValueError("features_df must contain a 'close' or 'price' column.")

    def _compute_zscore(self, prices: pd.Series) -> tuple[float, float, float]:
        ema = prices.ewm(span=self.ema_span, adjust=False).mean()
        latest_price = float(prices.iloc[-1])
        latest_ema = float(ema.iloc[-1])
        if len(prices) < 2:
            rolling_std = 1e-8
        else:
            recent_prices = prices.tail(self.std_lookback)
            rolling_std_value = recent_prices.std(ddof=0)
            if pd.isna(rolling_std_value) or rolling_std_value <= 0.0:
                rolling_std = self._safe_std(prices)
            else:
                rolling_std = float(rolling_std_value)
        if rolling_std <= 0.0:
            rolling_std = self._safe_std(prices)
        return (latest_price - latest_ema) / rolling_std, latest_price, latest_ema

    def generate_intent(self, features_df: pd.DataFrame) -> Intent:
        if features_df is None or features_df.empty:
            return Intent(desired_exposure=0.0, reason="No features available", diagnostics={})

        prices = self._extract_price(features_df).dropna()
        if prices.empty:
            return Intent(desired_exposure=0.0, reason="No price data", diagnostics={})

        zscore, latest_price, latest_ema = self._compute_zscore(prices)
        signal_strength = max(0.0, -zscore)  # long when price is below EMA

        if signal_strength <= self.entry_z:
            desired_exposure = 0.0
            reason = "No mean reversion signal"
        else:
            capped_strength = min(signal_strength, self.full_z)
            scale = (capped_strength - self.entry_z) / max(self.full_z - self.entry_z, 1e-8)
            raw_exposure = self.min_exposure + (self.max_exposure - self.min_exposure) * scale
            desired_exposure = clip01(raw_exposure)
            reason = "Mean reversion long bias"

        diagnostics = {
            "zscore": zscore,
            "signal_strength": signal_strength,
            "latest_price": latest_price,
            "latest_ema": latest_ema,
            "entry_z": self.entry_z,
            "full_z": self.full_z,
        }

        return Intent(desired_exposure=desired_exposure, reason=reason, diagnostics=diagnostics)
