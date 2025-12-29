#!/usr/bin/env python3
"""Volatility model sigma(t)."""

from __future__ import annotations

import pandas as pd


def compute_sigma(
    returns: pd.Series,
    window: int = 168,
    ewma_lambda: float | None = None,
    epsilon: float = 1e-8,
) -> pd.Series:
    """Compute diffusion intensity using trailing volatility."""
    roll = returns.rolling(window=window, min_periods=max(2, window // 4)).std()
    if ewma_lambda is not None:
        alpha = 1 - ewma_lambda
        ewma = returns.ewm(alpha=alpha, adjust=False).std(bias=False)
        sigma = roll.fillna(ewma)
    else:
        sigma = roll
    sigma = sigma.fillna(0.0)
    sigma = sigma.where(sigma > epsilon, other=epsilon)
    return sigma
