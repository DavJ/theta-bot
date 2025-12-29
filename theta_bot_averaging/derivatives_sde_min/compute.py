from __future__ import annotations

import numpy as np
import pandas as pd


def _min_periods(window: int) -> int:
    return max(5, window // 4)


def rolling_z(series: pd.Series, window: int, clip: float = 10.0) -> pd.Series:
    min_periods = _min_periods(window)
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std().clip(lower=1e-12)
    z = (series - mean) / std
    return z.clip(-clip, clip)


def compute_mu_sigma_lambda(
    panel: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 0.5,
    z_window: int = 168,
    sigma_window: int = 168,
    epsilon: float = 1e-8,
    q: float = 0.85,
) -> pd.DataFrame:
    """
    Compute drift (mu), volatility (sigma), and Lambda scores for the derivatives SDE model.
    mu = -alpha * z(doi) * z(funding) + beta * z(doi) * z(basis)
    sigma = rolling std of spot returns over sigma_window (with floor epsilon)
    Lambda = |mu| / sigma and active flag is set by quantile q.
    """
    df = panel.copy()
    df["doi"] = np.log(df["open_interest"]).diff()
    df["z_doi"] = rolling_z(df["doi"], window=z_window)
    df["z_funding"] = rolling_z(df["funding_rate"], window=z_window)
    df["z_basis"] = rolling_z(df["basis"], window=z_window)

    df["mu"] = -alpha * df["z_doi"] * df["z_funding"] + beta * df["z_doi"] * df["z_basis"]
    df["sigma"] = df["returns"].rolling(window=sigma_window, min_periods=_min_periods(sigma_window)).std().clip(
        lower=epsilon
    )
    df["lambda"] = (df["mu"].abs() / df["sigma"]).replace([np.inf, -np.inf], np.nan)

    lambda_clean = df["lambda"].dropna()
    lambda_threshold = lambda_clean.quantile(q) if len(lambda_clean) else np.nan
    df["lambda_threshold"] = lambda_threshold
    if np.isnan(lambda_threshold):
        df["active"] = False
    else:
        df["active"] = df["lambda"] >= lambda_threshold
    return df
