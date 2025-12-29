from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_z(series: pd.Series, window: int, clip: float = 10.0) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(5, window // 4)).mean()
    std = series.rolling(window=window, min_periods=max(5, window // 4)).std().clip(lower=1e-12)
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
    df = panel.copy()
    df["doi"] = np.log(df["open_interest"]).diff()
    df["z_doi"] = rolling_z(df["doi"], window=z_window)
    df["z_funding"] = rolling_z(df["funding_rate"], window=z_window)
    df["z_basis"] = rolling_z(df["basis"], window=z_window)

    df["mu"] = -alpha * df["z_doi"] * df["z_funding"] + beta * df["z_doi"] * df["z_basis"]
    df["sigma"] = df["returns"].rolling(window=sigma_window, min_periods=max(5, sigma_window // 4)).std().clip(
        lower=epsilon
    )
    df["lambda"] = (df["mu"].abs() / df["sigma"]).replace([np.inf, -np.inf], np.nan)

    lambda_threshold = df["lambda"].quantile(q) if len(df["lambda"].dropna()) else np.nan
    df["lambda_threshold"] = lambda_threshold
    if np.isnan(lambda_threshold):
        df["active"] = False
    else:
        df["active"] = df["lambda"] >= lambda_threshold
    return df
