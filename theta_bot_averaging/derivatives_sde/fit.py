#!/usr/bin/env python3
"""Parameter fitting for drift model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _design_matrix(state_df: pd.DataFrame) -> pd.DataFrame:
    """Base features for regression."""
    x1 = -1.0 * state_df["z_oi_change"] * state_df["z_funding"]
    x2 = state_df["z_oi_change"] * state_df["z_basis"]
    x3 = state_df["z_basis"]
    return pd.DataFrame({"mu1_base": x1, "mu2_base": x2, "mu3_base": x3}, index=state_df.index)


def fit_linear_drift(
    state_df: pd.DataFrame,
    lambda_threshold: float | None = None,
    min_samples: int = 5,
) -> dict:
    """Fit alpha, beta, gamma by regressing r on base components."""
    X = _design_matrix(state_df)
    y = state_df["r"]

    mask = X.notna().all(axis=1) & y.notna()
    if lambda_threshold is not None and "Lambda" in state_df.columns:
        mask &= state_df["Lambda"] > lambda_threshold

    if mask.sum() < min_samples:
        return {"alpha": np.nan, "beta": np.nan, "gamma": np.nan, "intercept": np.nan}

    model = LinearRegression(fit_intercept=True)
    model.fit(X[mask], y[mask])
    return {
        "alpha": model.coef_[0],
        "beta": model.coef_[1],
        "gamma": model.coef_[2],
        "intercept": model.intercept_,
    }


def walk_forward_fit(
    state_df: pd.DataFrame,
    n_folds: int = 3,
    lambda_threshold: float | None = None,
    min_samples: int = 5,
) -> dict:
    """Expanding-window walk-forward regression."""
    X = _design_matrix(state_df)
    y = state_df["r"]
    mask = X.notna().all(axis=1) & y.notna()
    if lambda_threshold is not None and "Lambda" in state_df.columns:
        mask &= state_df["Lambda"] > lambda_threshold

    df = pd.concat([X, y], axis=1)
    df = df[mask]
    # Require at least one observation per fold plus a small buffer to start the expansion.
    if len(df) < n_folds + 2:
        return {"folds": [], "summary": {}}

    fold_size = max(1, len(df) // n_folds)
    folds = []
    for k in range(1, n_folds + 1):
        train_end = min(len(df), k * fold_size)
        train = df.iloc[:train_end]
        if len(train) < min_samples:
            continue
        model = LinearRegression(fit_intercept=True)
        model.fit(train[["mu1_base", "mu2_base", "mu3_base"]], train["r"])
        folds.append(
            {
                "fold": k,
                "train_size": len(train),
                "alpha": model.coef_[0],
                "beta": model.coef_[1],
                "gamma": model.coef_[2],
                "intercept": model.intercept_,
            }
        )

    summary = {}
    if folds:
        summary = {
            "alpha_mean": np.mean([f["alpha"] for f in folds]),
            "alpha_std": np.std([f["alpha"] for f in folds]),
            "beta_mean": np.mean([f["beta"] for f in folds]),
            "beta_std": np.std([f["beta"] for f in folds]),
            "gamma_mean": np.mean([f["gamma"] for f in folds]),
            "gamma_std": np.std([f["gamma"] for f in folds]),
        }

    return {"folds": folds, "summary": summary}
