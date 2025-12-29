#!/usr/bin/env python3
"""Evaluation of deterministic bias conditioned on Lambda."""

from __future__ import annotations

import numpy as np
import pandas as pd


def future_return(returns: pd.Series, horizon: int) -> pd.Series:
    """Compute log price change over the next H hours."""
    return returns.shift(-1).rolling(window=horizon, min_periods=horizon).sum().shift(-(horizon - 1))


def _effect_metrics(mu: pd.Series, y: pd.Series) -> dict:
    valid = mu.notna() & y.notna()
    if valid.sum() == 0:
        return {"sign_agreement": np.nan, "cond_mean_up": np.nan, "cond_mean_down": np.nan, "effect_size": np.nan}

    mu_v = mu[valid]
    y_v = y[valid]
    sign_agreement = (np.sign(mu_v) == np.sign(y_v)).mean()
    up_slice = y_v[mu_v > 0]
    down_slice = y_v[mu_v < 0]
    cond_mean_up = up_slice.mean() if not up_slice.empty else np.nan
    cond_mean_down = down_slice.mean() if not down_slice.empty else np.nan
    std_all = y_v.std()
    valid_std = (
        std_all is not None
        and (not np.isnan(std_all))
        and std_all != 0
        and not np.isnan(cond_mean_up)
        and not np.isnan(cond_mean_down)
    )
    effect_size = (cond_mean_up - cond_mean_down) / std_all if valid_std else np.nan
    return {
        "sign_agreement": sign_agreement,
        "cond_mean_up": cond_mean_up,
        "cond_mean_down": cond_mean_down,
        "effect_size": effect_size,
    }


def evaluate_bias(df: pd.DataFrame, horizons: list[int], shuffle_seed: int | None = 0) -> dict:
    """Evaluate conditional bias for multiple horizons."""
    results: dict[int, dict] = {}
    active_mask = df.get("active", pd.Series(False, index=df.index)).astype(bool)
    Lambda = df.get("Lambda", pd.Series(index=df.index, dtype=float))

    for h in horizons:
        y_h = future_return(df["r"], horizon=h)

        active_metrics = _effect_metrics(df["mu"][active_mask], y_h[active_mask])
        inactive_metrics = _effect_metrics(df["mu"][~active_mask], y_h[~active_mask])

        shuffled_mu = df["mu"].sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
        shuffled_mu.index = df.index
        shuffled_metrics = _effect_metrics(shuffled_mu[active_mask], y_h[active_mask])

        # Lambda monotonicity across deciles on active set
        decile_metrics = {}
        if Lambda.notna().sum() > 0 and active_mask.any():
            active_lambda = Lambda[active_mask & Lambda.notna()]
            if len(active_lambda) >= 5:
                try:
                    deciles = pd.qcut(active_lambda, q=10, labels=False, duplicates="drop")
                    deciles_full = pd.Series(index=df.index, dtype=float)
                    deciles_full.loc[active_lambda.index] = deciles
                    for dec in sorted(deciles.dropna().unique()):
                        mask_dec = deciles_full == dec
                        decile_metrics[int(dec)] = _effect_metrics(df["mu"][mask_dec], y_h[mask_dec])
                except ValueError:
                    decile_metrics = {}

        results[h] = {
            "active": active_metrics,
            "inactive": inactive_metrics,
            "shuffled": shuffled_metrics,
            "active_share": float(active_mask.mean()),
            "lambda_deciles": decile_metrics,
        }

    return results
