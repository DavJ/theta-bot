from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def evaluate_symbol(df: pd.DataFrame, horizons: Iterable[int], q: float = 0.85, seed: int = 0) -> Dict:
    """
    Evaluate directional skill of the SDE signals.
    Computes sign agreement, conditional means, effect size, inactive baseline, decile means,
    and shuffled-mu baseline for each forecast horizon.
    """
    log_price = np.log(df["spot_close"])
    lambda_threshold = df["lambda"].quantile(q) if len(df["lambda"].dropna()) else np.nan
    active_mask = df["lambda"] >= lambda_threshold if not np.isnan(lambda_threshold) else pd.Series(False, index=df.index)
    inactive_mask = ~active_mask

    rng = np.random.default_rng(seed)
    metrics: Dict[int, Dict] = {}

    for h in horizons:
        future_ret = log_price.shift(-h) - log_price
        active_ret = future_ret[active_mask]
        active_mu = df.loc[active_ret.index, "mu"]

        sign_agree = (np.sign(active_mu) == np.sign(active_ret)).mean() if len(active_ret) else np.nan
        pos_mean = active_ret[active_mu > 0].mean() if len(active_ret) else np.nan
        neg_mean = active_ret[active_mu < 0].mean() if len(active_ret) else np.nan
        effect_size = pos_mean - neg_mean if pd.notna(pos_mean) and pd.notna(neg_mean) else np.nan
        inactive_mean = future_ret[inactive_mask].mean() if inactive_mask.any() else np.nan

        # Lambda deciles monotonicity
        decile_means: List[float] = []
        if active_mask.any():
            try:
                deciles = pd.qcut(df.loc[active_mask, "lambda"], 10, labels=False, duplicates="drop")
                for dec in sorted(deciles.dropna().unique()):
                    decile_idx = deciles[deciles == dec].index
                    decile_means.append(future_ret.loc[decile_idx].mean())
            except ValueError:
                decile_means = []

        shuffled_agree = np.nan
        if len(active_ret):
            shuffled_mu = rng.permutation(active_mu.values)
            shuffled_agree = (np.sign(shuffled_mu) == np.sign(active_ret.values)).mean()

        metrics[int(h)] = {
            "sign_agree": sign_agree,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "effect_size": effect_size,
            "inactive_mean": inactive_mean,
            "decile_means": decile_means,
            "shuffled_sign_agree": shuffled_agree,
        }

    return {
        "lambda_threshold": lambda_threshold,
        "active_share": float(active_mask.mean()) if len(df) else 0.0,
        "per_horizon": metrics,
    }
