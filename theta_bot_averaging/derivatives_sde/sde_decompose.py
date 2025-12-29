#!/usr/bin/env python3
"""SDE decomposition for derivatives-driven drift and diffusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .mu_model import compute_mu_components
from .sigma_model import compute_sigma
from .state import build_state


def compute_lambda(mu: pd.Series, sigma: pd.Series) -> pd.Series:
    """Compute determinism score Lambda."""
    lam = mu.abs() / sigma
    return lam.replace([np.inf, -np.inf], np.nan)


def gate_lambda(Lambda: pd.Series, tau: float | None = 1.0, tau_quantile: float | None = None) -> pd.Series:
    """Determine active periods based on threshold or quantile."""
    if tau_quantile is not None:
        threshold = Lambda.quantile(tau_quantile)
        return Lambda > threshold
    if tau is None:
        return pd.Series(False, index=Lambda.index)
    return Lambda > tau


def decompose_symbol(
    symbol: str,
    data_dir: str = "data/raw",
    out_dir: str = "data/processed/derivatives_sde",
    start: str | None = None,
    end: str | None = None,
    z_window: int = 168,
    sigma_window: int = 168,
    tau: float = 1.0,
    tau_quantile: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.5,
    ewma_lambda: float | None = None,
) -> pd.DataFrame:
    """Run full decomposition for a symbol and persist results."""
    state_df = build_state(symbol=symbol, data_dir=data_dir, z_window=z_window)
    if start:
        state_df = state_df.loc[pd.to_datetime(start, utc=True) :]
    if end:
        state_df = state_df.loc[: pd.to_datetime(end, utc=True)]
    mu, mu1, mu2, mu3 = compute_mu_components(state_df, alpha=alpha, beta=beta, gamma=gamma)
    sigma = compute_sigma(state_df["r"], window=sigma_window, ewma_lambda=ewma_lambda)
    Lambda = compute_lambda(mu, sigma)
    active = gate_lambda(Lambda, tau=tau, tau_quantile=tau_quantile)

    out = pd.DataFrame(
        {
            "r": state_df["r"],
            "mu": mu,
            "sigma": sigma,
            "Lambda": Lambda,
            "active": active,
            "mu1": mu1,
            "mu2": mu2,
            "mu3": mu3,
            "z_funding": state_df["z_funding"],
            "z_oi_change": state_df["z_oi_change"],
            "z_basis": state_df["z_basis"],
        },
        index=state_df.index,
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{symbol}_1h.csv.gz"
    out.to_csv(out_path, compression="gzip", index_label="timestamp")
    return out
