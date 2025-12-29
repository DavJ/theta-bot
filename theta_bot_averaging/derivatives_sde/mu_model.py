#!/usr/bin/env python3
"""Parametric drift model mu(d, t)."""

from __future__ import annotations

import pandas as pd


def compute_mu_components(
    state_df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.5,
    rho: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute mu(t) components based on standardized state."""
    z_oi_change = state_df.get("z_oi_change")
    z_funding = state_df.get("z_funding")
    z_basis = state_df.get("z_basis")

    mu1 = -alpha * z_oi_change * z_funding
    mu2 = beta * z_oi_change * z_basis

    if rho is not None and gamma != 0:
        rho_aligned = rho.reindex(state_df.index)
        mu3 = gamma * rho_aligned * z_basis
    else:
        mu3 = pd.Series(0.0, index=state_df.index, name="mu3")

    mu = mu1 + mu2 + mu3
    return mu, mu1, mu2, mu3
