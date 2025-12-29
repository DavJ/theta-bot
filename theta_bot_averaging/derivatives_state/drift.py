#!/usr/bin/env python3
"""
Drift computation from derivatives state.

Compute mu(t) directional pressure and D(t) determinism magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_drift(
    z_oi_change: pd.Series,
    z_funding: pd.Series,
    z_basis: pd.Series,
    rho: pd.Series = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute drift mu(t) from standardized derivatives features.
    
    Definitions:
        mu1(t) = -alpha * z(OI'(t)) * z(f(t))        # overcrowding unwind
        mu2(t) =  beta  * z(OI'(t)) * z(b(t))        # basis-pressure
        mu3(t) =  gamma * rho(t) * z(b(t))           # expiry/roll pressure
        mu(t) = mu1 + mu2 + mu3
    
    Sign convention for mu1:
    - Positive OI change + positive funding -> crowded long -> negative drift (expect unwind)
    - Hence the negative sign: -alpha
    
    Parameters
    ----------
    z_oi_change : pd.Series
        Z-scored open interest change
    z_funding : pd.Series
        Z-scored funding rate
    z_basis : pd.Series
        Z-scored basis
    rho : pd.Series, optional
        Expiry proximity indicator (0 to 1)
    alpha : float
        Weight for overcrowding unwind term (default: 1.0)
    beta : float
        Weight for basis-pressure term (default: 1.0)
    gamma : float
        Weight for expiry/roll pressure term (default: 0.0)
        
    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        (mu, mu1, mu2, mu3)
    """
    # Compute mu1: overcrowding unwind
    mu1 = -alpha * z_oi_change * z_funding
    
    # Compute mu2: basis-pressure
    mu2 = beta * z_oi_change * z_basis
    
    # Compute mu3: expiry/roll pressure (optional)
    if rho is not None and gamma != 0:
        mu3 = gamma * rho * z_basis
    else:
        mu3 = pd.Series(0.0, index=z_oi_change.index)
    
    # Total drift
    mu = mu1 + mu2 + mu3
    
    return mu, mu1, mu2, mu3


def compute_determinism(mu: pd.Series) -> pd.Series:
    """
    Compute determinism magnitude D(t) = |mu(t)|.
    
    Parameters
    ----------
    mu : pd.Series
        Drift series
        
    Returns
    -------
    pd.Series
        Determinism magnitude
    """
    D = np.abs(mu)
    return D
