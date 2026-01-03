#!/usr/bin/env python3
"""
Gating logic for drift signals.

Apply threshold or quantile-based gating to determine when drift is active.
"""

from __future__ import annotations

import pandas as pd


def apply_quantile_gate(
    D: pd.Series,
    quantile: float = 0.85,
) -> pd.Series:
    """
    Apply quantile-based gating to determinism.
    
    active(t) = D(t) > quantile_threshold
    
    Parameters
    ----------
    D : pd.Series
        Determinism magnitude series
    quantile : float
        Quantile threshold (default: 0.85 for 85th percentile)
        
    Returns
    -------
    pd.Series
        Boolean series indicating active periods
    """
    threshold = D.quantile(quantile)
    active = D > threshold
    return active


def apply_threshold_gate(
    D: pd.Series,
    threshold: float,
) -> pd.Series:
    """
    Apply fixed threshold gating to determinism.
    
    active(t) = D(t) > threshold
    
    Parameters
    ----------
    D : pd.Series
        Determinism magnitude series
    threshold : float
        Fixed threshold value
        
    Returns
    -------
    pd.Series
        Boolean series indicating active periods
    """
    active = D > threshold
    return active


def apply_combined_gate(
    D: pd.Series,
    quantile: float = 0.85,
    threshold: float = None,
) -> pd.Series:
    """
    Apply combined quantile OR threshold gating.
    
    active(t) = D(t) > quantile_threshold OR D(t) > fixed_threshold
    
    Parameters
    ----------
    D : pd.Series
        Determinism magnitude series
    quantile : float
        Quantile threshold (default: 0.85)
    threshold : float, optional
        Fixed threshold value (if None, only quantile is used)
        
    Returns
    -------
    pd.Series
        Boolean series indicating active periods
    """
    active_q = apply_quantile_gate(D, quantile)
    
    if threshold is not None:
        active_t = apply_threshold_gate(D, threshold)
        active = active_q | active_t
    else:
        active = active_q
    
    return active
