"""
Hysteresis logic to prevent excessive trading in noisy markets.

Single source of truth for hysteresis threshold computation and application.
"""
from __future__ import annotations
import math
from typing import Tuple
# spot_bot/core/hysteresis.py


def soft_max(a: float, b: float, alpha: float) -> float:
    """
    Smooth maximum using tanh approximation.
    
    soft_max(a, b, alpha) ≈ max(a, b) when alpha is large.
    When alpha → ∞, converges to hard max.
    
    Formula: 0.5*(a+b) + 0.5*(a-b)*tanh(alpha*(a-b))
    
    Args:
        a: First value
        b: Second value
        alpha: Smoothness parameter (higher = closer to hard max)
    
    Returns:
        Smooth maximum of a and b
    """
    return 0.5 * (a + b) + 0.5 * (a - b) * math.tanh(alpha * (a - b))


def soft_min(a: float, b: float, alpha: float) -> float:
    """
    Smooth minimum using tanh approximation.
    
    soft_min(a, b, alpha) ≈ min(a, b) when alpha is large.
    When alpha → ∞, converges to hard min.
    
    Formula: 0.5*(a+b) - 0.5*(a-b)*tanh(alpha*(a-b))
    
    Args:
        a: First value
        b: Second value
        alpha: Smoothness parameter (higher = closer to hard min)
    
    Returns:
        Smooth minimum of a and b
    """
    return 0.5 * (a + b) - 0.5 * (a - b) * math.tanh(alpha * (a - b))

def compute_hysteresis_threshold(
    *,
    rv_current: float,
    rv_ref: float,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    hyst_k: float,
    hyst_floor: float,
    k_vol: float = 0.5,
    edge_bps: float = 5.0,
    max_delta_e_min: float = 0.3,
    alpha_floor: float = 6.0,
    alpha_cap: float = 6.0,
) -> float:
    """
    Stable hysteresis threshold based on costs + volatility + edge with smooth bounds.
    
    Computes minimum exposure change threshold (delta_e_min) to prevent
    excessive trading in noisy markets. The threshold is scaled by trading
    costs, current volatility, and required edge. Uses smooth tanh-based
    bounding to avoid binary regime transitions.
    
    Args:
        rv_current: Current realized volatility
        rv_ref: Reference realized volatility (unused in stable formula)
        fee_rate: Exchange fee rate (e.g., 0.001 for 0.1%)
        slippage_bps: Slippage in basis points
        spread_bps: Spread in basis points
        hyst_k: Hysteresis scaling factor (converts return threshold to exposure threshold)
        hyst_floor: Minimum threshold floor (minimum allowed hysteresis)
        k_vol: Volatility multiplier for threshold
        edge_bps: Required edge in basis points
        max_delta_e_min: Maximum threshold cap (maximum allowed hysteresis, default 0.3)
        alpha_floor: Smoothness parameter for minimum bound (default 6.0)
        alpha_cap: Smoothness parameter for maximum bound (default 6.0)
    
    Returns:
        delta_e_min: Minimum exposure change threshold in [0, 1]
    
    Formula:
        rv = max(rv_current, 1e-12)
        cost_r = 2.0*fee_rate + (slippage_bps + spread_bps)*1e-4  # round-trip approx
        edge_r = edge_bps*1e-4
        vol_r  = k_vol * rv
        raw = hyst_k * (cost_r + edge_r + vol_r)
        # Smooth minimum + smooth maximum (avoid binary transitions)
        x = soft_max(raw, hyst_floor, alpha_floor)         # enforce MIN hysteresis smoothly
        x = soft_min(x, max_delta_e_min, alpha_cap)        # enforce MAX hysteresis smoothly
    """
    # Ensure rv_current is valid
    rv = max(float(rv_current) if rv_current else 0.0, 1e-12)
    
    # Convert costs into return units (round-trip approximation)
    cost_r = 2.0 * float(fee_rate) + (float(slippage_bps) + float(spread_bps)) * 1e-4
    
    # Add small extra required edge in bps
    edge_r = float(edge_bps) * 1e-4
    
    # Add volatility term proportional to rv_current (NOT ratios rv_ref/rv_current)
    vol_r = float(k_vol) * rv
    
    # Map return threshold to exposure threshold via hyst_k scaling
    raw = float(hyst_k) * (cost_r + edge_r + vol_r)
    
    # Apply smooth floor and cap for stability (avoid binary transitions)
    x = soft_max(raw, float(hyst_floor), float(alpha_floor))      # enforce floor smoothly
    x = soft_min(x, float(max_delta_e_min), float(alpha_cap))     # enforce cap smoothly
    
    return float(x)


def apply_hysteresis(
    current_exposure: float,
    target_exposure: float,
    delta_e_min: float,
) -> Tuple[float, bool]:
    """
    Apply hysteresis to suppress small exposure changes.

    Args:
        current_exposure: Current exposure fraction
        target_exposure: Desired target exposure fraction
        delta_e_min: Minimum threshold from compute_hysteresis_threshold

    Returns:
        Tuple of (final_target_exposure, suppressed)
        - final_target_exposure: Target after hysteresis (may equal current)
        - suppressed: True if trade was suppressed by hysteresis

    If abs(target - current) < delta_e_min, we suppress the trade by
    setting target = current.
    """
    delta_e = abs(target_exposure - current_exposure)
    if delta_e < delta_e_min:
        return float(current_exposure), True
    return float(target_exposure), False


__all__ = [
    "soft_max",
    "soft_min",
    "compute_hysteresis_threshold",
    "apply_hysteresis",
]
