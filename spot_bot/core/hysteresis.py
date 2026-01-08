"""
Hysteresis logic to prevent excessive trading in noisy markets.

Single source of truth for hysteresis threshold computation and application.
"""
from __future__ import annotations
from typing import Tuple
# spot_bot/core/hysteresis.py

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
) -> float:
    """
    Stable hysteresis threshold based on costs + volatility + edge.
    
    Computes minimum exposure change threshold (delta_e_min) to prevent
    excessive trading in noisy markets. The threshold is scaled by trading
    costs, current volatility, and required edge.
    
    Args:
        rv_current: Current realized volatility
        rv_ref: Reference realized volatility (unused in stable formula)
        fee_rate: Exchange fee rate (e.g., 0.001 for 0.1%)
        slippage_bps: Slippage in basis points
        spread_bps: Spread in basis points
        hyst_k: Hysteresis scaling factor (converts return threshold to exposure threshold)
        hyst_floor: Minimum threshold floor
        k_vol: Volatility multiplier for threshold
        edge_bps: Required edge in basis points
        max_delta_e_min: Maximum threshold cap (default 0.3)
    
    Returns:
        delta_e_min: Minimum exposure change threshold in [0, 1]
    
    Formula:
        rv = max(rv_current, 1e-12)
        cost_r = 2.0*fee_rate + (slippage_bps + spread_bps)*1e-4  # round-trip approx
        edge_r = edge_bps*1e-4
        vol_r  = k_vol * rv
        delta_e_min = hyst_k * (cost_r + edge_r + vol_r)
        delta_e_min = max(hyst_floor, min(max_delta_e_min, delta_e_min))
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
    delta_e_min = float(hyst_k) * (cost_r + edge_r + vol_r)
    
    # Apply floor and cap for stability
    delta_e_min = max(float(hyst_floor), delta_e_min)
    delta_e_min = min(float(max_delta_e_min), delta_e_min)
    
    return float(delta_e_min)


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
    "compute_hysteresis_threshold",
    "apply_hysteresis",
]
