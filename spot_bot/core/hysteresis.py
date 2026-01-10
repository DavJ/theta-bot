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
    vol_hyst_mode: str = "increase",
) -> float:
    """
    Stable hysteresis threshold based on costs + volatility + edge with smooth bounds.
    
    Computes minimum exposure change threshold (delta_e_min) to prevent
    excessive trading in noisy markets. The threshold is scaled by trading
    costs, current volatility, and required edge. Uses smooth tanh-based
    bounding to avoid binary regime transitions.
    
    Args:
        rv_current: Current realized volatility
        rv_ref: Reference realized volatility (long-horizon stable anchor)
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
        vol_hyst_mode: Volatility hysteresis mode ("increase"|"decrease"|"none")
    
    Returns:
        delta_e_min: Minimum exposure change threshold in [0, 1]
    
    Formula:
        rv = max(rv_current, 1e-12)
        rv_ref_safe = max(rv_ref, 1e-12)
        rv_norm = rv / rv_ref_safe
        cost_r = 2.0*fee_rate + (slippage_bps + spread_bps)*1e-4  # round-trip approx
        edge_r = edge_bps*1e-4
        
        # Volatility multiplier based on mode
        if vol_hyst_mode == "increase":
            vol_mult = 1.0 + k_vol * rv_norm
        elif vol_hyst_mode == "decrease":
            vol_mult = 1.0 / (1.0 + k_vol * rv_norm)
        else:  # "none"
            vol_mult = 1.0
            
        raw = hyst_k * (cost_r + edge_r) * vol_mult
        
        # Smooth minimum + smooth maximum (avoid binary transitions)
        x = soft_max(raw, hyst_floor, alpha_floor)         # enforce MIN hysteresis smoothly
        x = soft_min(x, max_delta_e_min, alpha_cap)        # enforce MAX hysteresis smoothly
    """
    # Ensure rv_current and rv_ref are valid
    rv = max(float(rv_current) if rv_current else 0.0, 1e-12)
    rv_ref_safe = max(float(rv_ref) if rv_ref else 0.0, 1e-12)
    
    # Compute normalized volatility
    rv_norm = rv / rv_ref_safe
    
    # Convert costs into return units (round-trip approximation)
    cost_r = 2.0 * float(fee_rate) + (float(slippage_bps) + float(spread_bps)) * 1e-4
    
    # Add small extra required edge in bps
    edge_r = float(edge_bps) * 1e-4
    
    # Compute volatility multiplier based on mode
    mode = str(vol_hyst_mode).strip().lower()
    if mode == "increase":
        # Higher volatility -> higher threshold (more conservative)
        vol_mult = 1.0 + float(k_vol) * rv_norm
    elif mode == "decrease":
        # Higher volatility -> lower threshold (less conservative)
        vol_mult = 1.0 / (1.0 + float(k_vol) * rv_norm)
    elif mode == "none":
        # No volatility adjustment
        vol_mult = 1.0
    else:
        raise ValueError(
            f"Invalid vol_hyst_mode: {vol_hyst_mode!r}. "
            f"Must be one of: 'increase', 'decrease', 'none'"
        )
    
    # Map return threshold to exposure threshold via hyst_k scaling
    # Apply volatility multiplier to the combined cost+edge
    raw = float(hyst_k) * (cost_r + edge_r) * vol_mult
    
    # Apply smooth floor and cap for stability (avoid binary transitions)
    x = soft_max(raw, float(hyst_floor), float(alpha_floor))      # enforce floor smoothly
    x = soft_min(x, float(max_delta_e_min), float(alpha_cap))     # enforce cap smoothly
    
    return float(x)


def apply_hysteresis(
    current_exposure: float,
    target_exposure: float,
    delta_e_min: float,
    mode: str = "exposure",
    current_zscore: float = 0.0,
    target_zscore: float = 0.0,
) -> Tuple[float, bool]:
    """
    Apply hysteresis to suppress small exposure changes.

    Args:
        current_exposure: Current exposure fraction
        target_exposure: Desired target exposure fraction
        delta_e_min: Minimum threshold from compute_hysteresis_threshold
        mode: Hysteresis mode - "exposure" or "zscore"
        current_zscore: Current z-score (only used if mode="zscore")
        target_zscore: Target z-score (only used if mode="zscore")

    Returns:
        Tuple of (final_target_exposure, suppressed)
        - final_target_exposure: Target after hysteresis (may equal current)
        - suppressed: True if trade was suppressed by hysteresis

    Modes:
        - "exposure": Compare abs(target_exposure - current_exposure) <= delta_e_min
        - "zscore": Compare abs(target_zscore - current_zscore) <= delta_e_min
          (Note: In zscore mode, delta_e_min is interpreted as minimum z-score change)

    If the threshold is not exceeded, we suppress the trade by setting target = current.
    
    Boundary semantics: Uses <= comparison so that delta_e_min is the MINIMUM
    allowed threshold (i.e., changes exactly equal to delta_e_min are suppressed).
    """
    if mode == "exposure":
        delta_e = abs(target_exposure - current_exposure)
        # Use <= so that delta_e_min is a true floor: changes AT the floor are suppressed
        if delta_e <= delta_e_min:
            return float(current_exposure), True
        return float(target_exposure), False
    elif mode == "zscore":
        # Z-score mode: compare z-score deltas instead of exposure deltas
        # This is useful when strategy operates in z-score space
        delta_z = abs(target_zscore - current_zscore)
        if delta_z <= delta_e_min:
            return float(current_exposure), True
        return float(target_exposure), False
    else:
        raise ValueError(
            f"Invalid hysteresis mode: {mode!r}. Must be 'exposure' or 'zscore'."
        )


__all__ = [
    "soft_max",
    "soft_min",
    "compute_hysteresis_threshold",
    "apply_hysteresis",
]
