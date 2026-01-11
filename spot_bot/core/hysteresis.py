"""
Hysteresis logic to prevent excessive trading in noisy markets.

Single source of truth for hysteresis threshold computation and application.
"""
from __future__ import annotations
import math
from typing import Tuple, Union, Dict
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

def compute_return_threshold(
    fee_rate: float,
    spread_bps: float,
    slippage_bps: float,
    edge_bps: float,
    min_profit_bps: float,
    rv_current: float,
    rv_ref: float,
    k_vol: float,
    vol_hyst_mode: str,
) -> float:
    """
    Compute return threshold for limit pricing and sell guard.
    
    This is the minimum return required to justify a trade, accounting for:
    - Round-trip costs (fees + spread + slippage)
    - Required edge
    - Minimum profit buffer
    - Volatility adjustment
    
    Args:
        fee_rate: Exchange fee rate (e.g., 0.001 for 0.1%)
        spread_bps: Spread in basis points
        slippage_bps: Slippage in basis points
        edge_bps: Required edge in basis points
        min_profit_bps: Minimum profit buffer in basis points
        rv_current: Current realized volatility
        rv_ref: Reference realized volatility
        k_vol: Volatility multiplier
        vol_hyst_mode: Volatility mode ("increase"|"decrease"|"none")
    
    Returns:
        Return threshold as a fraction (e.g., 0.01 for 1%)
    
    Formula:
        cost_r = 2*fee_rate + (spread_bps + slippage_bps)*1e-4
        edge_r = edge_bps*1e-4
        minp_r = min_profit_bps*1e-4
        
        rv_norm = rv_current / max(rv_ref, 1e-12)
        
        if vol_hyst_mode == "none": vol_mult = 1.0
        if "increase": vol_mult = 1.0 + k_vol*(rv_norm - 1.0)
        if "decrease": vol_mult = 1.0 + k_vol*(1.0/max(rv_norm,1e-6) - 1.0)
        
        vol_mult = clamp(vol_mult, 0.25, 4.0)
        
        return_threshold = (cost_r + edge_r + minp_r) * vol_mult
    """
    # Ensure valid inputs
    rv = max(float(rv_current) if rv_current else 0.0, 1e-12)
    rv_ref_safe = max(float(rv_ref) if rv_ref else 0.0, 1e-12)
    
    # Compute normalized volatility
    rv_norm = rv / rv_ref_safe
    
    # Convert costs to return units (round-trip)
    cost_r = 2.0 * float(fee_rate) + (float(spread_bps) + float(slippage_bps)) * 1e-4
    
    # Add edge and min profit
    edge_r = float(edge_bps) * 1e-4
    minp_r = float(min_profit_bps) * 1e-4
    
    # Compute volatility multiplier based on mode
    mode = str(vol_hyst_mode).strip().lower()
    if mode == "increase":
        # Higher volatility -> higher threshold (more conservative)
        vol_mult = 1.0 + float(k_vol) * (rv_norm - 1.0)
    elif mode == "decrease":
        # Higher volatility -> lower threshold (less conservative)
        vol_mult = 1.0 + float(k_vol) * (1.0 / max(rv_norm, 1e-6) - 1.0)
    elif mode == "none":
        # No volatility adjustment
        vol_mult = 1.0
    else:
        raise ValueError(
            f"Invalid vol_hyst_mode: {vol_hyst_mode!r}. "
            f"Must be one of: 'increase', 'decrease', 'none'"
        )
    
    # Clamp vol_mult to avoid explosions
    vol_mult = max(0.25, min(4.0, vol_mult))
    
    # Compute final return threshold
    return_threshold = (cost_r + edge_r + minp_r) * vol_mult
    
    return float(return_threshold)


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
    return_diagnostics: bool = False,
) -> Union[float, Tuple[float, Dict[str, Union[float, bool]]]]:
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
        return_diagnostics: If True, return (delta_e_min, diagnostics_dict) instead of just delta_e_min
    
    Returns:
        If return_diagnostics=False: delta_e_min (float)
        If return_diagnostics=True: (delta_e_min, diagnostics_dict) tuple
        
        diagnostics_dict contains:
            - hyst_raw: Raw hysteresis value before floor/cap
            - hyst_after_floor: After applying floor via soft_max
            - hyst_final: Final value after cap (same as delta_e_min)
            - floor_binding: True if floor constraint is active
            - cap_binding: True if cap constraint is active
            - rv_norm: Normalized volatility ratio
            - vol_mult: Volatility multiplier applied
    
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
    hyst_raw = float(hyst_k) * (cost_r + edge_r) * vol_mult
    
    # Apply smooth floor and cap for stability (avoid binary transitions)
    hyst_after_floor = soft_max(hyst_raw, float(hyst_floor), float(alpha_floor))      # enforce floor smoothly
    hyst_final = soft_min(hyst_after_floor, float(max_delta_e_min), float(alpha_cap))     # enforce cap smoothly
    
    # Compute binding flags (use small epsilon for numerical tolerance)
    eps = 1e-9
    floor_binding = (hyst_raw < float(hyst_floor) - eps)
    cap_binding = (hyst_after_floor > float(max_delta_e_min) + eps)
    
    if return_diagnostics:
        diagnostics = {
            "hyst_raw": float(hyst_raw),
            "hyst_after_floor": float(hyst_after_floor),
            "hyst_final": float(hyst_final),
            "floor_binding": bool(floor_binding),
            "cap_binding": bool(cap_binding),
            "rv_norm": float(rv_norm),
            "vol_mult": float(vol_mult),
        }
        return float(hyst_final), diagnostics
    
    return float(hyst_final)


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
    "compute_return_threshold",
    "compute_hysteresis_threshold",
    "apply_hysteresis",
]
