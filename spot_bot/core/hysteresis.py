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
    # nové, defaulty aby nic nerozbily
    k_vol: float = 0.5,
    edge_bps: float = 5.0,
    max_delta_e_min: float = 0.5,
) -> float:
    """
    Dynamic hysteresis:
      delta_e_min = max(hyst_floor, hyst_k * (cost + vol_term + edge))
    where:
      cost = fee + slippage + spread
      vol_term = k_vol * (rv_current/rv_ref)
      edge = edge_bps (bps)
    """

    # costs in fraction
    cost = float(fee_rate) + float(slippage_bps) * 1e-4 + float(spread_bps) * 1e-4

    rv_ref_safe = float(rv_ref) if rv_ref and rv_ref > 1e-12 else 1e-12
    rv_cur_safe = float(rv_current) if rv_current and rv_current > 0.0 else 0.0

    # volatility term (dimensionless; grows when current vol > typical)
    vol_term = float(k_vol) * (rv_cur_safe / rv_ref_safe)

    # edge buffer in fraction
    edge = float(edge_bps) * 1e-4

    threshold = cost + vol_term + edge
    delta_e_min = float(hyst_k) * threshold

    # clamp + floor
    if max_delta_e_min is not None:
        delta_e_min = min(delta_e_min, float(max_delta_e_min))
    delta_e_min = max(float(hyst_floor), delta_e_min)
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
