"""
Hysteresis logic to prevent excessive trading in noisy markets.

Single source of truth for hysteresis threshold computation and application.
"""

from typing import Tuple


def compute_hysteresis_threshold(
    hyst_k: float,
    hyst_floor: float,
    cost: float,
    rv_ref: float,
    rv_current: float,
) -> float:
    """
    Compute minimum exposure delta required to trade.

    Args:
        hyst_k: Hysteresis multiplier (typically 5.0)
        hyst_floor: Minimum threshold regardless of volatility (e.g., 0.02)
        cost: Cost per unit of turnover from cost_model
        rv_ref: Reference realized volatility (e.g., median of last 500 bars)
        rv_current: Current realized volatility

    Returns:
        Minimum absolute exposure change required to trade.

    Formula:
        rv_current_safe = max(rv_current, 1e-8)
        delta_e_min = max(hyst_floor, hyst_k * cost * (rv_ref / rv_current_safe))

    When volatility is low (rv_current < rv_ref), threshold increases,
    making it harder to trade. This prevents overtrading in calm markets.
    """
    rv_current_safe = max(rv_current, 1e-8)
    delta_e_min = max(hyst_floor, hyst_k * cost * (rv_ref / rv_current_safe))
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
