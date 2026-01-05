from typing import Literal

from spot_bot.utils.normalization import clip01


def compute_target_position(
    equity_usdt: float,
    price: float,
    desired_exposure: float,
    risk_budget: float,
    max_exposure: float,
    risk_state: Literal["ON", "REDUCE", "OFF"] = "ON",
) -> float:
    """
    Compute target BTC position given desired exposure and risk gating.

    Exposure is capped to max_exposure and set to zero when risk_state is not ON.
    """
    if price <= 0:
        raise ValueError("Price must be positive.")

    if risk_state == "OFF":
        target_exposure = 0.0
    elif risk_state == "REDUCE":
        target_exposure = clip01(desired_exposure) * clip01(risk_budget)
    else:
        target_exposure = clip01(desired_exposure) * clip01(risk_budget)

    target_exposure = min(float(max_exposure), target_exposure)
    return float(equity_usdt) * target_exposure / float(price)
