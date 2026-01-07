"""
Trade planning: rounding, min-notional guard, reserve guard, and order sizing.

Single source of truth for all trade planning logic.
"""

import math
from typing import Optional

from spot_bot.core.types import PortfolioState, TradePlan


def _round_qty_to_step(qty: float, step_size: Optional[float]) -> float:
    """
    Round quantity to exchange step size, flooring toward zero.

    Args:
        qty: Quantity to round (can be positive or negative)
        step_size: Exchange step size (e.g., 0.00001 BTC)

    Returns:
        Rounded quantity floored toward zero.
    """
    if step_size is None or step_size <= 0.0:
        return qty

    # Floor toward zero: sign * floor(abs(qty) / step) * step
    sign = 1.0 if qty >= 0 else -1.0
    abs_qty = abs(qty)
    rounded_abs = math.floor(abs_qty / step_size) * step_size
    return sign * rounded_abs


def plan_trade(
    portfolio: PortfolioState,
    price: float,
    target_exposure: float,
    min_notional: float,
    step_size: Optional[float] = None,
    min_usdt_reserve: float = 0.0,
    max_notional_per_trade: Optional[float] = None,
    allow_short: bool = False,
) -> TradePlan:
    """
    Plan a trade given current portfolio and target exposure.

    Args:
        portfolio: Current portfolio state
        price: Current market price
        target_exposure: Desired exposure fraction (post-hysteresis)
        min_notional: Minimum trade notional to execute
        step_size: Exchange quantity step size for rounding
        min_usdt_reserve: Minimum USDT balance to maintain (spot only)
        max_notional_per_trade: Maximum notional per trade (optional cap)
        allow_short: Allow negative positions (False for spot)

    Returns:
        TradePlan with action, deltas, and diagnostics.

    Guards applied in order:
    1. Clamp target_exposure to [0, 1] if not allow_short
    2. Round delta_base to step_size
    3. Check min_notional
    4. Apply USDT reserve guard for BUY
    5. Apply max_notional_per_trade cap
    6. Determine action (HOLD, BUY, SELL)
    """
    # Clamp target exposure for spot (no shorting)
    if not allow_short:
        target_exposure = max(0.0, min(1.0, target_exposure))

    # Compute target base position from target exposure
    if price <= 0.0:
        return TradePlan(
            action="HOLD",
            target_exposure=portfolio.exposure,
            target_base=portfolio.base,
            delta_base=0.0,
            notional=0.0,
            exec_price_hint=price,
            reason="invalid_price",
            diagnostics={"price": price},
        )

    equity = portfolio.equity
    if equity <= 0.0:
        target_base = 0.0
    else:
        target_base = (equity * target_exposure) / price

    delta_base = target_base - portfolio.base

    # Round to step size
    delta_base_rounded = _round_qty_to_step(delta_base, step_size)

    # If rounding brought delta to zero, HOLD
    if abs(delta_base_rounded) < 1e-12:
        return TradePlan(
            action="HOLD",
            target_exposure=portfolio.exposure,
            target_base=portfolio.base,
            delta_base=0.0,
            notional=0.0,
            exec_price_hint=price,
            reason="rounded_to_zero",
            diagnostics={
                "delta_base_raw": delta_base,
                "step_size": step_size,
            },
        )

    notional = abs(delta_base_rounded) * price

    # Min notional guard
    if notional < min_notional:
        return TradePlan(
            action="HOLD",
            target_exposure=portfolio.exposure,
            target_base=portfolio.base,
            delta_base=0.0,
            notional=0.0,
            exec_price_hint=price,
            reason="min_notional",
            diagnostics={
                "notional": notional,
                "min_notional": min_notional,
            },
        )

    # Reserve guard for BUY (spot only)
    if delta_base_rounded > 0 and min_usdt_reserve > 0:
        # Buying would consume USDT
        usdt_after = portfolio.usdt - notional  # approximate, ignoring fees
        if usdt_after < min_usdt_reserve:
            # Reduce qty to respect reserve
            max_spend = portfolio.usdt - min_usdt_reserve
            if max_spend <= 0:
                return TradePlan(
                    action="HOLD",
                    target_exposure=portfolio.exposure,
                    target_base=portfolio.base,
                    delta_base=0.0,
                    notional=0.0,
                    exec_price_hint=price,
                    reason="reserve_guard",
                    diagnostics={
                        "usdt": portfolio.usdt,
                        "min_usdt_reserve": min_usdt_reserve,
                    },
                )
            # Cap delta_base by max_spend
            max_base = max_spend / price
            delta_base_rounded = _round_qty_to_step(max_base, step_size)
            if abs(delta_base_rounded) < 1e-12:
                return TradePlan(
                    action="HOLD",
                    target_exposure=portfolio.exposure,
                    target_base=portfolio.base,
                    delta_base=0.0,
                    notional=0.0,
                    exec_price_hint=price,
                    reason="reserve_guard_rounded_to_zero",
                    diagnostics={
                        "max_spend": max_spend,
                        "max_base": max_base,
                    },
                )
            notional = abs(delta_base_rounded) * price

    # Max notional per trade cap
    if max_notional_per_trade is not None and notional > max_notional_per_trade:
        # Scale down delta_base proportionally
        scale = max_notional_per_trade / notional
        delta_base_capped = delta_base_rounded * scale
        delta_base_rounded = _round_qty_to_step(delta_base_capped, step_size)
        notional = abs(delta_base_rounded) * price
        if abs(delta_base_rounded) < 1e-12 or notional < min_notional:
            return TradePlan(
                action="HOLD",
                target_exposure=portfolio.exposure,
                target_base=portfolio.base,
                delta_base=0.0,
                notional=0.0,
                exec_price_hint=price,
                reason="max_notional_capped_to_zero",
                diagnostics={
                    "max_notional_per_trade": max_notional_per_trade,
                },
            )

    # Determine action
    action = "HOLD"
    if delta_base_rounded > 0:
        action = "BUY"
    elif delta_base_rounded < 0:
        action = "SELL"

    # Final target_base after rounding
    final_target_base = portfolio.base + delta_base_rounded

    return TradePlan(
        action=action,
        target_exposure=target_exposure,
        target_base=final_target_base,
        delta_base=delta_base_rounded,
        notional=notional,
        exec_price_hint=price,
        reason="trade_planned",
        diagnostics={
            "delta_base_raw": delta_base,
            "delta_base_rounded": delta_base_rounded,
            "notional": notional,
        },
    )


__all__ = ["plan_trade"]
