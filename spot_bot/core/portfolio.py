"""
Portfolio math: equity, exposure, position sizing, and fill application.

Single source of truth for all portfolio calculations.
"""

from spot_bot.core.types import ExecutionResult, PortfolioState


def compute_equity(usdt: float, base: float, price: float) -> float:
    """
    Compute total portfolio equity in quote currency.

    Args:
        usdt: Quote currency balance
        base: Base currency position
        price: Current market price

    Returns:
        Total equity = usdt + base * price
    """
    return float(usdt + base * price)


def compute_exposure(base: float, price: float, equity: float) -> float:
    """
    Compute current exposure as fraction of equity.

    Args:
        base: Base currency position
        price: Current market price
        equity: Total portfolio equity

    Returns:
        Exposure fraction = (base * price) / equity
        Returns 0.0 if equity <= 0
    """
    if equity <= 0.0:
        return 0.0
    return float((base * price) / equity)


def target_base_from_exposure(
    equity: float,
    target_exposure: float,
    price: float,
) -> float:
    """
    Convert target exposure fraction to target base position.

    Args:
        equity: Total portfolio equity
        target_exposure: Desired exposure fraction (0.0 to 1.0)
        price: Current market price

    Returns:
        Target base position = (equity * target_exposure) / price
        Returns 0.0 if price <= 0
    """
    if price <= 0.0:
        return 0.0
    return float((equity * target_exposure) / price)


def apply_fill(
    portfolio: PortfolioState,
    execution: ExecutionResult,
) -> PortfolioState:
    """
    Apply execution result to portfolio state.

    Args:
        portfolio: Current portfolio state
        execution: Execution result with fill details

    Returns:
        Updated portfolio state after applying the fill.

    For BUY (positive delta):
        usdt -= notional + fee + slippage
        base += filled_base

    For SELL (negative delta):
        usdt += notional - fee - slippage
        base -= filled_base

    Equity and exposure are recomputed from the updated balances.
    """
    if execution.status == "SKIPPED" or execution.filled_base == 0.0:
        # No change to portfolio
        return portfolio

    usdt = portfolio.usdt
    base = portfolio.base

    notional = abs(execution.filled_base) * execution.avg_price
    total_cost = execution.fee_paid + execution.slippage_paid

    if execution.filled_base > 0:
        # BUY: spend USDT, gain base
        usdt -= notional + total_cost
        base += execution.filled_base
    else:
        # SELL: gain USDT, lose base
        usdt += notional - total_cost
        base += execution.filled_base  # filled_base is negative for SELL

    # Recompute equity and exposure
    equity = compute_equity(usdt, base, execution.avg_price)
    exposure = compute_exposure(base, execution.avg_price, equity)

    return PortfolioState(
        usdt=usdt,
        base=base,
        equity=equity,
        exposure=exposure,
    )


__all__ = [
    "compute_equity",
    "compute_exposure",
    "target_base_from_exposure",
    "apply_fill",
]
