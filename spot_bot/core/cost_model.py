"""
Cost model for trading fees, slippage, and spread.

Single source of truth for cost computation matching live compute_step definition.
"""


def compute_cost_per_turnover(
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
) -> float:
    """
    Compute total cost per unit of turnover.

    Args:
        fee_rate: Transaction fee rate (e.g., 0.001 for 0.1%)
        slippage_bps: Slippage in basis points (e.g., 5.0 for 5 bps)
        spread_bps: Spread in basis points (e.g., 2.0 for 2 bps)

    Returns:
        Total cost as a fraction of turnover.

    Formula:
        cost = fee_rate + 2 * (slippage_bps / 10000) + (spread_bps / 10000)

    The slippage term is doubled because it affects both entry and exit.
    The spread is paid once on average when crossing the bid-ask.
    """
    return fee_rate + 2.0 * (slippage_bps / 10_000.0) + (spread_bps / 10_000.0)


__all__ = ["compute_cost_per_turnover"]
