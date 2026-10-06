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

    Turnover counts each buy and each sell separately. Each market fill pays
    one fee, one slippage allowance and half the full bid/ask spread.
    """
    return fee_rate + (slippage_bps + spread_bps / 2.0) / 10_000.0


def compute_round_trip_cost(fee_rate: float, slippage_bps: float, spread_bps: float) -> float:
    """First-order entry plus exit cost, in return units."""
    return 2.0 * compute_cost_per_turnover(fee_rate, slippage_bps, spread_bps)


__all__ = ["compute_cost_per_turnover", "compute_round_trip_cost"]
