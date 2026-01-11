"""
Core data types used across all trading modes.

These types ensure consistent data structures between live, paper, replay,
backtest, and fast_backtest modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass(frozen=True)
class MarketBar:
    """OHLCV bar data."""

    ts: int  # Unix timestamp in milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class DecisionContext:
    """Context information for decision making."""

    symbol: str
    timeframe: str
    mode: str  # "live", "paper", "replay", "backtest", "fast_backtest"
    params: Dict[str, Any]
    now_ts: int  # Unix timestamp in milliseconds


@dataclass(frozen=True)
class StrategyOutput:
    """Output from strategy intent generation."""

    target_exposure: float  # Target exposure as fraction of equity (0.0 to 1.0)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioState:
    """Current portfolio state."""

    usdt: float  # Quote currency balance
    base: float  # Base currency position (e.g., BTC)
    equity: float  # Total equity in quote currency
    exposure: float  # Current exposure as fraction (base * price / equity)
    avg_entry_price: Optional[float] = None  # Average entry price for current net-long position
    realized_pnl_quote: float = 0.0  # Realized P&L in quote currency

    def __post_init__(self) -> None:
        """Ensure values are floats."""
        self.usdt = float(self.usdt)
        self.base = float(self.base)
        self.equity = float(self.equity)
        self.exposure = float(self.exposure)
        if self.avg_entry_price is not None:
            self.avg_entry_price = float(self.avg_entry_price)
        self.realized_pnl_quote = float(self.realized_pnl_quote)


@dataclass(frozen=True)
class TradePlan:
    """Plan for a trade, output of trade planner."""

    action: Literal["HOLD", "BUY", "SELL"]  # Trade action
    target_exposure: float  # Final target exposure after hysteresis
    target_base: float  # Target base position
    delta_base: float  # Change in base position (rounded)
    notional: float  # Trade notional value in quote currency
    exec_price_hint: float  # Reference price for execution
    reason: str  # Human-readable reason for action
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    limit_price: Optional[float] = None  # Limit price for order (None = market)
    order_type: str = "limit"  # Order type: "limit" or "market"


@dataclass
class ExecutionResult:
    """Result of trade execution (real or simulated)."""

    filled_base: float  # Actual base quantity filled
    avg_price: float  # Average fill price
    fee_paid: float  # Fee paid in quote currency
    slippage_paid: float  # Slippage cost in quote currency
    status: str  # "filled", "partial", "rejected", "SKIPPED"
    raw: Optional[Dict[str, Any]] = None  # Raw execution data for debugging

    def __post_init__(self) -> None:
        """Ensure numeric values are floats."""
        self.filled_base = float(self.filled_base)
        self.avg_price = float(self.avg_price)
        self.fee_paid = float(self.fee_paid)
        self.slippage_paid = float(self.slippage_paid)


__all__ = [
    "MarketBar",
    "DecisionContext",
    "StrategyOutput",
    "PortfolioState",
    "TradePlan",
    "ExecutionResult",
]
