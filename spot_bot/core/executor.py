"""
Executor abstraction for trade execution.

Provides unified interface for executing trades in live vs simulated modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from spot_bot.core.engine import EngineParams, simulate_execution
from spot_bot.core.types import ExecutionResult, TradePlan

if TYPE_CHECKING:
    from spot_bot.execution.ccxt_executor import CCXTExecutor


class Executor(ABC):
    """Abstract interface for trade execution."""

    @abstractmethod
    def execute(self, plan: TradePlan, price: float) -> ExecutionResult:
        """
        Execute a trade plan.

        Args:
            plan: Trade plan from engine.run_step
            price: Current market price

        Returns:
            ExecutionResult with fill details
        """
        ...


class SimExecutor(Executor):
    """Simulated executor using engine simulation logic."""

    def __init__(self, params: EngineParams) -> None:
        """
        Initialize simulated executor.

        Args:
            params: Engine parameters for simulation (fees, slippage)
        """
        self.params = params

    def execute(self, plan: TradePlan, price: float) -> ExecutionResult:
        """
        Execute trade plan using simulation.

        Args:
            plan: Trade plan to execute
            price: Current market price

        Returns:
            ExecutionResult from simulate_execution
        """
        return simulate_execution(plan, price, self.params)


class LiveExecutor(Executor):
    """Live executor wrapping CCXT for real exchange execution."""

    def __init__(self, ccxt_executor: "CCXTExecutor") -> None:
        """
        Initialize live executor.

        Args:
            ccxt_executor: Configured CCXTExecutor instance
        """
        self.ccxt_executor = ccxt_executor

    def execute(self, plan: TradePlan, price: float) -> ExecutionResult:
        """
        Execute trade plan on real exchange.

        Args:
            plan: Trade plan to execute
            price: Current market price

        Returns:
            ExecutionResult with real fill details

        Note:
            If plan.action is HOLD or delta_base is 0, returns SKIPPED result.
            Otherwise, calls CCXT executor and converts result to ExecutionResult.
        """
        # Skip if no action needed
        if plan.action == "HOLD" or plan.delta_base == 0.0:
            return ExecutionResult(
                filled_base=0.0,
                avg_price=price,
                fee_paid=0.0,
                slippage_paid=0.0,
                status="SKIPPED",
                raw=None,
            )

        # Determine side
        side = "buy" if plan.delta_base > 0 else "sell"
        qty = abs(plan.delta_base)

        # Execute via CCXT
        ccxt_result = self.ccxt_executor.place_market_order(side, qty, price)

        # Convert CCXT result to core ExecutionResult
        status = ccxt_result.get("status", "error")

        if status == "filled":
            filled_qty = float(ccxt_result.get("filled_qty", qty))
            avg_price = float(ccxt_result.get("avg_price", price))
            fee_est = float(ccxt_result.get("fee_est", 0.0))

            # Restore sign
            filled_base = filled_qty if plan.delta_base > 0 else -filled_qty

            # Compute slippage
            slippage_paid = abs(avg_price - price) * filled_qty

            return ExecutionResult(
                filled_base=filled_base,
                avg_price=avg_price,
                fee_paid=fee_est,
                slippage_paid=slippage_paid,
                status="filled",
                raw=ccxt_result,
            )
        else:
            # Rejected or error
            return ExecutionResult(
                filled_base=0.0,
                avg_price=price,
                fee_paid=0.0,
                slippage_paid=0.0,
                status=status,
                raw=ccxt_result,
            )


__all__ = [
    "Executor",
    "SimExecutor",
    "LiveExecutor",
]
