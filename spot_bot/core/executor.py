"""
Executor abstraction for trade execution.

Provides unified interface for executing trades in live vs simulated modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from spot_bot.core.engine import EngineParams, simulate_execution
from spot_bot.core.types import ExecutionResult, TradePlan


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


__all__ = [
    "Executor",
    "SimExecutor",
]
