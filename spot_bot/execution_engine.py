"""
Execution layer for Spot Bot 2.0.

Responsible for translating sized intents into orders for either simulated
backtests or live exchange connectivity.
"""

from typing import Any


class ExecutionEngine:
    """Routes sized intents to the appropriate execution venue."""

    def execute(self, sized_intent: Any) -> Any:
        """Submit the sized intent and return execution metadata."""
        raise NotImplementedError("Execution handling is not implemented yet.")
