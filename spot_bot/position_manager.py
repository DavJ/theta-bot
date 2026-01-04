"""
Position management for Spot Bot 2.0.

Maintains long/flat state, synchronizes intents from strategies with executed
orders, and provides a single source of truth for current exposure.
"""

from typing import Any


class PositionManager:
    """Tracks and updates long/flat exposure."""

    def __init__(self) -> None:
        self.current_position: Any = None

    def update(self, intent: Any) -> None:
        """Apply a new intent and adjust internal state accordingly."""
        raise NotImplementedError("Position updates are not implemented yet.")
