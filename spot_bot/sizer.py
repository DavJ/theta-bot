"""
Position sizing for Spot Bot 2.0.

Translates strategy intents into executable order sizes with risk-aware
controls, including the risk sizing required in Phase C.
"""

from typing import Any


class PositionSizer:
    """Applies sizing rules to strategy intents before execution."""

    def size(self, intent: Any, features: Any = None, regime_state: Any = None) -> Any:
        """Return a sized order or target position."""
        raise NotImplementedError("Position sizing is not implemented yet.")
