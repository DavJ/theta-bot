"""
Kalman filter-based strategy placeholder for Spot Bot 2.0.

Phase C layers risk sizing on top of psi_logtime features and gating discipline.
"""

from typing import Any, Optional

from .base import Strategy


class KalmanStrategy(Strategy):
    """Generates long/flat intents using Kalman state estimates and risk sizing hooks."""

    def __init__(self, state_dim: Optional[int] = None) -> None:
        self.state_dim = state_dim

    def generate_signal(self, features: Any, regime_state: Any = None) -> Any:
        """Create a Kalman-driven signal ready for sizing."""
        raise NotImplementedError("Kalman strategy logic is not implemented yet.")
