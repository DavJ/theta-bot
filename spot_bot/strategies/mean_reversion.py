"""
Mean reversion strategy placeholder for Spot Bot 2.0.

Phase B pairs this with risk gating based on regime classification and
psi_logtime-aware features from the pipeline.
"""

from typing import Any, Optional

from .base import Strategy


class MeanReversionStrategy(Strategy):
    """Generates long/flat intents based on mean reversion signals and gating."""

    def __init__(self, lookback: Optional[int] = None) -> None:
        self.lookback = lookback

    def generate_signal(self, features: Any, regime_state: Any = None) -> Any:
        """Create a gated mean reversion signal."""
        raise NotImplementedError("Mean reversion logic is not implemented yet.")
