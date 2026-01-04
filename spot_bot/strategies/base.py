"""
Strategy interface for Spot Bot 2.0 long/flat decisions.

Strategies consume features and optional regime context to produce target
positions or intents that downstream sizing and execution can act upon.
"""

from abc import ABC, abstractmethod
from typing import Any


class Strategy(ABC):
    """Abstract long/flat strategy contract."""

    @abstractmethod
    def generate_signal(self, features: Any, regime_state: Any = None) -> Any:
        """
        Produce a desired position or intent given current features and regime.
        Should output an object that can be interpreted by the sizer/position manager.
        """
