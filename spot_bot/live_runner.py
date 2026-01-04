"""
Live orchestration for Spot Bot 2.0.

Streams live data into the feature pipeline and strategy, applies gating and
sizing, and routes intents through the execution engine with position tracking.
"""

from typing import Any

from .data_providers import LiveDataProvider
from .feature_pipeline import FeaturePipeline
from .regime_engine import RegimeEngine
from .strategies.base import Strategy
from .position_manager import PositionManager
from .sizer import PositionSizer
from .execution_engine import ExecutionEngine


class LiveRunner:
    """Runs the live pipeline end-to-end in long/flat mode."""

    def __init__(
        self,
        data_provider: LiveDataProvider,
        feature_pipeline: FeaturePipeline,
        regime_engine: RegimeEngine,
        strategy: Strategy,
        position_manager: PositionManager,
        sizer: PositionSizer,
        execution_engine: ExecutionEngine,
    ) -> None:
        self.data_provider = data_provider
        self.feature_pipeline = feature_pipeline
        self.regime_engine = regime_engine
        self.strategy = strategy
        self.position_manager = position_manager
        self.sizer = sizer
        self.execution_engine = execution_engine

    def run(self) -> Any:
        """Start consuming live data and orchestrate decisions."""
        raise NotImplementedError("Live orchestration is not implemented yet.")
