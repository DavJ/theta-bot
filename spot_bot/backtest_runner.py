"""
Backtest orchestration for Spot Bot 2.0.

Connects data providers, feature pipeline, regime engine, strategy, sizing, and
execution layers to evaluate long/flat performance offline.
"""

from typing import Any

from .data_providers import DataProvider
from .feature_pipeline import FeaturePipeline
from .regime_engine import RegimeEngine
from .strategies.base import Strategy
from .position_manager import PositionManager
from .sizer import PositionSizer
from .execution_engine import ExecutionEngine


class BacktestRunner:
    """Runs end-to-end backtests with interchangeable components."""

    def __init__(
        self,
        data_provider: DataProvider,
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
        """Execute a full backtest loop across historical data."""
        raise NotImplementedError("Backtest orchestration is not implemented yet.")
