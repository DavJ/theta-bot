"""
Spot Bot 2.0 (long/flat) package scaffolding.

This package wires modular components for data access, feature engineering,
regime detection, strategy selection, position management, sizing, execution,
and orchestration for both backtests and live operation.
"""

from .data_providers import DataProvider, HistoricalDataProvider, LiveDataProvider
from .feature_pipeline import FeaturePipeline
from .regime_engine import RegimeEngine
from .strategies.base import Strategy
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.kalman import KalmanStrategy
from .position_manager import PositionManager
from .sizer import PositionSizer
from .execution_engine import ExecutionEngine
from .backtest_runner import BacktestRunner
from .live_runner import LiveRunner

__all__ = [
    "DataProvider",
    "HistoricalDataProvider",
    "LiveDataProvider",
    "FeaturePipeline",
    "RegimeEngine",
    "Strategy",
    "MeanReversionStrategy",
    "KalmanStrategy",
    "PositionManager",
    "PositionSizer",
    "ExecutionEngine",
    "BacktestRunner",
    "LiveRunner",
]
