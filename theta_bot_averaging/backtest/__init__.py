"""
Vectorized backtesting utilities and performance metrics.
"""

from .engine import run_backtest
from .metrics import compute_metrics

__all__ = ["run_backtest", "compute_metrics"]
