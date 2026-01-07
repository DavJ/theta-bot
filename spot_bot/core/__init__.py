"""
Core trading engine modules for unified execution across all modes.

This package provides a single source of truth for:
- Trading math (cost model, hysteresis, portfolio)
- Trade planning (rounding, guards, sizing)
- Execution simulation
"""

from spot_bot.core.account import AccountProvider, LiveAccountProvider, SimAccountProvider
from spot_bot.core.cost_model import compute_cost_per_turnover
from spot_bot.core.engine import EngineParams, run_step, run_step_simulated, simulate_execution
from spot_bot.core.executor import Executor, SimExecutor
from spot_bot.core.hysteresis import apply_hysteresis, compute_hysteresis_threshold
from spot_bot.core.legacy_adapter import (
    LegacyStrategyAdapter,
    StepResultFromCore,
    compute_step_with_core,
    compute_step_with_core_full,
)
from spot_bot.core.portfolio import (
    apply_fill,
    compute_equity,
    compute_exposure,
    target_base_from_exposure,
)
from spot_bot.core.rv import compute_rv_ref_scalar, compute_rv_ref_series
from spot_bot.core.trade_planner import plan_trade
from spot_bot.core.types import (
    DecisionContext,
    ExecutionResult,
    MarketBar,
    PortfolioState,
    StrategyOutput,
    TradePlan,
)

__all__ = [
    # Types
    "MarketBar",
    "DecisionContext",
    "StrategyOutput",
    "PortfolioState",
    "TradePlan",
    "ExecutionResult",
    # Cost model
    "compute_cost_per_turnover",
    # Hysteresis
    "compute_hysteresis_threshold",
    "apply_hysteresis",
    # Portfolio
    "compute_equity",
    "compute_exposure",
    "target_base_from_exposure",
    "apply_fill",
    # Trade planning
    "plan_trade",
    # Engine
    "EngineParams",
    "run_step",
    "simulate_execution",
    "run_step_simulated",
    # RV helpers
    "compute_rv_ref_scalar",
    "compute_rv_ref_series",
    # Account providers
    "AccountProvider",
    "SimAccountProvider",
    "LiveAccountProvider",
    # Executors
    "Executor",
    "SimExecutor",
    # Legacy compatibility
    "LegacyStrategyAdapter",
    "StepResultFromCore",
    "compute_step_with_core",
    "compute_step_with_core_full",
]
