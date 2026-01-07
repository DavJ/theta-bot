"""
Helper adapters for integrating core engine with existing run_live code.

These adapters allow gradual migration to the unified core engine while
maintaining backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

# Import directly from modules to avoid circular import
from spot_bot.core.engine import EngineParams, run_step
from spot_bot.core.portfolio import compute_equity, compute_exposure
from spot_bot.core.types import MarketBar, PortfolioState, StrategyOutput, TradePlan
from spot_bot.features import FeatureConfig, compute_features
from spot_bot.portfolio.sizer import compute_target_position
from spot_bot.regime.regime_engine import RegimeEngine


class LegacyStrategyAdapter:
    """Adapter to make old-style strategies work with core engine."""

    def __init__(
        self,
        strategy: Any,
        regime_engine: RegimeEngine,
        max_exposure: float,
    ):
        self.strategy = strategy
        self.regime_engine = regime_engine
        self.max_exposure = max_exposure

    def generate_intent(self, features_df: pd.DataFrame) -> StrategyOutput:
        """
        Generate intent using legacy strategy + regime engine.

        This combines:
        1. Strategy intent generation
        2. Regime decision
        3. Position sizing

        Returns StrategyOutput with target_exposure.
        """
        # Get strategy intent
        intent = self.strategy.generate_intent(features_df)
        desired_exposure = float(getattr(intent, "desired_exposure", 0.0))

        # Get regime decision
        decision = self.regime_engine.decide(features_df)
        risk_budget = float(decision.risk_budget)
        risk_state = decision.risk_state

        # Compute target exposure with risk scaling
        # This mimics the logic from run_live.py and fast_backtest.py
        if risk_state == "ON":
            target_exposure = desired_exposure * risk_budget
            target_exposure = min(target_exposure, self.max_exposure)
        else:
            target_exposure = 0.0

        # Return as StrategyOutput
        diagnostics = getattr(intent, "diagnostics", {})
        diagnostics["risk_budget"] = risk_budget
        diagnostics["risk_state"] = risk_state
        diagnostics["desired_exposure"] = desired_exposure

        return StrategyOutput(
            target_exposure=target_exposure,
            diagnostics=diagnostics,
        )


def compute_step_with_core(
    ohlcv_df: pd.DataFrame,
    feature_cfg: FeatureConfig,
    regime_engine: RegimeEngine,
    strategy: Any,
    max_exposure: float,
    fee_rate: float,
    balances: Dict[str, float],
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
    hyst_k: float = 5.0,
    hyst_floor: float = 0.02,
    min_notional: float = 10.0,
    step_size: Optional[float] = None,
    min_usdt_reserve: float = 0.0,
) -> TradePlan:
    """
    Compute trading step using unified core engine.

    This is a drop-in replacement for the old compute_step function in run_live.py.
    It uses the core engine for all trading logic.

    Args:
        ohlcv_df: OHLCV DataFrame
        feature_cfg: Feature configuration
        regime_engine: Regime engine for risk gating
        strategy: Trading strategy
        max_exposure: Maximum exposure fraction
        fee_rate: Transaction fee rate
        balances: Current balances dict with 'usdt' and 'btc' keys
        slippage_bps: Slippage in basis points
        spread_bps: Spread in basis points
        hyst_k: Hysteresis multiplier
        hyst_floor: Hysteresis floor
        min_notional: Minimum trade notional
        step_size: Quantity rounding step size
        min_usdt_reserve: Minimum USDT reserve to maintain

    Returns:
        TradePlan with trade details

    This function:
    1. Computes features
    2. Wraps strategy with regime adapter
    3. Calls core engine run_step
    4. Returns TradePlan
    """
    # Compute features
    features = compute_features(ohlcv_df, feature_cfg).dropna(subset=["S", "C"])
    if isinstance(features.index, pd.DatetimeIndex):
        features.index = pd.to_datetime(features.index, utc=True)
    if features.empty:
        raise ValueError("Insufficient data to compute features.")

    # Get latest price and create market bar
    latest_price = float(ohlcv_df["close"].iloc[-1])
    latest_ts_pd = pd.to_datetime(features.index[-1], utc=True)
    latest_ts_ms = int(latest_ts_pd.value // 1_000_000)

    bar = MarketBar(
        ts=latest_ts_ms,
        open=latest_price,
        high=latest_price,
        low=latest_price,
        close=latest_price,
        volume=0.0,
    )

    # Create portfolio state from balances
    current_btc = float(balances.get("btc", 0.0))
    current_usdt = float(balances.get("usdt", 0.0))
    equity = compute_equity(current_usdt, current_btc, latest_price)
    exposure = compute_exposure(current_btc, latest_price, equity)

    portfolio = PortfolioState(
        usdt=current_usdt,
        base=current_btc,
        equity=equity,
        exposure=exposure,
    )

    # Wrap strategy with regime adapter
    adapted_strategy = LegacyStrategyAdapter(strategy, regime_engine, max_exposure)

    # Compute rv_current and rv_ref
    rv_series = features["rv"] if "rv" in features.columns else pd.Series(dtype=float)
    rv_series = rv_series.dropna()
    rv_current = float(rv_series.iloc[-1]) if not rv_series.empty else 1e-8
    rv_ref_candidates = rv_series.tail(500)
    rv_ref = (
        float(rv_ref_candidates.median())
        if not rv_ref_candidates.empty
        else float(rv_series.median()) if not rv_series.empty else 1.0
    )
    if rv_ref <= 0.0:
        rv_ref = max(abs(rv_current), 1.0)

    # Create engine params
    params = EngineParams(
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        hyst_k=hyst_k,
        hyst_floor=hyst_floor,
        min_notional=min_notional,
        step_size=step_size,
        min_usdt_reserve=min_usdt_reserve,
        allow_short=False,
    )

    # Run core engine step
    plan, strategy_output, diagnostics = run_step(
        bar=bar,
        features_df=features,
        portfolio=portfolio,
        strategy=adapted_strategy,
        params=params,
        rv_current=rv_current,
        rv_ref=rv_ref,
    )

    return plan


__all__ = [
    "LegacyStrategyAdapter",
    "compute_step_with_core",
]
