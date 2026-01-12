"""
Core trading engine: single step execution logic used across all modes.

This module implements THE single source of truth for one bar step.
All runtimes (live/paper/replay/backtest/fast_backtest) must call these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from spot_bot.core.cost_model import compute_cost_per_turnover
from spot_bot.core.hysteresis import (
    apply_hysteresis,
    compute_hysteresis_threshold,
    compute_return_threshold,
)
from spot_bot.core.portfolio import apply_fill, compute_equity, compute_exposure
from spot_bot.core.trade_planner import plan_trade
from spot_bot.core.types import (
    ExecutionResult,
    MarketBar,
    PortfolioState,
    StrategyOutput,
    TradePlan,
)


class Strategy(Protocol):
    """Protocol for strategy interface."""

    def generate_intent(self, features_df: pd.DataFrame) -> Any:
        """Generate trading intent from features."""
        ...


@dataclass
class EngineParams:
    """Parameters for engine execution."""

    fee_rate: float = 0.001
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    hyst_k: float = 5.0
    hyst_floor: float = 0.02
    hyst_mode: str = "exposure"  # "exposure" or "zscore"
    k_vol: float = 0.5
    edge_bps: float = 5.0
    max_delta_e_min: float = 0.3
    alpha_floor: float = 6.0
    alpha_cap: float = 6.0
    vol_hyst_mode: str = "increase"
    min_notional: float = 10.0
    step_size: Optional[float] = None
    min_usdt_reserve: float = 0.0
    max_notional_per_trade: Optional[float] = None
    allow_short: bool = False
    debug: bool = False
    hyst_conf_k: float = 0.0  # Confidence-based hysteresis adjustment (0 = disabled)
    min_profit_bps: float = 5.0  # Minimum profit buffer in basis points


def run_step(
    bar: MarketBar,
    features_df: pd.DataFrame,
    portfolio: PortfolioState,
    strategy: Strategy,
    params: EngineParams,
    rv_current: float,
    rv_ref: float,
) -> Tuple[TradePlan, StrategyOutput, Dict[str, Any]]:
    """
    Execute one trading step: strategy -> hysteresis -> trade planning.

    This is the single unified step function used by all modes.
    It does NOT execute trades or simulate fills - only plans.

    Args:
        bar: Current market bar (OHLCV)
        features_df: Features DataFrame (window for strategy)
        portfolio: Current portfolio state
        strategy: Strategy implementing generate_intent
        params: Engine parameters (fees, hysteresis, guards)
        rv_current: Current realized volatility
        rv_ref: Reference realized volatility (e.g., median of last 500)

    Returns:
        Tuple of (TradePlan, StrategyOutput, diagnostics)
        - TradePlan: Planned trade with action, deltas, notional
        - StrategyOutput: Raw strategy output for logging
        - diagnostics: Additional diagnostic info

    Process:
    1. Call strategy.generate_intent(features_df) -> raw target_exposure
    2. Compute cost from cost_model
    3. Compute delta_e_min from hysteresis
    4. Apply hysteresis to target_exposure
    5. Call trade_planner to get TradePlan (with rounding, guards)
    6. Return plan (no execution here)
    """
    # Step 1: Strategy generates intent
    intent = strategy.generate_intent(features_df)

    # Extract target_exposure from intent
    # Intent may be Intent dataclass or StrategyOutput
    if hasattr(intent, "desired_exposure"):
        target_exposure_raw = float(intent.desired_exposure)
        diagnostics_strategy = getattr(intent, "diagnostics", {})
    elif hasattr(intent, "target_exposure"):
        target_exposure_raw = float(intent.target_exposure)
        diagnostics_strategy = getattr(intent, "diagnostics", {})
    else:
        # Fallback: assume intent is a number
        target_exposure_raw = float(intent)
        diagnostics_strategy = {}

    # Step 2: Compute cost per turnover
    cost = compute_cost_per_turnover(
        fee_rate=params.fee_rate,
        slippage_bps=params.slippage_bps,
        spread_bps=params.spread_bps,
    )

    # Step 3: Compute hysteresis threshold with diagnostics
    result = compute_hysteresis_threshold(
        rv_current=rv_current,
        rv_ref=rv_ref,
        fee_rate=params.fee_rate,
        slippage_bps=params.slippage_bps,
        spread_bps=params.spread_bps,
        hyst_k=params.hyst_k,
        hyst_floor=params.hyst_floor,
        k_vol=params.k_vol,
        edge_bps=params.edge_bps,
        max_delta_e_min=params.max_delta_e_min,
        alpha_floor=params.alpha_floor,
        alpha_cap=params.alpha_cap,
        vol_hyst_mode=params.vol_hyst_mode,
        return_diagnostics=True,
    )
    delta_e_min, hyst_diagnostics = result
    
    # Step 3.1: Compute return threshold for limit pricing and sell guard
    return_threshold = compute_return_threshold(
        fee_rate=params.fee_rate,
        spread_bps=params.spread_bps,
        slippage_bps=params.slippage_bps,
        edge_bps=params.edge_bps,
        min_profit_bps=params.min_profit_bps,
        rv_current=rv_current,
        rv_ref=rv_ref,
        k_vol=params.k_vol,
        vol_hyst_mode=params.vol_hyst_mode,
    )
    
    # Step 3.5: Apply confidence-based hysteresis adjustment (if enabled)
    if params.hyst_conf_k > 0:
        confidence = float(diagnostics_strategy.get("confidence", 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        delta_e_min = delta_e_min * (1.0 + params.hyst_conf_k * (1.0 - confidence))
    
    # Extract z-scores if available for zscore mode
    # Note: For zscore mode to work properly, strategy must provide zscore in diagnostics
    current_zscore = 0.0  # Default: no current state zscore available
    target_zscore = 0.0
    
    # If zscore mode is requested, verify that zscore is available
    if params.hyst_mode == "zscore":
        if "zscore" not in diagnostics_strategy:
            raise RuntimeError(
                "hyst_mode=zscore not supported: missing zscore in step context. "
                "Strategy must provide 'zscore' in diagnostics to use zscore hysteresis mode."
            )
        target_zscore = float(diagnostics_strategy["zscore"])
    else:
        # For exposure mode, we don't need zscore (set to 0.0)
        target_zscore = float(diagnostics_strategy.get("zscore", 0.0))
    
    # Step 4: Apply hysteresis
    target_exposure_final, suppressed = apply_hysteresis(
        current_exposure=portfolio.exposure,
        target_exposure=target_exposure_raw,
        delta_e_min=delta_e_min,
        mode=params.hyst_mode,
        current_zscore=current_zscore,
        target_zscore=target_zscore,
    )
    
    # Calculate delta_e for diagnostics
    delta_e = abs(target_exposure_raw - portfolio.exposure)

    if params.debug:
        print(
            f"DBG hyst: edge_bps={params.edge_bps} k_vol={params.k_vol} "
            f"rv_cur={rv_current:.6g} rv_ref={rv_ref:.6g} "
            f"delta_e_min={delta_e_min:.6g} return_thr={return_threshold:.6g} "
            f"hyst_raw={hyst_diagnostics.get('hyst_raw', 0):.6g} "
            f"floor_bind={hyst_diagnostics.get('floor_binding', False)} "
            f"cap_bind={hyst_diagnostics.get('cap_binding', False)} "
            f"cur={portfolio.exposure:.3f} tgt_raw={target_exposure_raw:.3f} "
            f"tgt_final={target_exposure_final:.3f} supp={suppressed}"
        )


    # Step 5: Plan trade
    plan = plan_trade(
        portfolio=portfolio,
        price=bar.close,
        target_exposure=target_exposure_final,
        min_notional=params.min_notional,
        step_size=params.step_size,
        min_usdt_reserve=params.min_usdt_reserve,
        max_notional_per_trade=params.max_notional_per_trade,
        allow_short=params.allow_short,
        return_threshold=return_threshold,
    )
    
    # Compute clamped value for diagnostics (always, regardless of allow_short)
    # This shows what the target would be after long-only clamping
    target_exposure_clamped = max(0.0, min(1.0, target_exposure_raw))
    
    # Check if clamping actually occurred (only relevant when allow_short=False)
    clamped_long_only = False
    if not params.allow_short:
        if target_exposure_raw < 0.0 or target_exposure_raw > 1.0:
            clamped_long_only = True

    # If hysteresis suppressed the trade, update reason
    if suppressed and plan.action == "HOLD":
        plan = TradePlan(
            action="HOLD",
            target_exposure=target_exposure_final,
            target_base=plan.target_base,
            delta_base=0.0,
            notional=0.0,
            exec_price_hint=bar.close,
            reason="hysteresis_suppressed",
            diagnostics={
                **plan.diagnostics,
                "target_exposure_raw": target_exposure_raw,
                "target_exposure_clamped": target_exposure_clamped,
                "delta_e": delta_e,
                "delta_e_min": delta_e_min,
                "suppressed": True,
                "clamped_long_only": clamped_long_only,
            },
            limit_price=None,
            order_type="limit",
        )

    # Build strategy output for logging
    strategy_output = StrategyOutput(
        target_exposure=target_exposure_raw,
        diagnostics=diagnostics_strategy,
    )

    # Diagnostics
    diagnostics = {
        "cost": cost,
        "delta_e_min": delta_e_min,
        "delta_e": delta_e,
        "rv_current": rv_current,
        "rv_ref": rv_ref,
        "target_exposure_raw": target_exposure_raw,
        "target_exposure_clamped": target_exposure_clamped,
        "target_exposure_final": target_exposure_final,
        "hysteresis_suppressed": suppressed,
        "clamped_long_only": clamped_long_only,
        "return_threshold": return_threshold,
        **hyst_diagnostics,  # Include hyst_raw, floor_binding, cap_binding, etc.
    }

    return plan, strategy_output, diagnostics


def simulate_execution(
    plan: TradePlan,
    price: float,
    params: EngineParams,
    bar: Optional[MarketBar] = None,
) -> ExecutionResult:
    """
    Simulate execution of a trade plan.

    This is the single unified simulation logic for paper/replay/backtest modes.

    Args:
        plan: Trade plan from run_step
        price: Execution price (typically bar.close, used as fallback)
        params: Engine parameters (for slippage and fees)
        bar: Optional OHLC bar for limit fill simulation

    Returns:
        ExecutionResult with simulated fill details.

    Limit order simulation (when plan.order_type="limit" and plan.limit_price is set):
        BUY: fills only if bar.low <= limit_price, at limit_price
        SELL: fills only if bar.high >= limit_price, at limit_price
        If bar not provided or limit not touched, returns SKIPPED

    Market order simulation (plan.order_type="market" or no limit_price):
        exec_price = price * (1 + slippage_sign * slippage_bps / 10000)
        where slippage_sign = +1 for BUY, -1 for SELL

    Fee model:
        fee = notional * fee_rate
    """
    if plan.action == "HOLD" or plan.delta_base == 0.0:
        return ExecutionResult(
            filled_base=0.0,
            avg_price=price,
            fee_paid=0.0,
            slippage_paid=0.0,
            status="SKIPPED",
            raw=None,
        )

    # Determine if this is a limit order
    is_limit_order = (
        plan.order_type == "limit"
        and plan.limit_price is not None
        and bar is not None
    )

    exec_price = price
    slippage_paid = 0.0

    if is_limit_order:
        # Limit order simulation using OHLC
        # Defensive check: ensure OHLC data is valid for limit simulation
        if bar.low is None or bar.high is None or not np.isfinite(bar.low) or not np.isfinite(bar.high):
            raise ValueError(
                f"Limit simulation requires valid OHLC data. "
                f"Got bar.low={bar.low}, bar.high={bar.high}. "
                f"Ensure input data includes 'open', 'high', 'low', 'close' columns with valid float values."
            )
        
        limit_price = plan.limit_price
        
        if plan.delta_base > 0:
            # BUY: fill only if bar.low <= limit_price
            if bar.low <= limit_price:
                exec_price = limit_price
                # No slippage for limit fills (we get our price)
                slippage_paid = 0.0
            else:
                # Limit not touched, order not filled
                return ExecutionResult(
                    filled_base=0.0,
                    avg_price=price,
                    fee_paid=0.0,
                    slippage_paid=0.0,
                    status="SKIPPED",
                    raw={"reason": "limit_not_touched", "limit_price": limit_price, "bar_low": bar.low},
                )
        else:
            # SELL: fill only if bar.high >= limit_price
            if bar.high >= limit_price:
                exec_price = limit_price
                # No slippage for limit fills (we get our price)
                slippage_paid = 0.0
            else:
                # Limit not touched, order not filled
                return ExecutionResult(
                    filled_base=0.0,
                    avg_price=price,
                    fee_paid=0.0,
                    slippage_paid=0.0,
                    status="SKIPPED",
                    raw={"reason": "limit_not_touched", "limit_price": limit_price, "bar_high": bar.high},
                )
    else:
        # Market order simulation with slippage
        slippage_sign = 1.0 if plan.delta_base > 0 else -1.0
        slippage_mult = 1.0 + slippage_sign * (params.slippage_bps / 10_000.0)
        exec_price = price * slippage_mult
        # Slippage cost (difference from mid price)
        slippage_paid = abs(exec_price - price) * abs(plan.delta_base)

    # Compute notional and fee
    filled_base = plan.delta_base
    notional = abs(filled_base) * exec_price
    fee_paid = notional * params.fee_rate

    return ExecutionResult(
        filled_base=filled_base,
        avg_price=exec_price,
        fee_paid=fee_paid,
        slippage_paid=slippage_paid,
        status="filled",
        raw={"plan": plan, "price": price, "is_limit": is_limit_order},
    )


def run_step_simulated(
    bar: MarketBar,
    features_df: pd.DataFrame,
    portfolio: PortfolioState,
    strategy: Strategy,
    params: EngineParams,
    rv_current: float,
    rv_ref: float,
) -> Tuple[TradePlan, ExecutionResult, PortfolioState, Dict[str, Any]]:
    """
    Execute one step with simulated execution and portfolio update.

    This is a convenience wrapper for simulated modes (paper/replay/backtest).

    Args:
        Same as run_step

    Returns:
        Tuple of (TradePlan, ExecutionResult, updated_portfolio, diagnostics)

    Process:
    1. Call run_step to get plan
    2. Call simulate_execution to get execution result
    3. Apply fill to portfolio
    4. Return all results
    """
    plan, strategy_output, diagnostics = run_step(
        bar=bar,
        features_df=features_df,
        portfolio=portfolio,
        strategy=strategy,
        params=params,
        rv_current=rv_current,
        rv_ref=rv_ref,
    )

    # Simulate execution
    execution = simulate_execution(plan, bar.close, params, bar=bar)

    # Apply fill to portfolio
    updated_portfolio = apply_fill(portfolio, execution)

    # Merge diagnostics
    diagnostics["strategy_output"] = strategy_output
    diagnostics["execution"] = execution

    return plan, execution, updated_portfolio, diagnostics


__all__ = [
    "Strategy",
    "EngineParams",
    "run_step",
    "simulate_execution",
    "run_step_simulated",
]
