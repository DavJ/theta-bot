"""
Refactored fast backtest using unified core engine.

This module now uses spot_bot/core/engine.py for all trading logic,
ensuring consistency with live/paper/replay modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from spot_bot.core import (
    EngineParams,
    MarketBar,
    PortfolioState,
    run_step_simulated,
)
from spot_bot.core.portfolio import compute_equity, compute_exposure
from spot_bot.core.rv import compute_rv_ref_series
from spot_bot.features import FeatureConfig, compute_features
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.strategies.base import Intent
from spot_bot.strategies.kalman import KalmanStrategy
from spot_bot.strategies.mean_reversion import MeanReversionStrategy
from spot_bot.strategies.meanrev_dual_kalman import MeanRevDualKalmanStrategy

TIMESTAMP_COL = "timestamp"


@dataclass
class BacktestOutputs:
    equity: pd.DataFrame
    trades: pd.DataFrame
    summary: Dict[str, float]


class NullStrategy:
    """Fallback strategy producing zero desired exposure."""

    def generate_intent(self, features_df):
        return Intent(desired_exposure=0.0, reason="none", diagnostics={})


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    timeframe = timeframe.strip().lower()
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if TIMESTAMP_COL not in df.columns:
        df = df.copy()
        df[TIMESTAMP_COL] = df.index
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    df = df.sort_values(TIMESTAMP_COL)
    return df.reset_index(drop=True)


def _compute_risk_series(features: pd.DataFrame, regime_engine: RegimeEngine) -> Tuple[pd.Series, pd.Series]:
    """Compute risk state and budget series from features."""
    s_vals = pd.to_numeric(features["S"], errors="coerce")
    rv_vals = pd.to_numeric(features.get("rv"), errors="coerce")

    risk_state = pd.Series("ON", index=features.index)
    off_mask = s_vals < regime_engine.s_off
    if regime_engine.rv_off is not None:
        off_mask = off_mask | (rv_vals > regime_engine.rv_off)
    risk_state = risk_state.mask(off_mask, "OFF")

    reduce_mask = (risk_state == "ON") & (s_vals < regime_engine.s_on)
    if regime_engine.rv_reduce is not None:
        reduce_mask = reduce_mask | ((risk_state == "ON") & (rv_vals > regime_engine.rv_reduce))
    risk_state = risk_state.mask(reduce_mask, "REDUCE")

    span = max(regime_engine.s_budget_high - regime_engine.s_budget_low, 1e-6)
    budget_raw = (s_vals - regime_engine.s_budget_low) / span
    risk_budget = budget_raw.clip(lower=0.0, upper=1.0)
    if regime_engine.rv_guard is not None and abs(regime_engine.rv_guard) > 1e-12:
        guard = (1.0 - (rv_vals / regime_engine.rv_guard)).clip(lower=0.0, upper=1.0)
        risk_budget = (risk_budget * guard).clip(lower=0.0, upper=1.0)

    return risk_state, risk_budget


def _compute_intents_with_regime(
    features: pd.DataFrame,
    strategy: Any,
    risk_state: pd.Series,
    risk_budget: pd.Series,
    max_exposure: float,
) -> pd.Series:
    """
    Compute intent series applying risk regime gating.

    This generates per-bar intent and applies risk budgets/states,
    which will be passed to the engine for hysteresis and planning.
    """
    close = pd.to_numeric(features["close"], errors="coerce")

    # Generate raw intent from strategy
    if isinstance(strategy, MeanRevDualKalmanStrategy):
        # Dual Kalman generates series directly
        raw_intent = strategy.generate_series(features, risk_budgets=risk_budget, apply_budget=False)
        raw_intent = raw_intent.reindex(features.index).fillna(0.0)
    elif isinstance(strategy, MeanReversionStrategy):
        # Use mean reversion series logic from strategy
        raw_intent = _meanrev_series_from_strategy(close, strategy)
    elif isinstance(strategy, KalmanStrategy):
        # Use Kalman series logic
        raw_intent = _kalman_series_from_strategy(close, strategy)
    else:
        raw_intent = pd.Series(0.0, index=features.index, dtype=float)

    # Apply risk budget and max exposure
    target_exposure = (raw_intent * risk_budget).clip(lower=0.0, upper=float(max_exposure))
    # Turn off when risk state is OFF
    target_exposure = target_exposure.where(risk_state == "ON", 0.0)

    return target_exposure


def _meanrev_series_from_strategy(close: pd.Series, strategy: MeanReversionStrategy) -> pd.Series:
    """Generate mean reversion intent series."""
    prices = close.astype(float)
    ema = prices.ewm(span=strategy.ema_span, adjust=False).mean()
    rolling_std = prices.rolling(strategy.std_lookback).std(ddof=0)

    safe_std = float(np.std(prices.values)) if len(prices) > 1 else 1e-8
    if safe_std <= 0.0:
        safe_std = 1e-8
    fallback_std = prices.expanding().std(ddof=0).fillna(safe_std).replace(0.0, safe_std)
    effective_std = rolling_std.where((rolling_std.notna()) & (rolling_std > 0.0), fallback_std)

    zscore = (prices - ema) / effective_std.replace(0.0, safe_std)
    signal_strength = (-zscore).clip(lower=0.0)

    entry = strategy.entry_z
    full = strategy.full_z
    scale = ((signal_strength.clip(upper=full) - entry) / max(full - entry, 1e-8)).clip(lower=0.0)
    raw_exposure = strategy.min_exposure + (strategy.max_exposure - strategy.min_exposure) * scale
    desired = raw_exposure.where(signal_strength > entry, 0.0).clip(lower=0.0, upper=1.0)
    return desired.reindex(prices.index).fillna(0.0)


def _kalman_series_from_strategy(close: pd.Series, strategy: KalmanStrategy) -> pd.Series:
    """Generate Kalman filter intent series."""
    prices = close.astype(float)
    exposures: List[float] = []
    level = float(prices.iloc[0]) if not prices.empty else 0.0
    trend = 0.0
    P = np.eye(2, dtype=float)
    F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
    Q = np.array([[strategy.params.q_level, 0.0], [0.0, strategy.params.q_trend]], dtype=float)
    H = np.array([1.0, 0.0], dtype=float)
    R = float(strategy.params.r)
    for price in prices:
        x = np.array([level, trend], dtype=float)
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        innovation = float(price) - H @ x_pred
        innovation_var = float(H @ P_pred @ H.T + R)
        if math.isnan(innovation_var) or innovation_var <= 0.0:
            innovation_var = 1e-8
        K = (P_pred @ H) / innovation_var
        x = x_pred + K * innovation
        P = (np.eye(2) - np.outer(K, H)) @ P_pred
        level = float(x[0])
        trend = float(x[1])
        z = (float(price) - level) / math.sqrt(innovation_var) if innovation_var > 0 else 0.0
        exposure_raw = 1.0 / (1.0 + float(np.exp(strategy.params.k * z)))
        exposures.append(min(1.0, max(0.0, exposure_raw)))
    exposures_series = pd.Series(exposures, index=prices.index, dtype=float)
    exposures_series.iloc[: strategy.params.min_bars - 1] = 0.0
    return exposures_series


class StrategyAdapter:
    """Adapter to wrap pre-computed intent series as a strategy."""

    def __init__(self, intent_series: pd.Series):
        self.intent_series = intent_series
        self._current_index = 0

    def generate_intent(self, features_df: pd.DataFrame) -> Intent:
        """Return intent for current bar."""
        if self._current_index < len(self.intent_series):
            exposure = float(self.intent_series.iloc[self._current_index])
        else:
            exposure = 0.0
        return Intent(desired_exposure=exposure, reason="precomputed", diagnostics={})


def run_backtest(
    df: pd.DataFrame,
    timeframe: str,
    strategy_name: str,
    psi_mode: str,
    psi_window: int,
    rv_window: int,
    conc_window: int,
    base: float,
    fee_rate: float,
    slippage_bps: float,
    max_exposure: float,
    initial_usdt: float = 1000.0,
    min_notional: float = 5.0,
    step_size: float | None = None,
    bar_state: str = "closed",
    log: bool = True,
    hyst_k: float = 5.0,
    hyst_floor: float = 0.02,
    spread_bps: float = 0.0,
    k_vol: float = 0.5,
    edge_bps: float = 5.0,
    max_delta_e_min: float = 0.5,
    alpha_floor: float = 6.0,
    alpha_cap: float = 6.0,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Run fast backtest using unified core engine.

    This is the refactored version that eliminates duplicated trading logic
    by using spot_bot.core.engine for all decisions.
    """
    # Normalize and validate input
    df_norm = _normalize_df(df)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df_norm.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {sorted(missing)}")

    # Compute features once
    feat_cfg = FeatureConfig(
        base=base,
        rv_window=rv_window,
        conc_window=conc_window,
        psi_mode=psi_mode,
        psi_window=psi_window,
    )
    features = compute_features(df_norm, feat_cfg)
    features["close"] = pd.to_numeric(df_norm["close"], errors="coerce").values
    features[TIMESTAMP_COL] = pd.to_datetime(features.index, utc=True)

    # Filter valid rows
    valid_mask = (
        features["C"].notna()
        & features["S"].notna()
        & features["close"].notna()
        & features["rv"].notna()
    )
    features = features.loc[valid_mask].copy()
    if features.empty:
        raise ValueError("Insufficient data to run backtest.")

    # Compute risk regime series
    regime_engine = RegimeEngine({})
    risk_state, risk_budget = _compute_risk_series(features, regime_engine)

    # Instantiate strategy
    strategy_obj: Any
    if strategy_name == "kalman_mr_dual":
        strategy_obj = MeanRevDualKalmanStrategy()
    elif strategy_name == "kalman":
        strategy_obj = KalmanStrategy()
    elif strategy_name == "meanrev":
        strategy_obj = MeanReversionStrategy()
    else:
        strategy_obj = NullStrategy()

    # Compute intent series with regime gating
    intent_series = _compute_intents_with_regime(
        features, strategy_obj, risk_state, risk_budget, max_exposure
    )

    # Compute rv_ref series using centralized helper
    rv_series = features["rv"].fillna(0.0)
    rv_ref_series = compute_rv_ref_series(rv_series, window=500)

    # Initialize engine params
    engine_params = EngineParams(
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        hyst_k=hyst_k,
        hyst_floor=hyst_floor,
        k_vol=k_vol,
        edge_bps=edge_bps,
        max_delta_e_min=max_delta_e_min,
        alpha_floor=alpha_floor,
        alpha_cap=alpha_cap,
        min_notional=min_notional,
        step_size=step_size,
        allow_short=False,
        debug=False  # Keep debug False - do not enable
    )

    # Initialize portfolio
    portfolio = PortfolioState(
        usdt=float(initial_usdt),
        base=0.0,
        equity=float(initial_usdt),
        exposure=0.0,
    )

    # Run backtest using core engine
    timestamps = pd.to_datetime(features[TIMESTAMP_COL], utc=True)
    closes = pd.to_numeric(features["close"], errors="coerce").astype(float).values

    equity_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    peak_equity = portfolio.equity
    exposures_time: List[float] = []

    for i, (ts, price, target_exp, rv_current, rv_ref) in enumerate(
        zip(timestamps, closes, intent_series, rv_series, rv_ref_series)
    ):
        if not np.isfinite(price) or price <= 0.0:
            continue

        # Recompute portfolio state at current bar's price for consistent exposure
        # This ensures exposure/equity are computed at the current price, not the
        # previous fill price, which is critical for correct hysteresis behavior.
        equity = compute_equity(portfolio.usdt, portfolio.base, price)
        exposure = compute_exposure(portfolio.base, price, equity)
        portfolio = PortfolioState(
            usdt=portfolio.usdt,
            base=portfolio.base,
            equity=equity,
            exposure=exposure,
        )

        # Create market bar
        bar = MarketBar(
            ts=int(ts.value // 1_000_000),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=0.0,
        )

        # Create minimal features_df for strategy adapter
        # Strategy adapter will return pre-computed target_exp
        features_window = pd.DataFrame({"close": [price]})
        adapter = StrategyAdapter(pd.Series([target_exp]))

        # Run single step with core engine
        try:
            plan, execution, portfolio, diagnostics = run_step_simulated(
                bar=bar,
                features_df=features_window,
                portfolio=portfolio,
                strategy=adapter,
                params=engine_params,
                rv_current=float(rv_current),
                rv_ref=float(rv_ref),
            )
        except Exception as e:
            raise RuntimeError(f"Backtest failed at i={i}, ts={ts}, price={price}: {e}") from e

        # Record trade if executed
        action = plan.action
        if execution.status == "filled" and abs(execution.filled_base) > 0:
            trade_rows.append(
                {
                    "timestamp": ts,
                    "side": "buy" if execution.filled_base > 0 else "sell",
                    "price": execution.avg_price,
                    "qty": abs(execution.filled_base),
                    "fee": execution.fee_paid,
                    "slippage": execution.slippage_paid,
                    "notional": abs(execution.filled_base) * execution.avg_price,
                }
            )

        # Record equity
        peak_equity = max(peak_equity, portfolio.equity)
        drawdown = (portfolio.equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        exposures_time.append(abs(portfolio.exposure))

        equity_rows.append(
            {
                "timestamp": ts,
                "close": price,
                "position_btc": portfolio.base,
                "equity": portfolio.equity,
                "drawdown": drawdown,
                "target_exposure": target_exp,
                "action": action,
                "bar_state": bar_state,
            }
        )

    # Build output DataFrames
    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)

    # Compute summary metrics
    equity_series = equity_df.set_index("timestamp")["equity"] if not equity_df.empty else pd.Series(dtype=float)
    returns = equity_series.pct_change().dropna()
    delta = _timeframe_to_timedelta(timeframe)
    period_sec = delta.total_seconds()
    periods_per_year = (365 * 24 * 3600) / period_sec if period_sec > 0 else 0.0
    vol = float(returns.std(ddof=0) * math.sqrt(periods_per_year)) if not returns.empty and periods_per_year > 0 else 0.0
    sharpe = float(returns.mean() * math.sqrt(periods_per_year) / vol) if vol > 0 else 0.0
    total_return = float(equity_series.iloc[-1] / initial_usdt - 1.0) if not equity_series.empty else 0.0
    duration_years = (
        (equity_df["timestamp"].iloc[-1] - equity_df["timestamp"].iloc[0]).total_seconds() / (365 * 24 * 3600)
        if len(equity_df) > 1
        else 0.0
    )
    duration_threshold = 1e-6
    if duration_years > duration_threshold:
        cagr = float((equity_series.iloc[-1] / initial_usdt) ** (1 / duration_years) - 1)
    else:
        cagr = total_return
    max_dd = float(equity_df["drawdown"].min()) if not equity_df.empty else 0.0

    turnover_value = sum(t["notional"] for t in trade_rows)
    turnover = float(turnover_value / initial_usdt) if initial_usdt else 0.0
    time_in_market = float(np.mean(exposures_time)) if exposures_time else 0.0

    summary = {
        "final_equity": float(equity_series.iloc[-1]) if not equity_series.empty else float(initial_usdt),
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "maxDD": max_dd,
        "trades_count": float(len(trades_df)),
        "turnover": turnover,
        "time_in_market": time_in_market,
    }

    if log:
        print(
            f"processed bars: {len(equity_df)}, trades: {len(trades_df)}, final equity: {summary['final_equity']:.2f}"
        )

    return equity_df, trades_df, summary
