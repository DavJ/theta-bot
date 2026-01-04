from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from spot_bot.portfolio import compute_target_position
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.strategies.mean_reversion import MeanReversionStrategy


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    positions: pd.Series
    metrics: Dict[str, float]


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak.replace(0, np.nan)
    return float(drawdown.min())


def _build_regime_features(ohlcv_df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    close = ohlcv_df["close"].astype(float)
    returns = close.pct_change().fillna(0.0)
    rolling_mean = returns.rolling(window, min_periods=1).mean()
    rolling_std = returns.rolling(window, min_periods=1).std(ddof=0).fillna(0.0)
    regime_features = pd.DataFrame(
        {"S": rolling_mean, "C": rolling_std, "rv": rolling_std},
        index=ohlcv_df.index,
    )
    return regime_features


def _run_backtest_core(
    ohlcv_df: pd.DataFrame,
    strategy: MeanReversionStrategy,
    regime_engine: Optional[RegimeEngine],
    fee_rate: float,
    max_exposure: float,
    initial_equity: float,
) -> BacktestResult:
    if "close" not in ohlcv_df.columns:
        raise ValueError("ohlcv_df must contain a 'close' column.")
    prices = ohlcv_df["close"].astype(float)
    if len(prices) < 2:
        empty_series = pd.Series(dtype=float)
        return BacktestResult(
            equity_curve=empty_series,
            positions=empty_series,
            metrics={"final_return": 0.0, "max_drawdown": 0.0, "time_in_market": 0.0, "turnover": 0.0},
        )

    equity = float(initial_equity)
    position = 0.0
    equity_curve = []
    position_history = []
    exposure_history = []
    timestamps = []
    turnover_value = 0.0

    for i in range(len(prices) - 1):
        window_df = ohlcv_df.iloc[: i + 1]
        intent = strategy.generate_intent(window_df[["close"]])

        risk_state = "ON"
        risk_budget = 1.0
        if regime_engine is not None:
            regime_features = _build_regime_features(window_df)
            decision = regime_engine.decide(regime_features)
            risk_state = decision.risk_state
            risk_budget = decision.risk_budget

        target_position = compute_target_position(
            equity_usdt=equity,
            price=prices.iloc[i],
            desired_exposure=intent.desired_exposure,
            risk_budget=risk_budget,
            max_exposure=max_exposure,
            risk_state=risk_state,
        )

        trade = target_position - position
        turnover_value += abs(trade) * prices.iloc[i]
        fee = abs(trade) * prices.iloc[i] * fee_rate
        equity -= fee

        exposure_fraction = 0.0
        if equity > 0:
            exposure_fraction = min(1.0, abs(target_position) * prices.iloc[i] / equity)
        exposure_history.append(exposure_fraction)

        position = target_position
        next_price = prices.iloc[i + 1]
        pnl = position * (next_price - prices.iloc[i])
        equity += pnl

        equity_curve.append(equity)
        position_history.append(position)
        timestamps.append(prices.index[i + 1])

    equity_series = pd.Series(equity_curve, index=timestamps)
    position_series = pd.Series(position_history, index=timestamps)
    time_in_market = float(np.mean(exposure_history)) if exposure_history else 0.0
    turnover = turnover_value / initial_equity if initial_equity else 0.0

    metrics = {
        "final_return": float(equity_series.iloc[-1] / initial_equity - 1.0) if not equity_series.empty else 0.0,
        "max_drawdown": _max_drawdown(equity_series),
        "time_in_market": time_in_market,
        "turnover": float(turnover),
    }

    return BacktestResult(equity_curve=equity_series, positions=position_series, metrics=metrics)


def run_mean_reversion_backtests(
    ohlcv_df: pd.DataFrame,
    fee_rate: float = 0.0005,
    max_exposure: float = 1.0,
    initial_equity: float = 1000.0,
    regime_config: Optional[Dict] = None,
    strategy: Optional[MeanReversionStrategy] = None,
) -> Dict[str, BacktestResult]:
    """
    Run mean reversion backtests with and without risk gating.

    Returns a mapping with 'baseline' (no gating) and 'gated' (RegimeEngine).
    """
    mr_strategy = strategy or MeanReversionStrategy()
    baseline = _run_backtest_core(
        ohlcv_df=ohlcv_df,
        strategy=mr_strategy,
        regime_engine=None,
        fee_rate=fee_rate,
        max_exposure=max_exposure,
        initial_equity=initial_equity,
    )

    regime_engine = RegimeEngine(regime_config or {})
    gated = _run_backtest_core(
        ohlcv_df=ohlcv_df,
        strategy=mr_strategy,
        regime_engine=regime_engine,
        fee_rate=fee_rate,
        max_exposure=max_exposure,
        initial_equity=initial_equity,
    )

    return {"baseline": baseline, "gated": gated}
