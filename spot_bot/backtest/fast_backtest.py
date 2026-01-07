from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

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


def _meanrev_series(close: pd.Series, strategy: MeanReversionStrategy) -> pd.Series:
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


def _kalman_series(close: pd.Series, strategy: KalmanStrategy) -> pd.Series:
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


def _compute_intents(
    features: pd.DataFrame, strategy: Any, risk_budget: pd.Series
) -> pd.Series:
    close = pd.to_numeric(features["close"], errors="coerce")
    if isinstance(strategy, MeanRevDualKalmanStrategy):
        return (
            strategy.generate_series(features, risk_budget, apply_budget=False)
            .reindex(features.index)
            .fillna(0.0)
        )
    if isinstance(strategy, MeanReversionStrategy):
        return _meanrev_series(close, strategy)
    if isinstance(strategy, KalmanStrategy):
        return _kalman_series(close, strategy)

    return pd.Series(0.0, index=features.index, dtype=float)


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
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    df_norm = _normalize_df(df)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df_norm.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {sorted(missing)}")

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

    valid_mask = (
        features["C"].notna()
        & features["S"].notna()
        & features["close"].notna()
        & features["rv"].notna()
    )
    features = features.loc[valid_mask].copy()
    if features.empty:
        raise ValueError("Insufficient data to run backtest.")

    regime_engine = RegimeEngine({})
    risk_state, risk_budget = _compute_risk_series(features, regime_engine)

    strategy: Any
    if strategy_name == "kalman_mr_dual":
        strategy = MeanRevDualKalmanStrategy()
    elif strategy_name == "kalman":
        strategy = KalmanStrategy()
    elif strategy_name == "meanrev":
        strategy = MeanReversionStrategy()
    else:
        class _NullStrategy:
            def generate_intent(self, features_df):
                return Intent(desired_exposure=0.0, reason="none", diagnostics={})

        strategy = _NullStrategy()  # type: ignore

    intent_series = _compute_intents(features, strategy, risk_budget)
    target_exposure = (intent_series * risk_budget).clip(lower=0.0, upper=float(max_exposure))
    target_exposure = target_exposure.where(risk_state == "ON", 0.0)

    timestamps = pd.to_datetime(features[TIMESTAMP_COL], utc=True)
    closes = pd.to_numeric(features["close"], errors="coerce").astype(float).values
    target_exp_vals = target_exposure.reindex(features.index).fillna(0.0).astype(float).values

    usdt = float(initial_usdt)
    btc = 0.0
    peak_equity = usdt
    equity_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    turnover_value = 0.0
    exposures_time: List[float] = []

    for ts, price, tgt_exp in zip(timestamps, closes, target_exp_vals):
        if not np.isfinite(price) or price <= 0.0:
            continue
        equity = usdt + btc * price
        target_btc = equity * tgt_exp / price if equity > 0 else 0.0
        delta_btc = target_btc - btc
        if step_size and step_size > 0:
            step = float(step_size)
            delta_btc = math.copysign(math.floor(abs(delta_btc) / step) * step, delta_btc)
        notional_ref = abs(delta_btc) * price
        action = "HOLD"
        fee_paid = 0.0
        slip_cost = 0.0

        if notional_ref >= min_notional and abs(delta_btc) > 0:
            slip_mult = 1 + (slippage_bps * 1e-4) * (1 if delta_btc > 0 else -1)
            exec_price = price * slip_mult
            notional = abs(delta_btc) * exec_price
            fee_paid = notional * fee_rate
            slip_cost = abs(exec_price - price) * abs(delta_btc)
            turnover_value += notional
            if delta_btc > 0:
                usdt -= notional + fee_paid
                btc += delta_btc
                action = "BUY"
            else:
                usdt += notional - fee_paid
                btc += delta_btc
                action = "SELL"
            trade_rows.append(
                {
                    "timestamp": ts,
                    "side": action.lower(),
                    "price": exec_price,
                    "qty": abs(delta_btc),
                    "fee": fee_paid,
                    "slippage": slip_cost,
                    "notional": notional,
                }
            )

        equity = usdt + btc * price
        peak_equity = max(peak_equity, equity)
        drawdown = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        exposure_now = (btc * price / equity) if equity > 0 else 0.0
        exposures_time.append(abs(exposure_now))

        equity_rows.append(
            {
                "timestamp": ts,
                "close": price,
                "position_btc": btc,
                "equity": equity,
                "drawdown": drawdown,
                "target_exposure": tgt_exp,
                "action": action,
                "bar_state": bar_state,
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)

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
    cagr = float((equity_series.iloc[-1] / initial_usdt) ** (1 / duration_years) - 1) if duration_years > 0 else total_return
    max_dd = float(equity_df["drawdown"].min()) if not equity_df.empty else 0.0
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

    print(
        f"processed bars: {len(equity_df)}, trades: {len(trades_df)}, final equity: {summary['final_equity']:.2f}"
    )

    return equity_df, trades_df, summary

