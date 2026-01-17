#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
import os
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Tuple

if __package__ is None and __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yaml

from spot_bot.backtest import run_backtest
from spot_bot.core.engine import EngineParams, simulate_execution
from spot_bot.core.legacy_adapter import plan_from_live_inputs
from spot_bot.core.portfolio import apply_fill, apply_live_fill_to_balances, compute_equity, compute_exposure
from spot_bot.core.types import ExecutionResult, PortfolioState
from spot_bot.features import FeatureConfig, compute_features
from spot_bot.live import PaperBroker
from spot_bot.persist import SQLiteLogger
from spot_bot.portfolio.sizer import compute_target_position
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.strategies.base import Intent
from spot_bot.strategies.kalman import KalmanStrategy
from spot_bot.strategies.mean_reversion import MeanReversionStrategy
from spot_bot.strategies.meanrev_dual_kalman import MeanRevDualKalmanStrategy


def _str_to_bool(v: str) -> bool:
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


DEFAULT_FEATURE_CFG = FeatureConfig()
CSV_OUTPUT_COLUMNS = [
    "timestamp",
    "last_closed_ts",
    "bar_state",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rv",
    "C",
    "psi",
    "C_int",
    "S",
    "risk_state",
    "risk_budget",
    "intent_exposure",
    "target_exposure",
    "action",
]
REQUIRED_BAR_COLUMNS = {"open", "high", "low", "close", "volume"}
CSV_OUT_MODES = ("latest", "features")
LAST_CLOSED_KV_KEY = "last_closed_ts"


class NullStrategy:
    def generate_intent(self, features_df):
        return Intent(desired_exposure=0.0, reason="strategy none", diagnostics={})


class LoopStateStore:
    """Persist loop state across runs using SQLite kv store or JSON sidecar."""

    def __init__(self, logger: Optional[SQLiteLogger] = None, path: Optional[pathlib.Path] = None) -> None:
        self.logger = logger
        self.path = pathlib.Path(path) if path else None

    def load_last_closed_ts(self) -> Optional[int]:
        if self.logger:
            value = self.logger.get_kv(LAST_CLOSED_KV_KEY)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    return None
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                value = data.get(LAST_CLOSED_KV_KEY)
                if value is not None:
                    return int(value)
            except Exception:
                return None
        return None

    def save_last_closed_ts(self, ts_ms: Optional[int]) -> None:
        if ts_ms is None:
            return
        ts_int = int(ts_ms)
        if self.logger:
            self.logger.set_kv(LAST_CLOSED_KV_KEY, ts_int)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({LAST_CLOSED_KV_KEY: ts_int}))


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    match = re.fullmatch(r"(\d+)([mhd])", timeframe.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    value = int(match.group(1))
    unit = match.group(2).lower()
    unit_map = {"m": "minutes", "h": "hours", "d": "days"}
    return pd.Timedelta(**{unit_map[unit]: value})


def latest_closed_ohlcv(df: pd.DataFrame, timeframe: str, now: Optional[datetime] = None) -> pd.DataFrame:
    """
    Return df possibly truncated so the last row is a CLOSED bar.
    Assumes df is time-ordered and timestamps are UTC-aware or UTC-naive consistently.
    For live data, the last bar may be in-progress; drop it if it's not closed yet.
    """
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df

    tf_delta = _timeframe_to_timedelta(timeframe)
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    now_utc = now_ts.tz_convert("UTC") if now_ts.tzinfo else now_ts.tz_localize("UTC")

    if isinstance(df.index, pd.DatetimeIndex):
        last_ts_raw = df.index[-1]
    elif "timestamp" in df.columns:
        last_ts_raw = df["timestamp"].iloc[-1]
    else:
        raise ValueError("DataFrame must have a datetime index or 'timestamp' column.")

    last_ts = pd.to_datetime(last_ts_raw, utc=True)
    last_close = last_ts + tf_delta

    if now_utc < last_close:
        return df.iloc[:-1]
    return df


def _to_epoch_ms(ts: Any) -> int:
    return int(pd.to_datetime(ts, utc=True).value // 1_000_000)


def _to_utc_timestamp(ts: Any) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True)


def _load_config(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _load_cached(cache_path: Optional[str]) -> Optional[pd.DataFrame]:
    if cache_path and pathlib.Path(cache_path).exists():
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        return df
    return None


def _fetch_http(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    from download_market_data import download_binance_data

    df = download_binance_data(symbol=symbol.replace("/", ""), interval=timeframe, limit=limit)
    df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_ccxt(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    from theta_features.binance_data import fetch_ohlcv_ccxt

    df = fetch_ohlcv_ccxt(symbol=symbol, timeframe=timeframe, limit_total=limit)
    df = df.rename(columns={"dt": "timestamp"})
    df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    return df[["open", "high", "low", "close", "volume"]]


def _load_or_fetch(mode: str, symbol: str, timeframe: str, limit: int, cache: Optional[str]) -> pd.DataFrame:
    cached = _load_cached(cache)
    if cached is not None:
        return cached
    if mode == "live":
        df = _fetch_ccxt(symbol, timeframe, limit)
    else:
        try:
            df = _fetch_http(symbol, timeframe, limit)
        except Exception:
            df = _fetch_ccxt(symbol, timeframe, limit)
    if cache:
        pathlib.Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_csv(cache, index=False)
    return df


def _latest_bar_row(df: pd.DataFrame) -> tuple[pd.Series, pd.Timestamp]:
    latest_bar = df.tail(1).reset_index()
    if latest_bar.empty:
        raise ValueError("Latest bar is missing.")
    bar_row = latest_bar.iloc[0]
    bar_cols = set(latest_bar.columns)
    missing_cols = [c for c in REQUIRED_BAR_COLUMNS if c not in bar_cols]
    if missing_cols:
        raise ValueError(f"Latest bar missing columns {missing_cols}; available columns: {sorted(bar_cols)}")
    if "timestamp" in bar_cols:
        ts_col = "timestamp"
    elif "index" in bar_cols:
        ts_col = "index"
    else:
        raise ValueError(f"Latest bar is missing a timestamp/index column. Available columns: {sorted(bar_cols)}")
    ts_value = pd.to_datetime(bar_row[ts_col], utc=True)
    return bar_row, ts_value


def _prepare_trade_data(df: pd.DataFrame, timeframe: str, trade_on: str) -> tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Timestamp], str]:
    if trade_on == "bar_close":
        truncated = latest_closed_ohlcv(df, timeframe)
        if truncated.empty:
            return pd.DataFrame(), None, None, "closed"
        bar_row, ts_value = _latest_bar_row(truncated)
        return truncated, bar_row, ts_value, "closed"

    truncated = df
    if truncated.empty:
        return truncated, None, None, "live"
    bar_row, ts_value = _latest_bar_row(truncated)
    tf_delta = _timeframe_to_timedelta(timeframe)
    expected_close = _to_utc_timestamp(ts_value) + tf_delta
    now_utc = _to_utc_timestamp(datetime.now(timezone.utc))
    bar_state = "closed" if now_utc >= expected_close else "live"
    return truncated, bar_row, _to_utc_timestamp(ts_value), bar_state


def _apply_live_fill_to_balances(
    balances: Dict[str, float],
    side: str,
    qty: float,
    price: float,
    fee_rate: float,
) -> float:
    """
    Apply live exchange fill to local balances dictionary.
    
    This is a PURE DELEGATION to core.portfolio.apply_live_fill_to_balances.
    No math is performed in run_live.py - all computation is in core.
    
    Args:
        balances: Local balances dict with 'usdt' and 'btc' keys
        side: 'buy' or 'sell'
        qty: Quantity filled (always positive)
        price: Fill price
        fee_rate: Fee rate
    
    Returns:
        Fee paid
    """
    return apply_live_fill_to_balances(balances, side, qty, price, fee_rate)


@dataclass
class StepResult:
    ts: pd.Timestamp
    close: float
    decision: Any
    intent: Any
    target_exposure: float
    target_btc: float
    delta_btc: float
    equity: Dict[str, float]
    execution: Optional[Dict[str, Any]]
    features_row: pd.Series


def compute_step(
    ohlcv_df: pd.DataFrame,
    feature_cfg: FeatureConfig,
    regime_engine: RegimeEngine,
    strategy: MeanReversionStrategy,
    max_exposure: float,
    fee_rate: float,
    balances: Dict[str, float],
    mode: str = "dryrun",
    broker: Optional[PaperBroker] = None,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
    hyst_k: float = 5.0,
    hyst_floor: float = 0.02,
    hyst_mode: str = "exposure",
    min_notional: float = 10.0,
    step_size: Optional[float] = None,
    min_usdt_reserve: float = 0.0,
    k_vol: float = 0.5,
    edge_bps: float = 5.0,
    max_delta_e_min: float = 0.3,
    alpha_floor: float = 6.0,
    alpha_cap: float = 6.0,
    vol_hyst_mode: str = "increase",
    min_profit_bps: float = 5.0,
) -> StepResult:
    """
    Compute trading step using unified core engine.
    
    This is a thin wrapper around plan_from_live_inputs that delegates
    all trading math to the core engine. No cost/hysteresis/rounding is computed here.
    
    All trading decisions (target exposure, hysteresis, costs, rounding, guards, fills)
    are computed ONLY by spot_bot/core. This function is a pure orchestrator.
    """
    # Call core adapter - this is the ONLY place where trade planning happens
    result = plan_from_live_inputs(
        ohlcv_df=ohlcv_df,
        feature_cfg=feature_cfg,
        regime_engine=regime_engine,
        strategy=strategy,
        max_exposure=max_exposure,
        fee_rate=fee_rate,
        balances=balances,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        hyst_k=hyst_k,
        hyst_floor=hyst_floor,
        hyst_mode=hyst_mode,
        min_notional=min_notional,
        step_size=step_size,
        min_usdt_reserve=min_usdt_reserve,
        k_vol=k_vol,
        edge_bps=edge_bps,
        max_delta_e_min=max_delta_e_min,
        alpha_floor=alpha_floor,
        alpha_cap=alpha_cap,
        vol_hyst_mode=vol_hyst_mode,
        min_profit_bps=min_profit_bps,
    )
    
    # For paper mode, execute the trade
    execution_result = None
    current_btc = result.equity["btc"]
    current_usdt = result.equity["usdt"]
    equity_usdt = result.equity["equity_usdt"]
    
    if mode == "paper" and abs(result.delta_btc) > 0:
        # ALWAYS use core SimExecutor for consistent behavior with fast_backtest
        # The legacy PaperBroker path has been removed to eliminate all math from run_live.py
        params = EngineParams(
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            hyst_k=hyst_k,
            hyst_floor=hyst_floor,
            hyst_mode=hyst_mode,
            k_vol=k_vol,
            edge_bps=edge_bps,
            max_delta_e_min=max_delta_e_min,
            alpha_floor=alpha_floor,
            alpha_cap=alpha_cap,
            vol_hyst_mode=vol_hyst_mode,
            min_notional=min_notional,
            step_size=step_size,
            min_usdt_reserve=min_usdt_reserve,
            min_profit_bps=min_profit_bps,
        )
        core_execution = simulate_execution(result.plan, result.close, params)
        
        # Build execution_result dict for backward compatibility
        if core_execution.status == "filled":
            side = "buy" if core_execution.filled_base > 0 else "sell"
            execution_result = {
                "status": "filled",
                "side": side,
                "qty": abs(core_execution.filled_base),
                "filled_qty": abs(core_execution.filled_base),
                "price": core_execution.avg_price,
                "avg_price": core_execution.avg_price,
                "fee": core_execution.fee_paid,
                "fee_est": core_execution.fee_paid,
            }
            
            # Apply fill to get updated balances
            portfolio = PortfolioState(
                usdt=current_usdt,
                base=current_btc,
                equity=equity_usdt,
                exposure=result.plan.target_exposure if result.plan else 0.0,
            )
            updated = apply_fill(portfolio, core_execution)
            current_btc = updated.base
            current_usdt = updated.usdt
            equity_usdt = updated.equity
            
            # Update broker state if provided (for state persistence)
            if broker:
                broker.set_balances(current_usdt, current_btc)
        else:
            execution_result = {"status": "noop", "side": "hold", "qty": 0.0}
    
    # Return StepResult compatible with existing orchestration
    return StepResult(
        ts=result.ts,
        close=result.close,
        decision=result.decision,
        intent=result.intent,
        target_exposure=result.target_exposure,
        target_btc=result.target_btc,
        delta_btc=result.delta_btc,
        equity={"equity_usdt": equity_usdt, "btc": current_btc, "usdt": current_usdt},
        execution=execution_result,
        features_row=result.features_row,
    )


def _write_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_replay(
    ohlcv_df: pd.DataFrame,
    feature_cfg: FeatureConfig,
    regime_engine: RegimeEngine,
    strategy: MeanReversionStrategy,
    max_exposure: float,
    fee_rate: float,
    slippage_bps: float,
    spread_bps: float,
    hyst_k: float,
    hyst_floor: float,
    hyst_mode: str,
    min_notional: float,
    step_size: Optional[float],
    initial_usdt: float,
    initial_btc: float,
    equity_path: pathlib.Path,
    trades_path: pathlib.Path,
    features_path: Optional[pathlib.Path] = None,
    k_vol: float = 0.5,
    edge_bps: float = 5.0,
    max_delta_e_min: float = 0.3,
    alpha_floor: float = 6.0,
    alpha_cap: float = 6.0,
    vol_hyst_mode: str = "increase",
    min_profit_bps: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    if ohlcv_df is None or ohlcv_df.empty:
        raise ValueError("Replay requires non-empty OHLCV data.")
    if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
        if "timestamp" in ohlcv_df.columns:
            ohlcv_df = ohlcv_df.copy()
            ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"], utc=True)
            ohlcv_df = ohlcv_df.set_index("timestamp")
        else:
            raise ValueError("OHLCV data must be indexed by timestamp for replay.")
    ohlcv_df = ohlcv_df.sort_index()

    broker = PaperBroker(
        initial_usdt=initial_usdt,
        initial_btc=initial_btc,
        fee_rate=fee_rate,
        min_notional=min_notional,
        step_size=step_size,
    )
    equity_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    features_rows: List[Dict[str, Any]] = []

    last_error_msg: Optional[str] = None
    if len(ohlcv_df) > 10_000:
        print(f"Replay warning: large dataset detected ({len(ohlcv_df)} rows); per-step feature recomputation may be slow.")

    def _extract_execution_field(data: Dict[str, Any], primary: str, secondary: Optional[str], default_value: Any) -> Any:
        value = data.get(primary)
        if value is None and secondary:
            value = data.get(secondary)
        return default_value if value is None else value

    # Replay recomputes features on the expanding history to avoid lookahead; datasets are expected to be modest.
    for i in range(len(ohlcv_df)):
        df_slice = ohlcv_df.iloc[: i + 1]
        try:
            result = compute_step(
                ohlcv_df=df_slice,
                feature_cfg=feature_cfg,
                regime_engine=regime_engine,
                strategy=strategy,
                max_exposure=max_exposure,
                fee_rate=fee_rate,
                balances=broker.balances(),
                mode="paper",
                broker=broker,
                slippage_bps=slippage_bps,
                spread_bps=spread_bps,
                hyst_k=hyst_k,
                hyst_floor=hyst_floor,
                hyst_mode=hyst_mode,
                k_vol=k_vol,
                edge_bps=edge_bps,
                max_delta_e_min=max_delta_e_min,
                alpha_floor=alpha_floor,
                alpha_cap=alpha_cap,
                vol_hyst_mode=vol_hyst_mode,
                min_profit_bps=min_profit_bps,
            )
        except ValueError as exc:
            msg = str(exc)
            if msg != last_error_msg:
                print(f"Replay skipping step {i}: {msg}")
                last_error_msg = msg
            continue

        execution_result = result.execution
        action = "HOLD"
        if abs(result.delta_btc) > 0:
            action = "BUY" if result.delta_btc > 0 else "SELL"
            if execution_result and execution_result.get("status") not in ("filled", "partial"):
                action = f"{action}-rejected"

        equity_rows.append(
            {
                "timestamp": result.ts,
                "close": result.close,
                "equity_usdt": result.equity["equity_usdt"],
                "btc": result.equity["btc"],
                "usdt": result.equity["usdt"],
                "action": action,
                "target_exposure": result.target_exposure,
            }
        )

        if execution_result:
            # Prefer filled quantity; fall back to requested quantity if fill metadata is absent.
            qty_raw = _extract_execution_field(execution_result, "filled_qty", "qty", 0.0)
            qty = float(qty_raw or 0.0)
            if qty > 0 and execution_result.get("status") in ("filled", "partial"):
                # Prefer explicit execution price, then average fill price, and finally last close as a fallback.
                price_raw = _extract_execution_field(execution_result, "price", "avg_price", result.close)
                price = float(price_raw)
                fee_raw = _extract_execution_field(execution_result, "fee", "fee_est", 0.0)
                fee = float(fee_raw)
                notional = qty * price
                side = "buy" if result.delta_btc > 0 else "sell"
                cost = notional + fee if side == "buy" else notional - fee
                trade_rows.append(
                    {
                        "timestamp": result.ts,
                        "side": side,
                        "size": qty,
                        "price": price,
                        "fee": fee,
                        "cost": cost,
                    }
                )

        if features_path:
            feat_df = _compute_feature_outputs(
                ohlcv_df=df_slice,
                feature_cfg=feature_cfg,
                regime_engine=regime_engine,
                strategy=strategy,
                max_exposure=max_exposure,
                equity_usdt=result.equity["equity_usdt"],
                tail_rows=1,
                latest_action=action,
            )
            if not feat_df.empty:
                features_rows.append(feat_df.tail(1).reset_index(drop=True).iloc[0].to_dict())

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)
    features_df = None
    if features_rows:
        features_df = pd.DataFrame(features_rows)

    _write_csv(equity_df, equity_path)
    _write_csv(trades_df, trades_path)
    if features_path and features_df is not None:
        _write_csv(features_df.reset_index(drop=True), features_path)

    return equity_df, trades_df, features_df


def _compute_feature_outputs(
    ohlcv_df: pd.DataFrame,
    feature_cfg: FeatureConfig,
    regime_engine: RegimeEngine,
    strategy: MeanReversionStrategy,
    max_exposure: float,
    equity_usdt: float,
    tail_rows: Optional[int] = None,
    latest_action: str = "",
) -> pd.DataFrame:
    features = compute_features(ohlcv_df, feature_cfg)
    if isinstance(features.index, pd.DatetimeIndex):
        features.index = pd.to_datetime(features.index, utc=True)
    if features.empty:
        return features.head(0)

    # Add OHLCV columns for completeness
    for col in ("open", "high", "low", "close", "volume"):
        if col in ohlcv_df.columns:
            features[col] = ohlcv_df[col]

    valid = features.dropna(subset=["S", "C"]).copy()
    if valid.empty:
        return valid

    equity = float(equity_usdt or 0.0)

    rv_values = valid["rv"] if "rv" in valid.columns else pd.Series(index=valid.index, dtype=float)
    risk_state_series = pd.Series("ON", index=valid.index)

    off_mask = valid["S"] < regime_engine.s_off
    if regime_engine.rv_off is not None:
        off_mask = off_mask | (rv_values > regime_engine.rv_off)
    risk_state_series = risk_state_series.mask(off_mask, "OFF")

    reduce_mask = risk_state_series.eq("ON") & (valid["S"] < regime_engine.s_on)
    if regime_engine.rv_reduce is not None:
        reduce_mask = reduce_mask | (risk_state_series.eq("ON") & (rv_values > regime_engine.rv_reduce))
    risk_state_series = risk_state_series.mask(reduce_mask, "REDUCE")

    span = max(regime_engine.s_budget_high - regime_engine.s_budget_low, 1e-6)
    budget_raw = (valid["S"] - regime_engine.s_budget_low) / span
    risk_budget_series = budget_raw.clip(lower=0.0, upper=1.0)
    if regime_engine.rv_guard is not None and abs(regime_engine.rv_guard) > 1e-6:
        vol_guard = (1.0 - (rv_values / regime_engine.rv_guard)).clip(lower=0.0, upper=1.0)
        risk_budget_series = (risk_budget_series * vol_guard).clip(lower=0.0, upper=1.0)

    intent_series: pd.Series
    if isinstance(strategy, MeanRevDualKalmanStrategy):
        # Risk budgets are applied below via risk_budget_series to avoid double scaling.
        intent_series = strategy.generate_series(valid, risk_budget_series, apply_budget=False).reindex(valid.index)
        intent_series = intent_series.fillna(0.0)
    else:
        closes = ohlcv_df["close"].astype(float)
        closes_non_na = closes.dropna()
        ema = closes.ewm(span=strategy.ema_span, adjust=False).mean()
        rolling_std = closes.rolling(strategy.std_lookback).std(ddof=0)
        fallback_std = closes.expanding().std(ddof=0).fillna(0.0)
        safe_std = float(closes_non_na.std(ddof=0)) if not closes_non_na.empty else 0.0
        if safe_std <= 0.0:
            safe_std = 1e-8
        fallback_std = fallback_std.where(fallback_std > 0.0, safe_std)
        effective_std = rolling_std.where((rolling_std.notna()) & (rolling_std > 0.0), fallback_std)
        effective_std = effective_std.replace(0.0, safe_std)
        zscore = (closes - ema) / effective_std
        signal_strength = (-zscore).clip(lower=0.0)
        entry = strategy.entry_z
        full = strategy.full_z
        capped_strength = signal_strength.clip(upper=full)
        scale = (capped_strength - entry) / max(full - entry, 1e-8)
        raw_exposure = strategy.min_exposure + (strategy.max_exposure - strategy.min_exposure) * scale
        desired_exposure_series = raw_exposure.where(signal_strength > entry, 0.0).clip(lower=0.0, upper=1.0)
        intent_series = desired_exposure_series.reindex(valid.index).fillna(0.0).clip(lower=0.0, upper=1.0)

    target_exp_series = (intent_series * risk_budget_series).clip(lower=0.0, upper=float(max_exposure))
    target_exp_series = target_exp_series.where(risk_state_series == "ON", 0.0)
    target_btc_series = target_exp_series * equity / valid["close"]

    valid["risk_state"] = risk_state_series
    valid["risk_budget"] = risk_budget_series
    valid["intent_exposure"] = intent_series
    valid["target_exposure"] = target_exp_series.fillna(0.0)
    valid["bar_state"] = "closed"
    valid["last_closed_ts"] = pd.to_datetime(valid.index, utc=True)
    valid["action"] = ""
    if latest_action and not valid.empty:
        valid.loc[valid.index[-1], "action"] = latest_action

    if tail_rows is not None:
        valid = valid.tail(tail_rows)

    return valid


def run_once_on_df(
    ohlcv_df: pd.DataFrame,
    feature_cfg: FeatureConfig,
    regime_engine: RegimeEngine,
    strategy: MeanReversionStrategy,
    max_exposure: float,
    fee_rate: float,
    balances: Dict[str, float],
    mode: str = "dryrun",
    broker: Optional[PaperBroker] = None,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
    hyst_k: float = 5.0,
    hyst_floor: float = 0.02,
    hyst_mode: str = "exposure",
    k_vol: float = 0.5,
    edge_bps: float = 5.0,
    max_delta_e_min: float = 0.3,
    alpha_floor: float = 6.0,
    alpha_cap: float = 6.0,
    vol_hyst_mode: str = "increase",
    min_profit_bps: float = 5.0,
) -> StepResult:
    return compute_step(
        ohlcv_df=ohlcv_df,
        feature_cfg=feature_cfg,
        regime_engine=regime_engine,
        strategy=strategy,
        max_exposure=max_exposure,
        fee_rate=fee_rate,
        balances=balances,
        mode=mode,
        broker=broker,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        hyst_k=hyst_k,
        hyst_floor=hyst_floor,
        hyst_mode=hyst_mode,
        k_vol=k_vol,
        edge_bps=edge_bps,
        max_delta_e_min=max_delta_e_min,
        alpha_floor=alpha_floor,
        alpha_cap=alpha_cap,
        vol_hyst_mode=vol_hyst_mode,
        min_profit_bps=min_profit_bps,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Spot Bot live loop.")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--limit-total", dest="limit_total", type=int, default=2000)
    parser.add_argument("--mode", choices=["dryrun", "paper", "live", "replay", "backtest"], default="dryrun")
    parser.add_argument("--db", type=str, default=None, help="SQLite DB path (required for paper/live).")
    parser.add_argument("--initial-usdt", dest="initial_usdt", type=float, default=1000.0)
    parser.add_argument("--max-exposure", dest="max_exposure", type=float, default=0.3)
    parser.add_argument("--fee-rate", dest="fee_rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=0.0)
    parser.add_argument("--spread-bps", dest="spread_bps", type=float, default=0.0)
    parser.add_argument("--min-notional", dest="min_notional", type=float, default=10.0)
    parser.add_argument("--step-size", dest="step_size", type=float, default=None)
    parser.add_argument("--hyst-k", dest="hyst_k", type=float, default=5.0)
    parser.add_argument("--hyst-floor", dest="hyst_floor", type=float, default=0.02)
    parser.add_argument("--hyst-mode", dest="hyst_mode", choices=["exposure", "zscore"], default="exposure")
    parser.add_argument("--k-vol", dest="k_vol", type=float, default=0.5, help="Volatility multiplier for hysteresis threshold")
    parser.add_argument("--edge-bps", dest="edge_bps", type=float, default=5.0, help="Required edge in basis points")
    parser.add_argument("--max-delta-e-min", dest="max_delta_e_min", type=float, default=0.3, help="Maximum hysteresis threshold cap")
    parser.add_argument("--alpha-floor", dest="alpha_floor", type=float, default=6.0, help="Smoothness parameter for minimum bound")
    parser.add_argument("--alpha-cap", dest="alpha_cap", type=float, default=6.0, help="Smoothness parameter for maximum bound")
    parser.add_argument("--vol-hyst-mode", dest="vol_hyst_mode", type=str, default="increase", choices=["increase", "decrease", "none"], help="Volatility hysteresis mode")
    parser.add_argument("--loop", action="store_true", help="Run in continuous loop mode.")
    parser.add_argument("--poll-seconds", dest="poll_seconds", type=float, default=10.0)
    parser.add_argument("--trade-on", dest="trade_on", choices=["bar_close", "tick"], default="bar_close")
    parser.add_argument("--csv-in", dest="csv_in", type=str, default=None, help="Optional CSV input for offline testing.")
    parser.add_argument("--csv-out", dest="csv_out", type=str, default=None, help="Optional CSV output to export results.")
    parser.add_argument(
        "--csv-out-mode",
        dest="csv_out_mode",
        choices=list(CSV_OUT_MODES),
        default="latest",
        help="CSV export mode: latest (single row) or features (full feature table).",
    )
    parser.add_argument(
        "--csv-out-tail", dest="csv_out_tail", type=int, default=None, help="Optional tail N rows when exporting features."
    )
    parser.add_argument("--csv", type=str, default=None, help="(Deprecated) Use --csv-in/--csv-out instead.")
    parser.add_argument("--cache", type=str, default=None, help="Optional cache file for OHLCV.")
    parser.add_argument("--config", type=str, default=str(pathlib.Path(__file__).parent / "config.yaml"))
    parser.add_argument("--exchange-id", dest="exchange_id", type=str, default="binance",
                        help="CCXT exchange id for live trading (default: binance)")
    parser.add_argument("--api-key", dest="api_key", type=str, default=None,
                        help="API key for live trading (or env BINANCE_API_KEY)")
    parser.add_argument("--api-secret", dest="api_secret", type=str, default=None,
                        help="API secret for live trading (or env BINANCE_API_SECRET)")
    parser.add_argument("--i-understand-live-risk", action="store_true", help="Required to enable live execution.")
    parser.add_argument("--replay-equity-out", dest="replay_equity_out", type=str, default="equity_curve.csv")
    parser.add_argument("--replay-trades-out", dest="replay_trades_out", type=str, default="trades.csv")
    parser.add_argument(
        "--replay-features-out", dest="replay_features_out", type=str, default=None, help="Optional features.csv export."
    )
    parser.add_argument("--out-equity", dest="out_equity", type=str, default=None, help="Backtest equity CSV output path.")
    parser.add_argument("--out-trades", dest="out_trades", type=str, default=None, help="Backtest trades CSV output path.")
    parser.add_argument(
        "--out-summary",
        dest="out_summary",
        type=str,
        default=None,
        help="Backtest summary output path (CSV or JSON).",
    )
    # Feature config overrides
    parser.add_argument("--rv-window", type=int, default=DEFAULT_FEATURE_CFG.rv_window)
    parser.add_argument("--conc-window", type=int, default=DEFAULT_FEATURE_CFG.conc_window)
    parser.add_argument(
        "--psi-mode",
        type=str,
        default=DEFAULT_FEATURE_CFG.psi_mode,
        choices=["none", "scale_phase"],
        help="Internal phase method: none or scale_phase (log-scale phase).",
    )
    parser.add_argument("--psi-window", type=int, default=DEFAULT_FEATURE_CFG.psi_window)
    parser.add_argument("--base", type=float, default=DEFAULT_FEATURE_CFG.base)
    # Regime thresholds
    parser.add_argument("--s-off", dest="s_off", type=float, default=None)
    parser.add_argument("--s-on", dest="s_on", type=float, default=None)
    parser.add_argument("--rv-off", dest="rv_off", type=float, default=None)
    parser.add_argument("--rv-reduce", dest="rv_reduce", type=float, default=None)
    parser.add_argument("--rv-guard", dest="rv_guard", type=float, default=None)
    parser.add_argument(
        "--strategy", type=str, choices=["none", "meanrev", "kalman", "kalman_mr_dual"], default="meanrev"
    )
    # Execution type flags
    parser.add_argument(
        "--order-type",
        dest="order_type",
        type=str,
        choices=["market", "limit_maker"],
        default="market",
        help="Order execution type: market (immediate fill) or limit_maker (post-only limit order)",
    )
    parser.add_argument(
        "--maker-offset-bps",
        dest="maker_offset_bps",
        type=float,
        default=1.0,
        help="Offset in basis points from best bid/ask for limit maker orders (default: 1.0)",
    )
    parser.add_argument(
        "--order-validity-seconds",
        dest="order_validity_seconds",
        type=int,
        default=60,
        help="Cancel limit maker orders older than this many seconds (default: 60)",
    )
    parser.add_argument(
        "--max-spread-bps",
        dest="max_spread_bps",
        type=float,
        default=20.0,
        help="Maximum allowed spread in basis points for limit maker orders (default: 20.0)",
    )
    parser.add_argument(
        "--maker-fee-rate",
        dest="maker_fee_rate",
        type=float,
        default=None,
        help="Fee rate for maker orders (default: same as --fee-rate)",
    )
    parser.add_argument(
        "--taker-fee-rate",
        dest="taker_fee_rate",
        type=float,
        default=None,
        help="Fee rate for taker orders (default: same as --fee-rate)",
    )
    parser.add_argument(
        "--min-profit-bps",
        dest="min_profit_bps",
        type=float,
        default=5.0,
        help="Minimum profit requirement in basis points for limit maker orders (default: 5.0)",
    )
    parser.add_argument(
        "--edge-softmax-alpha",
        dest="edge_softmax_alpha",
        type=float,
        default=20.0,
        help="Smoothness parameter for edge threshold calculation (default: 20.0)",
    )
    parser.add_argument(
        "--edge-floor-bps",
        dest="edge_floor_bps",
        type=float,
        default=0.0,
        help="Minimum edge threshold in basis points (default: 0.0)",
    )
    parser.add_argument(
        "--fee-roundtrip-mode",
        dest="fee_roundtrip_mode",
        type=str,
        choices=["maker_maker", "maker_taker"],
        default="maker_maker",
        help="Fee calculation mode: maker_maker (2*maker) or maker_taker (maker+taker) (default: maker_maker)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg_path = pathlib.Path(args.config)
    cfg = _load_config(cfg_path)

    if args.mode in ("paper", "live") and not args.db:
        print("Paper and live modes require --db for persistence.")
        sys.exit(1)

    logger = SQLiteLogger(args.db) if args.db else None
    state_path: Optional[pathlib.Path] = None
    if args.loop or args.db:
        if args.db:
            state_path = pathlib.Path(args.db).with_suffix(".state.json")
        elif args.loop:
            state_path = pathlib.Path("spot_bot_state.json")
    state_store = LoopStateStore(logger=logger, path=state_path)
    try:
        last_equity = logger.get_latest_equity() if logger else None

        symbol = args.symbol or cfg.get("symbol", "BTC/USDT")
        timeframe = args.timeframe or cfg.get("timeframe", "1h")
        limit_total = int(args.limit_total or cfg.get("limit_total", 2000))
        max_exposure = float(args.max_exposure or cfg.get("max_exposure", 0.3))
        fee_rate = float(args.fee_rate or cfg.get("fee_rate", 0.001))
        spread_bps = float(args.spread_bps if args.spread_bps is not None else cfg.get("spread_bps", 0.0))
        min_notional = float(args.min_notional or cfg.get("min_notional", 10.0))
        step_size = args.step_size

        csv_in = args.csv_in
        if args.csv:
            print(
                "--csv is deprecated; use --csv-in for input (old behavior) or --csv-out for exporting results.",
                file=sys.stderr,
            )
            if not csv_in:
                csv_in = args.csv

        if csv_in:
            csv_in_path = pathlib.Path(csv_in)
            if not csv_in_path.exists():
                print(f"CSV input not found: {csv_in_path}", file=sys.stderr)
                sys.exit(1)
            csv_in_path = csv_in_path
        else:
            csv_in_path = None

        feat_cfg = FeatureConfig(
            base=args.base,
            rv_window=args.rv_window,
            conc_window=args.conc_window,
            psi_mode=args.psi_mode,
            psi_window=args.psi_window,
        )

        regime_cfg = {
            "s_off": args.s_off if args.s_off is not None else cfg.get("s_off"),
            "s_on": args.s_on if args.s_on is not None else cfg.get("s_on"),
            "rv_off": args.rv_off if args.rv_off is not None else cfg.get("rv_off"),
            "rv_reduce": args.rv_reduce if args.rv_reduce is not None else cfg.get("rv_reduce"),
            "rv_guard": args.rv_guard if args.rv_guard is not None else cfg.get("rv_guard"),
        }
        regime_cfg = {k: v for k, v in regime_cfg.items() if v is not None}
        regime_engine = RegimeEngine(regime_cfg)
        if args.strategy == "kalman":
            strategy = KalmanStrategy()
        elif args.strategy == "kalman_mr_dual":
            strategy = MeanRevDualKalmanStrategy()
        elif args.strategy == "none":
            class NullStrategy:
                def generate_intent(self, features_df):
                    return Intent(desired_exposure=0.0, reason="strategy none", diagnostics={})

            strategy = NullStrategy()  # type: ignore
        else:
            strategy = MeanReversionStrategy()

        balances = {
            "usdt": last_equity["usdt"] if last_equity else float(args.initial_usdt),
            "btc": last_equity["btc"] if last_equity else 0.0,
        }

        if args.mode == "backtest":
            if not csv_in_path:
                print("Backtest mode requires --csv-in with historical OHLCV.", file=sys.stderr)
                sys.exit(1)
            try:
                df_bt = pd.read_csv(csv_in_path)
            except (FileNotFoundError, pd.errors.EmptyDataError, OSError) as exc:
                print(f"Failed to read CSV input {csv_in_path}: {exc}", file=sys.stderr)
                sys.exit(1)
            if "timestamp" not in df_bt.columns:
                print("CSV input must contain a 'timestamp' column.", file=sys.stderr)
                sys.exit(1)
            try:
                ts = df_bt["timestamp"]
                if pd.api.types.is_numeric_dtype(ts):
                    df_bt["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)
                else:
                    df_bt["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")
            except (ValueError, TypeError) as exc:
                print(f"Failed to parse timestamp column: {exc}", file=sys.stderr)
                sys.exit(1)
            if args.limit_total:
                df_bt = df_bt.head(int(args.limit_total))
            equity_df, trades_df, summary = run_backtest(
                df=df_bt,
                timeframe=timeframe,
                strategy_name=args.strategy,
                psi_mode=args.psi_mode,
                psi_window=args.psi_window,
                rv_window=args.rv_window,
                conc_window=args.conc_window,
                base=args.base,
                fee_rate=fee_rate,
                slippage_bps=args.slippage_bps,
                max_exposure=max_exposure,
                initial_usdt=balances["usdt"],
                min_notional=min_notional,
                step_size=step_size,
                bar_state="closed",
                hyst_k=args.hyst_k,
                hyst_floor=args.hyst_floor,
                hyst_mode=args.hyst_mode,
                spread_bps=spread_bps,
                k_vol=args.k_vol,
                edge_bps=args.edge_bps,
                max_delta_e_min=args.max_delta_e_min,
                alpha_floor=args.alpha_floor,
                alpha_cap=args.alpha_cap,
                vol_hyst_mode=args.vol_hyst_mode,
            )
            if args.out_equity:
                try:
                    out_e = pathlib.Path(args.out_equity)
                    out_e.parent.mkdir(parents=True, exist_ok=True)
                    equity_df.to_csv(out_e, index=False)
                except OSError as exc:
                    print(f"Failed to write equity output: {exc}", file=sys.stderr)
                    sys.exit(1)
            if args.out_trades:
                try:
                    out_t = pathlib.Path(args.out_trades)
                    out_t.parent.mkdir(parents=True, exist_ok=True)
                    trades_df.to_csv(out_t, index=False)
                except OSError as exc:
                    print(f"Failed to write trades output: {exc}", file=sys.stderr)
                    sys.exit(1)
            if args.out_summary:
                try:
                    out_s = pathlib.Path(args.out_summary)
                    out_s.parent.mkdir(parents=True, exist_ok=True)
                    if out_s.suffix.lower() == ".json":
                        out_s.write_text(json.dumps(summary, indent=2))
                    else:
                        pd.DataFrame([summary]).to_csv(out_s, index=False)
                except OSError as exc:
                    print(f"Failed to write summary output: {exc}", file=sys.stderr)
                    sys.exit(1)
            print(
                f"Backtest complete: bars={len(equity_df)}, trades={len(trades_df)}, final_equity={summary.get('final_equity', 0):.2f}"
            )
            if logger:
                logger.close()
            return

        if args.mode == "replay":
            if not csv_in_path:
                print("Replay mode requires --csv-in with historical OHLCV.", file=sys.stderr)
                sys.exit(1)
            df_replay = pd.read_csv(
                csv_in_path, parse_dates=["timestamp"], date_parser=lambda x: pd.to_datetime(x, utc=True)
            )
            df_replay = df_replay.set_index("timestamp")
            equity_df, trades_df, features_df = run_replay(
                ohlcv_df=df_replay,
                feature_cfg=feat_cfg,
                regime_engine=regime_engine,
                strategy=strategy,
                max_exposure=max_exposure,
                fee_rate=fee_rate,
                slippage_bps=args.slippage_bps,
                spread_bps=spread_bps,
                hyst_k=args.hyst_k,
                hyst_floor=args.hyst_floor,
                hyst_mode=args.hyst_mode,
                min_notional=min_notional,
                step_size=step_size,
                initial_usdt=balances["usdt"],
                initial_btc=balances["btc"],
                equity_path=pathlib.Path(args.replay_equity_out),
                trades_path=pathlib.Path(args.replay_trades_out),
                features_path=pathlib.Path(args.replay_features_out) if args.replay_features_out else None,
                k_vol=args.k_vol,
                edge_bps=args.edge_bps,
                max_delta_e_min=args.max_delta_e_min,
                alpha_floor=args.alpha_floor,
                alpha_cap=args.alpha_cap,
                vol_hyst_mode=args.vol_hyst_mode,
            )
            print(f"Replay finished: {len(equity_df)} steps, {len(trades_df)} trades, equity_out={args.replay_equity_out}")
            if logger:
                logger.close()
            return

        broker = None
        if args.mode == "paper":
            broker = PaperBroker.from_logger(
                logger=logger,
                fee_rate=fee_rate,
                min_notional=min_notional,
                fallback_usdt=balances["usdt"],
                step_size=step_size,
            )

        def _load_input_df() -> pd.DataFrame:
            if csv_in_path:
                df_local = pd.read_csv(csv_in_path, parse_dates=["timestamp"])
                df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], utc=True)
                return df_local.set_index("timestamp")
            return _load_or_fetch(args.mode, symbol, timeframe, limit_total, args.cache)

        latest_action = ""
        while True:
            df = _load_input_df()
            if df.empty:
                print("No data available.")
                if not args.loop:
                    break
                time.sleep(args.poll_seconds)
                continue

            df_for_trade, bar_row, ts_value, bar_state = _prepare_trade_data(df, timeframe, args.trade_on)
            if df_for_trade.empty or bar_row is None or ts_value is None:
                print("No new closed bar")
                if not args.loop:
                    break
                time.sleep(args.poll_seconds)
                continue

            latest_ts_int = _to_epoch_ms(ts_value)
            last_seen = state_store.load_last_closed_ts() if state_store else None
            if last_seen is None and logger:
                last_seen = logger.get_last_ts()
            if last_seen is not None and last_seen == latest_ts_int:
                print("No new closed bar")
                if not args.loop:
                    break
                time.sleep(args.poll_seconds)
                continue

            try:
                result = compute_step(
                    ohlcv_df=df_for_trade,
                    feature_cfg=feat_cfg,
                    regime_engine=regime_engine,
                    strategy=strategy,
                    max_exposure=max_exposure,
                    fee_rate=fee_rate,
                    balances=balances,
                    mode=args.mode,
                    broker=broker,
                    slippage_bps=args.slippage_bps,
                    spread_bps=spread_bps,
                    hyst_k=args.hyst_k,
                    hyst_floor=args.hyst_floor,
                    hyst_mode=args.hyst_mode,
                    k_vol=args.k_vol,
                    edge_bps=args.edge_bps,
                    max_delta_e_min=args.max_delta_e_min,
                    alpha_floor=args.alpha_floor,
                    alpha_cap=args.alpha_cap,
                    vol_hyst_mode=args.vol_hyst_mode,
                )
            except ValueError as exc:
                print(str(exc))
                sys.exit(1)

            execution_result = result.execution
            current_btc = result.equity["btc"]
            equity_usdt = result.equity["equity_usdt"]

            if args.mode == "live":
                if not args.i_understand_live_risk:
                    print("Refusing to run live mode without --i-understand-live-risk.")
                    sys.exit(1)
                try:
                    from spot_bot.execution.ccxt_executor import CCXTExecutor, ExecutorConfig
                except Exception as exc:  # pragma: no cover - optional dependency
                    print(f"Live execution unavailable: {exc}")
                    sys.exit(1)
                api_key = args.api_key or os.getenv('BINANCE_API_KEY') or os.getenv('CCXT_API_KEY')
                api_secret = args.api_secret or os.getenv('BINANCE_API_SECRET') or os.getenv('CCXT_API_SECRET')
                if not api_key or not api_secret:
                    print('Missing API credentials for live mode. Set env BINANCE_API_KEY and BINANCE_API_SECRET (recommended) or pass --api-key/--api-secret.')
                    sys.exit(1)

                # Determine fee rates
                maker_fee = args.maker_fee_rate if args.maker_fee_rate is not None else fee_rate
                taker_fee = args.taker_fee_rate if args.taker_fee_rate is not None else fee_rate

                exec_cfg = ExecutorConfig(
                    exchange_id=args.exchange_id,
                    symbol=symbol,
                    api_key=api_key,
                    api_secret=api_secret,
                    max_notional_per_trade=cfg.get('max_notional_per_trade', 300.0),
                    max_trades_per_day=cfg.get('max_trades_per_day', 10),
                    max_turnover_per_day=cfg.get('max_turnover_per_day', 2000.0),
                    slippage_bps_limit=cfg.get('slippage_bps_limit', 10.0),
                    min_balance_reserve_usdt=cfg.get('min_usdt_reserve', 50.0),
                    fee_rate=fee_rate,
                    min_notional=min_notional,
                    order_type=args.order_type,
                    maker_offset_bps=args.maker_offset_bps,
                    order_validity_seconds=args.order_validity_seconds,
                    max_spread_bps=args.max_spread_bps,
                    maker_fee_rate=maker_fee,
                    taker_fee_rate=taker_fee,
                    slippage_bps=args.slippage_bps,
                    min_profit_bps=args.min_profit_bps,
                    edge_softmax_alpha=args.edge_softmax_alpha,
                    edge_floor_bps=args.edge_floor_bps,
                    fee_roundtrip_mode=args.fee_roundtrip_mode,
                )
                executor = CCXTExecutor(exec_cfg)
                if abs(result.delta_btc) > 0:
                    side = "buy" if result.delta_btc > 0 else "sell"
                    
                    # Get avg_entry_price for SELL guard
                    avg_entry_price = balances.get("avg_entry_price")
                    
                    # Cancel stale orders before placing new ones (for limit_maker mode)
                    if args.order_type == "limit_maker":
                        executor.cancel_stale_orders()
                        execution_result = executor.place_limit_maker_order(
                            side, 
                            abs(result.delta_btc), 
                            result.close,
                            portfolio_avg_entry_price=avg_entry_price,
                        )
                    else:
                        execution_result = executor.place_market_order(side, abs(result.delta_btc), result.close)
                    
                    if execution_result.get("status") == "filled":
                        qty = float(execution_result.get("filled_qty") or execution_result.get("qty") or 0.0)
                        _apply_live_fill_to_balances(balances, side, qty, result.close, fee_rate)
                    elif execution_result.get("status") == "open":
                        # Limit maker order placed but not filled yet
                        # Do not update balances until order fills
                        print(f"Limit maker order placed: {execution_result.get('order_id')}")
                    
                    current_btc = balances["btc"]
                    equity_usdt = balances["usdt"] + current_btc * result.close

            action = "HOLD"
            if abs(result.delta_btc) > 0:
                action = "BUY" if result.delta_btc > 0 else "SELL"
                if execution_result and execution_result.get("status") not in ("filled", "partial"):
                    action = f"{action}-rejected"

            balances["btc"] = current_btc
            balances["usdt"] = float(equity_usdt - current_btc * result.close)
            latest_action = action

            # Extract diagnostics for logging
            diag = result.diagnostics if hasattr(result, 'diagnostics') else {}
            target_raw = diag.get('target_exposure_raw', result.target_exposure)
            target_clamped = diag.get('target_exposure_clamped', result.target_exposure)
            delta_e = diag.get('delta_e', 0.0)
            delta_e_min = diag.get('delta_e_min', 0.0)
            suppressed = diag.get('hysteresis_suppressed', False)
            clamped = diag.get('clamped_long_only', False)

            summary = (
                f"{result.ts} | mode={args.mode} price={result.close:.2f} "
                f"S={result.features_row.get('S', float('nan')):.4f} "
                f"C={result.features_row.get('C', float('nan')):.4f} "
                f"C_int={result.features_row.get('C_int', float('nan')):.4f} "
                f"psi={result.features_row.get('psi', float('nan')):.4f} "
                f"rv={result.features_row.get('rv', float('nan')):.4f} "
                f"risk={result.decision.risk_state} budget={result.decision.risk_budget:.3f} "
                f"intent={result.intent.desired_exposure:.3f} tgt_raw={target_raw:.3f} "
                f"tgt_clmp={target_clamped:.3f} tgt_final={result.target_exposure:.3f} "
                f"delta_e={delta_e:.3f} delta_e_min={delta_e_min:.3f} "
                f"supp={suppressed} clmp={clamped} "
                f"delta_btc={result.delta_btc:.6f} equity={equity_usdt:.2f} action={action} bar_state={bar_state}"
            )
            print(summary)

            if args.csv_out:
                out_path = pathlib.Path(args.csv_out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if args.csv_out_mode == "features":
                    export_df = _compute_feature_outputs(
                        ohlcv_df=df_for_trade,
                        feature_cfg=feat_cfg,
                        regime_engine=regime_engine,
                        strategy=strategy,
                        max_exposure=max_exposure,
                        equity_usdt=result.equity["equity_usdt"],
                        tail_rows=args.csv_out_tail,
                        latest_action=action,
                    )
                    export_df["bar_state"] = bar_state
                    export_df["last_closed_ts"] = _to_utc_timestamp(ts_value)
                    if "timestamp" not in export_df.columns:
                        export_df = export_df.reset_index()
                        if export_df.columns.empty:
                            export_df["timestamp"] = pd.Series(dtype="datetime64[ns]")
                        else:
                            timestamp_candidates = [c for c in export_df.columns if pd.api.types.is_datetime64_any_dtype(export_df[c])]
                            timestamp_col = timestamp_candidates[0] if timestamp_candidates else export_df.columns[0]
                            export_df = export_df.rename(columns={timestamp_col: "timestamp"})
                    export_df.to_csv(out_path, index=False)
                else:
                    export_row = {
                        "timestamp": ts_value,
                        "last_closed_ts": ts_value,
                        "bar_state": bar_state,
                        "open": bar_row["open"],
                        "high": bar_row["high"],
                        "low": bar_row["low"],
                        "close": bar_row["close"],
                        "volume": bar_row["volume"],
                        "rv": result.features_row.get("rv"),
                        "C": result.features_row.get("C"),
                        "psi": result.features_row.get("psi"),
                        "C_int": result.features_row.get("C_int"),
                        "S": result.features_row.get("S"),
                        "risk_state": result.decision.risk_state,
                        "risk_budget": result.decision.risk_budget,
                        "intent_exposure": result.intent.desired_exposure,
                        "target_exposure": result.target_exposure,
                        "action": action,
                    }
                    with out_path.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_OUTPUT_COLUMNS)
                        writer.writeheader()
                        writer.writerow(export_row)

            if logger:
                logger.upsert_bar(
                    ts=ts_value,
                    open=bar_row["open"],
                    high=bar_row["high"],
                    low=bar_row["low"],
                    close=bar_row["close"],
                    volume=bar_row["volume"],
                )
                logger.upsert_features(
                    ts=result.ts,
                    rv=result.features_row.get("rv"),
                    C=result.features_row.get("C"),
                    psi=result.features_row.get("psi"),
                    C_int=result.features_row.get("C_int"),
                    S=result.features_row.get("S"),
                )
                logger.upsert_decision(
                    ts=result.ts, risk_state=result.decision.risk_state, risk_budget=result.decision.risk_budget, reason=result.decision.reason
                )
                logger.upsert_intent(ts=result.ts, desired_exposure=result.target_exposure, reason=result.intent.reason)

                if execution_result:
                    qty_value = float(execution_result.get("filled_qty") or execution_result.get("qty") or 0.0)
                else:
                    qty_value = 0.0
                if execution_result and qty_value > 0:
                    logger.insert_execution(
                        ts=result.ts,
                        mode=args.mode,
                        side="buy" if result.delta_btc > 0 else "sell",
                        qty=qty_value,
                        price=float(execution_result.get("price") or execution_result.get("avg_price") or result.close),
                        fee=float(execution_result.get("fee", execution_result.get("fee_est", 0.0))),
                        order_id=str(execution_result.get("order_id", "")),
                        status=execution_result.get("status", "filled"),
                        meta={"delta_btc": result.delta_btc},
                    )

                logger.upsert_equity(
                    ts=result.ts,
                    equity_usdt=equity_usdt,
                    btc=current_btc,
                    usdt=float(equity_usdt - current_btc * result.close),
                )

                state_store.save_last_closed_ts(latest_ts_int)
            elif state_store:
                state_store.save_last_closed_ts(latest_ts_int)

            if not args.loop:
                break
            time.sleep(args.poll_seconds)
    finally:
        if logger:
            logger.close()


if __name__ == "__main__":
    main()
