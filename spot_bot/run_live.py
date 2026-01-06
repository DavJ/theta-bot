#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
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


def _apply_fill_to_balances(balances: Dict[str, float], side: str, qty: float, price: float, fee_rate: float) -> float:
    fee = qty * price * fee_rate
    if side == "buy":
        balances["usdt"] = balances.get("usdt", 0.0) - qty * price - fee
        balances["btc"] = balances.get("btc", 0.0) + qty
    else:
        balances["usdt"] = balances.get("usdt", 0.0) + qty * price - fee
        balances["btc"] = max(0.0, balances.get("btc", 0.0) - qty)
    return fee


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
) -> StepResult:
    features = compute_features(ohlcv_df, feature_cfg).dropna(subset=["S", "C"])
    if isinstance(features.index, pd.DatetimeIndex):
        features.index = pd.to_datetime(features.index, utc=True)
    if features.empty:
        raise ValueError("Insufficient data to compute features.")

    latest_price = float(ohlcv_df["close"].iloc[-1])
    latest_ts = _to_utc_timestamp(features.index[-1])
    decision = regime_engine.decide(features)
    intent = strategy.generate_intent(features)

    current_btc = float(balances.get("btc", 0.0))
    current_usdt = float(balances.get("usdt", 0.0))
    equity_usdt = current_usdt + current_btc * latest_price

    if mode == "paper" and broker:
        bal = broker.balances()
        current_btc = bal["btc"]
        current_usdt = bal["usdt"]
        equity_usdt = broker.equity(latest_price)

    target_btc = compute_target_position(
        equity_usdt=equity_usdt,
        price=latest_price,
        desired_exposure=float(intent.desired_exposure),
        risk_budget=float(decision.risk_budget),
        max_exposure=max_exposure,
        risk_state=decision.risk_state,
    )
    target_exposure = (target_btc * latest_price / equity_usdt) if equity_usdt > 0 else 0.0
    current_exposure = (current_btc * latest_price / equity_usdt) if equity_usdt > 0 else 0.0

    cost = fee_rate + 2 * (slippage_bps / 10_000.0) + (spread_bps / 10_000.0)
    rv_series = features["rv"] if "rv" in features.columns else pd.Series(dtype=float)
    rv_series = rv_series.dropna()
    rv_current = float(rv_series.iloc[-1]) if not rv_series.empty else 0.0
    rv_ref_candidates = rv_series.tail(500)
    rv_ref = float(rv_ref_candidates.median()) if not rv_ref_candidates.empty else float(rv_series.median()) if not rv_series.empty else 0.0
    if rv_ref <= 0.0 and not rv_series.empty:
        rv_ref = float(rv_series.median())
    if rv_ref <= 0.0:
        rv_ref = abs(rv_current)
    rv_ref = rv_ref if rv_ref and rv_ref > 0.0 else 1.0
    rv_current_safe = rv_current if rv_current and rv_current > 0.0 else 1e-8

    delta_e = abs(target_exposure - current_exposure)
    delta_e_min = max(hyst_floor, hyst_k * cost * (rv_ref / rv_current_safe))
    apply_hysteresis = False
    if hyst_mode == "zscore":
        zscore_val = intent.diagnostics.get("zscore") if intent.diagnostics else None
        if zscore_val is not None:
            delta_signal = abs(float(zscore_val))
            delta_signal_min = max(hyst_floor, hyst_k * cost)
            apply_hysteresis = delta_signal < delta_signal_min
        else:
            apply_hysteresis = delta_e < delta_e_min
    else:
        apply_hysteresis = delta_e < delta_e_min

    if apply_hysteresis:
        target_exposure = current_exposure
        target_btc = current_btc

    delta_btc = target_btc - current_btc

    execution: Optional[Dict[str, Any]] = None
    if mode == "paper" and broker and abs(delta_btc) > 0:
        slip = slippage_bps / 10000.0
        fill_price = latest_price * (1 + slip if delta_btc > 0 else 1 - slip)
        execution = broker.trade_to_target_btc(target_btc, fill_price)
        bal = broker.balances()
        current_btc = bal["btc"]
        current_usdt = bal["usdt"]
        equity_usdt = broker.equity(latest_price)

    equity_snapshot = {"equity_usdt": equity_usdt, "btc": current_btc, "usdt": current_usdt}
    return StepResult(
        ts=latest_ts,
        close=latest_price,
        decision=decision,
        intent=intent,
        target_exposure=target_exposure,
        target_btc=target_btc,
        delta_btc=delta_btc,
        equity=equity_snapshot,
        execution=execution,
        features_row=features.iloc[-1],
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
    features_rows: List[pd.DataFrame] = []

    last_error_msg: Optional[str] = None

    for i in range(len(ohlcv_df)):
        df_slice = ohlcv_df.iloc[: i + 1].copy()
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
            qty_raw = execution_result.get("filled_qty")
            if qty_raw is None:
                qty_raw = execution_result.get("qty")
            qty = float(qty_raw or 0.0)
            if qty > 0 and execution_result.get("status") in ("filled", "partial"):
                price_raw = execution_result.get("price")
                if price_raw is None:
                    price_raw = execution_result.get("avg_price")
                if price_raw is None:
                    price_raw = result.close
                price = float(price_raw)
                fee_raw = execution_result.get("fee")
                if fee_raw is None:
                    fee_raw = execution_result.get("fee_est", 0.0)
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
                features_rows.append(feat_df.tail(1))

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)
    features_df = None
    if features_rows:
        features_df = pd.concat(features_rows, axis=0, ignore_index=True)

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
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Spot Bot live loop.")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--limit-total", dest="limit_total", type=int, default=2000)
    parser.add_argument("--mode", choices=["dryrun", "paper", "live", "replay"], default="dryrun")
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
    parser.add_argument("--i-understand-live-risk", action="store_true", help="Required to enable live execution.")
    parser.add_argument("--replay-equity-out", dest="replay_equity_out", type=str, default="equity_curve.csv")
    parser.add_argument("--replay-trades-out", dest="replay_trades_out", type=str, default="trades.csv")
    parser.add_argument(
        "--replay-features-out", dest="replay_features_out", type=str, default=None, help="Optional features.csv export."
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

        if args.mode == "replay":
            if not csv_in_path:
                print("Replay mode requires --csv-in with historical OHLCV.", file=sys.stderr)
                sys.exit(1)
            df_replay = pd.read_csv(csv_in_path, parse_dates=["timestamp"])
            df_replay["timestamp"] = pd.to_datetime(df_replay["timestamp"], utc=True)
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
                exec_cfg = ExecutorConfig(
                    symbol=symbol,
                    max_notional_per_trade=cfg.get("max_notional_per_trade", 300.0),
                    max_trades_per_day=cfg.get("max_trades_per_day", 10),
                    max_turnover_per_day=cfg.get("max_turnover_per_day", 2000.0),
                    slippage_bps_limit=cfg.get("slippage_bps_limit", 10.0),
                    min_balance_reserve_usdt=cfg.get("min_usdt_reserve", 50.0),
                    fee_rate=fee_rate,
                    min_notional=min_notional,
                )
                executor = CCXTExecutor(exec_cfg)
                if abs(result.delta_btc) > 0:
                    side = "buy" if result.delta_btc > 0 else "sell"
                    execution_result = executor.place_market_order(side, abs(result.delta_btc), result.close)
                    if execution_result.get("status") == "filled":
                        qty = float(execution_result.get("filled_qty") or execution_result.get("qty") or 0.0)
                        _apply_fill_to_balances(balances, side, qty, result.close, fee_rate)
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

            summary = (
                f"{result.ts} | mode={args.mode} price={result.close:.2f} "
                f"S={result.features_row.get('S', float('nan')):.4f} "
                f"C={result.features_row.get('C', float('nan')):.4f} "
                f"C_int={result.features_row.get('C_int', float('nan')):.4f} "
                f"psi={result.features_row.get('psi', float('nan')):.4f} "
                f"rv={result.features_row.get('rv', float('nan')):.4f} "
                f"risk={result.decision.risk_state} budget={result.decision.risk_budget:.3f} "
                f"intent={result.intent.desired_exposure:.3f} target_exp={result.target_exposure:.3f} "
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
