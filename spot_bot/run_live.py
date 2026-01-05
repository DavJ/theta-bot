#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Dict, Optional

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
from spot_bot.strategies.mean_reversion import MeanReversionStrategy


DEFAULT_FEATURE_CFG = FeatureConfig()
CSV_OUTPUT_COLUMNS = [
    "timestamp",
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


def _load_config(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _load_cached(cache_path: Optional[str]) -> Optional[pd.DataFrame]:
    if cache_path and pathlib.Path(cache_path).exists():
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
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
) -> StepResult:
    features = compute_features(ohlcv_df, feature_cfg).dropna(subset=["S", "C"])
    if features.empty:
        raise ValueError("Insufficient data to compute features.")

    latest_price = float(ohlcv_df["close"].iloc[-1])
    latest_ts = features.index[-1]
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
    if features.empty:
        return features.head(0)

    # Add OHLCV columns for completeness
    for col in ("open", "high", "low", "close", "volume"):
        if col in ohlcv_df.columns:
            features[col] = ohlcv_df[col]

    valid = features.dropna(subset=["S", "C"]).copy()
    if valid.empty:
        return valid

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

    intent_series = desired_exposure_series.reindex(valid.index).fillna(0.0).clip(lower=0.0, upper=1.0)
    target_exp_series = (intent_series * risk_budget_series).clip(lower=0.0, upper=float(max_exposure))
    target_exp_series = target_exp_series.where(risk_state_series == "ON", 0.0)
    target_btc_series = target_exp_series * equity / valid["close"]

    valid["risk_state"] = risk_state_series
    valid["risk_budget"] = risk_budget_series
    valid["intent_exposure"] = intent_series
    valid["target_exposure"] = target_exp_series.fillna(0.0)
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
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Spot Bot live loop.")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--limit-total", dest="limit_total", type=int, default=2000)
    parser.add_argument("--mode", choices=["dryrun", "paper", "live"], default="dryrun")
    parser.add_argument("--db", type=str, default=None, help="SQLite DB path (required for paper/live).")
    parser.add_argument("--initial-usdt", dest="initial_usdt", type=float, default=1000.0)
    parser.add_argument("--max-exposure", dest="max_exposure", type=float, default=0.3)
    parser.add_argument("--fee-rate", dest="fee_rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=0.0)
    parser.add_argument("--min-notional", dest="min_notional", type=float, default=10.0)
    parser.add_argument("--step-size", dest="step_size", type=float, default=None)
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
    # Feature config overrides
    parser.add_argument("--rv-window", type=int, default=DEFAULT_FEATURE_CFG.rv_window)
    parser.add_argument("--conc-window", type=int, default=DEFAULT_FEATURE_CFG.conc_window)
    parser.add_argument("--psi-window", type=int, default=DEFAULT_FEATURE_CFG.psi_window)
    parser.add_argument("--cepstrum-domain", type=str, default=DEFAULT_FEATURE_CFG.cepstrum_domain)
    parser.add_argument("--cepstrum-min-bin", type=int, default=DEFAULT_FEATURE_CFG.cepstrum_min_bin)
    parser.add_argument("--cepstrum-max-frac", type=float, default=DEFAULT_FEATURE_CFG.cepstrum_max_frac)
    parser.add_argument("--base", type=float, default=DEFAULT_FEATURE_CFG.base)
    # Regime thresholds
    parser.add_argument("--s-off", dest="s_off", type=float, default=None)
    parser.add_argument("--s-on", dest="s_on", type=float, default=None)
    parser.add_argument("--rv-off", dest="rv_off", type=float, default=None)
    parser.add_argument("--rv-reduce", dest="rv_reduce", type=float, default=None)
    parser.add_argument("--rv-guard", dest="rv_guard", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg_path = pathlib.Path(args.config)
    cfg = _load_config(cfg_path)

    if args.mode in ("paper", "live") and not args.db:
        print("Paper and live modes require --db for persistence.")
        sys.exit(1)

    logger = SQLiteLogger(args.db) if args.db else None
    try:
        last_equity = logger.get_latest_equity() if logger else None

        symbol = args.symbol or cfg.get("symbol", "BTC/USDT")
        timeframe = args.timeframe or cfg.get("timeframe", "1h")
        limit_total = int(args.limit_total or cfg.get("limit_total", 2000))
        max_exposure = float(args.max_exposure or cfg.get("max_exposure", 0.3))
        fee_rate = float(args.fee_rate or cfg.get("fee_rate", 0.001))
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
            df = pd.read_csv(csv_in_path, parse_dates=["timestamp"])
            df = df.set_index("timestamp")
        else:
            df = _load_or_fetch(args.mode, symbol, timeframe, limit_total, args.cache)

        if df.empty:
            print("No data available.")
            sys.exit(1)

        df = latest_closed_ohlcv(df, timeframe)
        if df.empty:
            print("No new closed bar.")
            return

        latest_ts = df.index[-1]
        latest_ts_int = _to_epoch_ms(latest_ts)
        if logger and logger.get_last_ts() == latest_ts_int:
            print("No new closed bar since last run.")
            return

        feat_cfg = FeatureConfig(
            base=args.base,
            rv_window=args.rv_window,
            conc_window=args.conc_window,
            psi_window=args.psi_window,
            cepstrum_domain=args.cepstrum_domain,
            cepstrum_min_bin=args.cepstrum_min_bin,
            cepstrum_max_frac=args.cepstrum_max_frac,
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
        strategy = MeanReversionStrategy()

        balances = {
            "usdt": last_equity["usdt"] if last_equity else float(args.initial_usdt),
            "btc": last_equity["btc"] if last_equity else 0.0,
        }

        broker = None
        if args.mode == "paper":
            broker = PaperBroker.from_logger(
                logger=logger,
                fee_rate=fee_rate,
                min_notional=min_notional,
                fallback_usdt=balances["usdt"],
                step_size=step_size,
            )

        try:
            result = compute_step(
                ohlcv_df=df,
                feature_cfg=feat_cfg,
                regime_engine=regime_engine,
                strategy=strategy,
                max_exposure=max_exposure,
                fee_rate=fee_rate,
                balances=balances,
                mode=args.mode,
                broker=broker,
                slippage_bps=args.slippage_bps,
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

        summary = (
            f"{result.ts} | mode={args.mode} price={result.close:.2f} "
            f"S={result.features_row.get('S', float('nan')):.4f} "
            f"C={result.features_row.get('C', float('nan')):.4f} "
            f"C_int={result.features_row.get('C_int', float('nan')):.4f} "
            f"psi={result.features_row.get('psi', float('nan')):.4f} "
            f"rv={result.features_row.get('rv', float('nan')):.4f} "
            f"risk={result.decision.risk_state} budget={result.decision.risk_budget:.3f} "
            f"intent={result.intent.desired_exposure:.3f} target_exp={result.target_exposure:.3f} "
            f"delta_btc={result.delta_btc:.6f} equity={equity_usdt:.2f} action={action}"
        )
        print(summary)

        bar_row = None
        ts_value = None
        if args.csv_out:
            out_path = pathlib.Path(args.csv_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if args.csv_out_mode == "features":
                export_df = _compute_feature_outputs(
                    ohlcv_df=df,
                    feature_cfg=feat_cfg,
                    regime_engine=regime_engine,
                    strategy=strategy,
                    max_exposure=max_exposure,
                    equity_usdt=result.equity["equity_usdt"],
                    tail_rows=args.csv_out_tail,
                    latest_action=action,
                )
                if "timestamp" not in export_df.columns:
                    export_df = export_df.reset_index()
                    if export_df.columns.empty:
                        export_df["timestamp"] = pd.Series(dtype="datetime64[ns]")
                    else:
                        timestamp_candidates = [c for c in export_df.columns if pd.api.types.is_datetime64_any_dtype(export_df[c])]
                        timestamp_col = timestamp_candidates[0] if timestamp_candidates else export_df.columns[0]
                        export_df = export_df.rename(columns={timestamp_col: "timestamp"})
                    if "timestamp" not in export_df.columns:
                        export_df["timestamp"] = pd.Series(dtype="datetime64[ns]")
                export_df.to_csv(out_path, index=False)
            else:
                bar_row, ts_value = _latest_bar_row(df)
                export_row = {
                    "timestamp": ts_value,
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
            if bar_row is None or ts_value is None:
                bar_row, ts_value = _latest_bar_row(df)
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
                price_value = float(execution_result.get("price") or execution_result.get("avg_price") or result.close)
            else:
                qty_value = 0.0
                price_value = result.close

            if execution_result and qty_value > 0:
                logger.insert_execution(
                    ts=result.ts,
                    mode=args.mode,
                    side="buy" if result.delta_btc > 0 else "sell",
                    qty=qty_value,
                    price=price_value,
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
    finally:
        if logger:
            logger.close()


if __name__ == "__main__":
    main()
