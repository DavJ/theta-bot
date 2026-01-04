#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import pandas as pd
import yaml

from spot_bot.features import FeatureConfig, compute_features
from spot_bot.live import PaperBroker, BrokerConfig
from spot_bot.persist import SQLiteLogger
from spot_bot.portfolio.sizer import compute_target_position
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.strategies.mean_reversion import MeanReversionStrategy


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


def _has_new_bar(logger: Optional[SQLiteLogger], df: pd.DataFrame, state_file: Optional[str]) -> bool:
    latest_ts = str(df.index[-1])
    if logger:
        prev_ts = logger.latest_bar_timestamp()
        if prev_ts == latest_ts:
            return False
    elif state_file:
        state_path = pathlib.Path(state_file)
        if state_path.exists():
            prev = state_path.read_text().strip()
            if prev == latest_ts:
                return False
        state_path.write_text(latest_ts)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Spot Bot live loop.")
    parser.add_argument("--mode", choices=["dryrun", "paper", "live"], default="dryrun")
    parser.add_argument("--config", type=str, default=str(pathlib.Path(__file__).parent / "config.yaml"))
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV for offline testing.")
    parser.add_argument("--cache", type=str, default=None, help="Optional cache file for OHLCV.")
    parser.add_argument("--db", type=str, default=None, help="Optional SQLite DB for persistence.")
    parser.add_argument("--state-file", type=str, default=".run_live_state", help="State file for last processed bar.")
    parser.add_argument("--i-understand-live-risk", action="store_true", help="Required to enable live execution.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg_path = pathlib.Path(args.config)
    cfg = _load_config(cfg_path)
    logger = SQLiteLogger(args.db) if args.db else None
    latest_equity = logger.latest_equity() if logger else None

    symbol = cfg.get("symbol", "BTC/USDT")
    timeframe = cfg.get("timeframe", "1h")
    limit_total = int(cfg.get("limit_total", 500))
    max_exposure = float(cfg.get("max_exposure", 0.3))
    fee_rate = float(cfg.get("fee_rate", 0.001))

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
    else:
        df = _load_or_fetch(args.mode, symbol, timeframe, limit_total, args.cache)

    if df.empty:
        print("No data available.")
        sys.exit(1)

    if not _has_new_bar(logger, df, args.state_file):
        print("No new closed bar; exiting.")
        return

    feat_cfg = FeatureConfig()
    features = compute_features(df, feat_cfg).dropna(subset=["S", "C"])
    if features.empty:
        print("Insufficient data to compute features.")
        sys.exit(1)

    regime = RegimeEngine({})
    decision = regime.decide(features)
    strategy = MeanReversionStrategy()
    intent = strategy.generate_intent(features)

    latest_price = float(df["close"].iloc[-1])
    default_usdt = float(cfg.get("initial_equity", 1000.0))
    equity_usdt = float(latest_equity.get("equity_usdt", default_usdt)) if latest_equity else default_usdt
    current_btc = float(latest_equity.get("btc", 0.0)) if latest_equity else 0.0
    broker = None
    execution_result = None

    if args.mode == "paper":
        broker_cfg = BrokerConfig(
            fee_rate=fee_rate,
            min_notional=float(cfg.get("min_notional", 10.0)),
            max_exposure=max_exposure,
            starting_usdt=latest_equity.get("usdt", cfg.get("initial_equity", 1000.0)) if latest_equity else cfg.get("initial_equity", 1000.0),
            starting_btc=latest_equity.get("btc", 0.0) if latest_equity else 0.0,
        )
        broker = PaperBroker(broker_cfg)
        equity_snapshot = broker.mark_to_market(latest_price)
        equity_usdt = equity_snapshot["equity_usdt"]
        current_btc = equity_snapshot["btc"]

    target_exposure = 0.0 if decision.risk_state != "ON" else intent.desired_exposure * decision.risk_budget
    target_exposure = min(max_exposure, max(target_exposure, 0.0))
    target_btc = compute_target_position(
        equity_usdt=equity_usdt,
        price=latest_price,
        desired_exposure=target_exposure,
        risk_budget=1.0,
        max_exposure=max_exposure,
        risk_state="ON",
    )
    delta_btc = target_btc - current_btc

    if args.mode == "paper" and broker:
        side = "buy" if delta_btc > 0 else "sell"
        execution_result = broker.place_order(side=side, qty=abs(delta_btc), price=latest_price) if abs(delta_btc) > 0 else None
        equity_snapshot = broker.mark_to_market(latest_price)
        equity_usdt = equity_snapshot["equity_usdt"]
        current_btc = equity_snapshot["btc"]
    elif args.mode == "live":
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
            min_notional=cfg.get("min_notional", 10.0),
        )
        executor = CCXTExecutor(exec_cfg)
        if abs(delta_btc) > 0:
            execution_result = executor.place_market_order("buy" if delta_btc > 0 else "sell", abs(delta_btc), latest_price)

    summary = (
        f"{df.index[-1]} | mode={args.mode} price={latest_price:.2f} "
        f"risk={decision.risk_state} budget={decision.risk_budget:.3f} "
        f"intent={intent.desired_exposure:.3f} target_exp={target_exposure:.3f} "
        f"delta_btc={delta_btc:.6f} equity={equity_usdt:.2f}"
    )
    print(summary)

    if logger:
        logger.log_bars(df.tail(1).reset_index().rename(columns={"index": "timestamp"}).to_dict(orient="records"))
        latest_feat = features.tail(1)
        feat_records = (
            latest_feat.reset_index()
            .rename(columns={"index": "timestamp"})
            .reindex(columns=["timestamp", "rv", "C", "psi", "C_int", "S"])
            .to_dict(orient="records")
        )
        logger.log_features(feat_records)
        logger.log_decision(
            timestamp=features.index[-1],
            risk_state=decision.risk_state,
            risk_budget=decision.risk_budget,
            reason=decision.reason,
        )
        logger.log_intent(timestamp=features.index[-1], desired_exposure=target_exposure, reason=intent.reason)
        if execution_result:
            logger.log_execution(
                timestamp=features.index[-1],
                action=execution_result.get("action", "market"),
                qty=execution_result.get("qty", 0.0) if isinstance(execution_result, dict) else 0.0,
                price=latest_price,
                fee=execution_result.get("fee", 0.0) if isinstance(execution_result, dict) else 0.0,
                order_id=str(execution_result.get("order_id", "")) if isinstance(execution_result, dict) else "",
                status=execution_result.get("status", "filled") if isinstance(execution_result, dict) else "filled",
            )
        logger.log_equity(timestamp=features.index[-1], equity_usdt=equity_usdt, btc=current_btc, usdt=equity_usdt - current_btc * latest_price)
        logger.close()


if __name__ == "__main__":
    main()
