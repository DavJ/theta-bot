#!/usr/bin/env python3
"""Smoke-test CLI for the regime engine."""

from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd

try:
    from download_market_data import download_binance_data as _download_binance_data
except ImportError:  # pragma: no cover - defensive import for optional dependency
    _download_binance_data = None

try:
    from theta_bot_averaging.data import load_dataset as _load_dataset
except ImportError:  # pragma: no cover - defensive import for optional dependency
    _load_dataset = None

from spot_bot.regime.regime_engine import RegimeEngine


def load_ohlcv(csv: Optional[str], symbol: str, interval: str, limit: int) -> pd.DataFrame:
    if csv:
        if _load_dataset is None:
            raise ImportError(
                "theta_bot_averaging.data.load_dataset (shipped with this repo) is required when using --csv"
            )
        df = _load_dataset(csv)
    else:
        if _download_binance_data is None:
            raise ImportError(
                "download_market_data.download_binance_data (shipped with this repo) is required when downloading data"
            )
        raw = _download_binance_data(symbol=symbol, interval=interval, limit=limit)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw = raw.set_index("timestamp")
        df = raw[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "timestamp"
    return df


def compute_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    returns = df["close"].pct_change()
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_abs_mean = returns.abs().rolling(window).mean()
    cumulative = returns.rolling(window).sum()

    features = pd.DataFrame(
        {
            "S": rolling_mean,
            "C": rolling_abs_mean,
            "C_int": cumulative,
            "rv": rolling_std,
        },
        index=df.index,
    )
    return features.dropna()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run regime engine on latest bar.")
    parser.add_argument("--csv", type=str, help="Path to OHLCV CSV (uses loader utilities)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to download (if no CSV provided)")
    parser.add_argument("--interval", type=str, default="1h", help="Interval to download (if no CSV provided)")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles to download")
    parser.add_argument("--window", type=int, default=48, help="Rolling window for feature computation")
    parser.add_argument("--s-off", dest="s_off", type=float, default=-0.05, help="Score threshold for OFF")
    parser.add_argument("--s-on", dest="s_on", type=float, default=0.1, help="Score threshold for ON/REDUCE")
    parser.add_argument("--rv-off", dest="rv_off", type=float, default=0.05, help="Vol threshold for OFF")
    parser.add_argument("--rv-reduce", dest="rv_reduce", type=float, default=0.03, help="Vol threshold for REDUCE")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    df = load_ohlcv(args.csv, args.symbol, args.interval, args.limit)
    features = compute_features(df, window=args.window)

    config = {
        "s_off": args.s_off,
        "s_on": args.s_on,
        "rv_off": args.rv_off,
        "rv_reduce": args.rv_reduce,
        "s_budget_low": args.s_off,
        "s_budget_high": args.s_on,
        "rv_guard": args.rv_off,
    }
    engine = RegimeEngine(config)
    decision = engine.decide(features)

    print("\n=== Regime Decision ===")
    print(f"Risk state : {decision.risk_state}")
    print(f"Risk budget: {decision.risk_budget:.3f}")
    print(f"Reason     : {decision.reason}")
    print("Diagnostics:")
    for k, v in decision.diagnostics.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
