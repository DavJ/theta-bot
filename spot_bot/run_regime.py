#!/usr/bin/env python3
"""Smoke-test CLI for the regime engine."""

from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd

if __package__ is None and __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from download_market_data import download_binance_data as _download_binance_data
except ImportError:  # pragma: no cover - defensive import for optional dependency
    _download_binance_data = None

try:
    from theta_bot_averaging.data import load_dataset as _load_dataset
except ImportError:  # pragma: no cover - defensive import for optional dependency
    _load_dataset = None

from spot_bot.features import FeatureConfig, compute_features
from spot_bot.persist import SQLiteLogger
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run regime engine on latest bar.")
    parser.add_argument("--csv", type=str, help="Path to OHLCV CSV (uses loader utilities)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to download (if no CSV provided)")
    parser.add_argument("--interval", type=str, default="1h", help="Interval to download (if no CSV provided)")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles to download")
    parser.add_argument("--db", type=str, default=None, help="Optional SQLite database to log results.")
    parser.add_argument("--base", type=float, default=FeatureConfig.base, help="Log-phase base (default: 10)")
    parser.add_argument("--rv-window", type=int, default=FeatureConfig.rv_window, help="Rolling window for RV")
    parser.add_argument(
        "--conc-window", type=int, default=FeatureConfig.conc_window, help="Rolling window for concentration"
    )
    parser.add_argument("--psi-window", type=int, default=FeatureConfig.psi_window, help="Rolling window for psi")
    parser.add_argument("--s-off", dest="s_off", type=float, default=-0.05, help="Score threshold for OFF")
    parser.add_argument("--s-on", dest="s_on", type=float, default=0.1, help="Score threshold for ON/REDUCE")
    parser.add_argument("--rv-off", dest="rv_off", type=float, default=0.05, help="Vol threshold for OFF")
    parser.add_argument("--rv-reduce", dest="rv_reduce", type=float, default=0.03, help="Vol threshold for REDUCE")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = SQLiteLogger(args.db) if args.db else None

    df = load_ohlcv(args.csv, args.symbol, args.interval, args.limit)
    feat_cfg = FeatureConfig(
        base=args.base,
        rv_window=args.rv_window,
        conc_window=args.conc_window,
        psi_window=args.psi_window,
    )
    features = compute_features(df, cfg=feat_cfg)
    features = features.dropna(subset=["S", "C"])
    if features.empty:
        raise ValueError(
            "Insufficient data to compute regime features; increase data length or reduce rv/conc/psi windows."
        )

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

    if logger:
        bars_records = df.reset_index().rename(columns={"timestamp": "timestamp"}).to_dict(orient="records")
        logger.log_bars(bars_records)
        feat_records = (
            features.reset_index()
            .rename(columns={"index": "timestamp"})
            .reindex(columns=["timestamp", "rv", "C", "psi", "C_int", "S"])
            .to_dict(orient="records")
        )
        logger.log_features(feat_records)
        latest_ts = features.index[-1]
        logger.log_decision(timestamp=latest_ts, risk_state=decision.risk_state, risk_budget=decision.risk_budget, reason=decision.reason)
        logger.close()

    print("\n=== Regime Decision ===")
    print(f"Risk state : {decision.risk_state}")
    print(f"Risk budget: {decision.risk_budget:.3f}")
    print(f"Reason     : {decision.reason}")
    print("Diagnostics:")
    for k, v in decision.diagnostics.items():
        print(f"  - {k}: {v}")
    latest = features.iloc[-1]
    print("Features (latest):")
    print(
        f"  S={latest['S']:.4f}, C={latest['C']:.4f}, "
        f"C_int={latest.get('C_int', float('nan')):.4f}, "
        f"rv={latest.get('rv', float('nan')):.4f}, "
        f"psi={latest.get('psi', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
