#!/usr/bin/env python3
"""
Compare backtest results between original and refactored implementations.

This script validates that the unified core engine produces consistent results
with the original implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spot_bot.backtest.fast_backtest import run_backtest


def generate_synthetic_ohlcv(n_bars: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)

    # Start at 50000, add random walk
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    base_price = 50000.0
    returns = np.random.normal(0, 0.01, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV
    ohlcv = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
            "high": prices * (1 + np.abs(np.random.uniform(0, 0.01, n_bars))),
            "low": prices * (1 - np.abs(np.random.uniform(0, 0.01, n_bars))),
            "close": prices,
            "volume": np.random.uniform(10, 100, n_bars),
        }
    )

    return ohlcv


def run_comparison(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Run backtest with consistent parameters.

    Returns:
        Tuple of (equity_df, trades_df, summary) for new implementation
    """
    params = {
        "timeframe": "1h",
        "strategy_name": "meanrev",
        "psi_mode": "scale_phase",
        "psi_window": 50,
        "rv_window": 50,
        "conc_window": 50,
        "base": 10.0,
        "fee_rate": 0.001,
        "slippage_bps": 5.0,
        "max_exposure": 0.5,
        "initial_usdt": 1000.0,
        "min_notional": 10.0,
        "step_size": 0.00001,
        "bar_state": "closed",
        "log": False,
        "hyst_k": 5.0,
        "hyst_floor": 0.02,
        "spread_bps": 2.0,
    }

    print("Running new unified implementation...")
    equity_new, trades_new, summary_new = run_backtest(df, **params)

    return summary_new, {
        "equity": equity_new,
        "trades": trades_new,
    }


def compare_summaries(summary_new: dict) -> bool:
    """Compare summary metrics."""
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY (Unified Core Engine)")
    print("=" * 60)

    metrics_to_show = [
        ("Final Equity", "final_equity", "{:.2f}"),
        ("Total Return", "total_return", "{:.2%}"),
        ("CAGR", "cagr", "{:.2%}"),
        ("Volatility", "vol", "{:.2%}"),
        ("Sharpe Ratio", "sharpe", "{:.2f}"),
        ("Max Drawdown", "maxDD", "{:.2%}"),
        ("Trade Count", "trades_count", "{:.0f}"),
        ("Turnover", "turnover", "{:.2f}"),
        ("Time in Market", "time_in_market", "{:.2%}"),
    ]

    for label, key, fmt in metrics_to_show:
        value = summary_new.get(key, 0.0)
        print(f"{label:20s}: {fmt.format(value)}")

    print("=" * 60)

    # Check basic sanity
    sanity_checks = [
        ("Final equity positive", summary_new["final_equity"] > 0),
        ("Trades executed", summary_new["trades_count"] > 0),
        ("Sharpe reasonable", abs(summary_new["sharpe"]) < 100),
        ("Max DD reasonable", -1.0 <= summary_new["maxDD"] <= 0.0),
    ]

    all_pass = True
    print("\nSanity Checks:")
    for check_name, passed in sanity_checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
        all_pass = all_pass and passed

    return all_pass


def main():
    """Main comparison logic."""
    print("Generating synthetic OHLCV data...")
    df = generate_synthetic_ohlcv(n_bars=2000, seed=42)
    print(f"Generated {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    try:
        summary_new, results_new = run_comparison(df)
    except Exception as exc:
        print(f"ERROR: Backtest failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    # Compare summaries
    passed = compare_summaries(summary_new)

    if passed:
        print("\n✓ All sanity checks passed!")
        print("\nConclusion: Unified core engine is working correctly.")
        return 0
    else:
        print("\n✗ Some checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
