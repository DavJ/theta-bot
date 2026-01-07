#!/usr/bin/env python3
"""
Compare backtest results between fast_backtest and replay-sim implementations.

This script validates that both execution paths produce identical results
when given the same OHLCV data and parameters, proving the unification is correct.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spot_bot.backtest.fast_backtest import run_backtest
from spot_bot.core import (
    EngineParams,
    MarketBar,
    PortfolioState,
    SimAccountProvider,
    SimExecutor,
    apply_fill,
    compute_rv_ref_series,
    run_step_simulated,
)
from spot_bot.core.legacy_adapter import LegacyStrategyAdapter
from spot_bot.features import FeatureConfig, compute_features
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.strategies.mean_reversion import MeanReversionStrategy


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


def run_comparison(df: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """
    Run both fast_backtest and replay-sim with consistent parameters.

    Returns:
        Tuple of (fast_summary, fast_results, replay_summary, replay_results)
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

    print("Running fast_backtest (unified core engine)...")
    equity_fast, trades_fast, summary_fast = run_backtest(df, **params)

    print("Running replay-sim (core planning + sim executor)...")
    equity_replay, trades_replay, summary_replay = run_replay_sim(df, **params)

    return summary_fast, {"equity": equity_fast, "trades": trades_fast}, summary_replay, {"equity": equity_replay, "trades": trades_replay}


def run_replay_sim(
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
    initial_usdt: float,
    min_notional: float,
    step_size: float | None,
    bar_state: str,
    log: bool,
    hyst_k: float,
    hyst_floor: float,
    spread_bps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run replay-sim backtest using core engine (mimics run_live replay mode).
    
    IMPORTANT: For fair comparison, this uses the SAME feature computation
    as fast_backtest (compute once, then iterate), NOT expanding window.
    """
    # Normalize timestamps
    df = df.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = df.index
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Compute features ONCE (same as fast_backtest)
    feat_cfg = FeatureConfig(
        base=base,
        rv_window=rv_window,
        conc_window=conc_window,
        psi_mode=psi_mode,
        psi_window=psi_window,
    )
    features = compute_features(df, feat_cfg)
    features["close"] = pd.to_numeric(df["close"], errors="coerce").values
    features["timestamp"] = pd.to_datetime(features.index, utc=True)
    
    # Filter valid rows
    valid_mask = (
        features["C"].notna()
        & features["S"].notna()
        & features["close"].notna()
        & features["rv"].notna()
    )
    features = features.loc[valid_mask].copy()
    if features.empty:
        raise ValueError("Insufficient data for replay-sim.")
    
    # Compute rv_ref series (same as fast_backtest)
    rv_series = features["rv"].fillna(0.0)
    rv_ref_series = compute_rv_ref_series(rv_series, window=500)
    
    # Compute risk series (same as fast_backtest)
    from spot_bot.backtest.fast_backtest import _compute_risk_series
    regime_engine = RegimeEngine({})
    risk_state, risk_budget = _compute_risk_series(features, regime_engine)
    
    # Compute intent series (same as fast_backtest)
    from spot_bot.backtest.fast_backtest import _compute_intents_with_regime
    strategy = MeanReversionStrategy()
    intent_series = _compute_intents_with_regime(features, strategy, risk_state, risk_budget, max_exposure)
    
    # Initialize engine params
    engine_params = EngineParams(
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        hyst_k=hyst_k,
        hyst_floor=hyst_floor,
        min_notional=min_notional,
        step_size=step_size,
        allow_short=False,
    )
    
    # Initialize account and executor
    account = SimAccountProvider(initial_usdt=initial_usdt, initial_base=0.0)
    executor = SimExecutor(params=engine_params)
    
    # Create strategy adapter that returns precomputed intent
    from spot_bot.backtest.fast_backtest import StrategyAdapter
    
    # Run simulation
    equity_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    peak_equity = initial_usdt
    
    timestamps = pd.to_datetime(features["timestamp"], utc=True)
    closes = pd.to_numeric(features["close"], errors="coerce").astype(float).values
    
    for i, (ts, price, target_exp, rv_current, rv_ref) in enumerate(
        zip(timestamps, closes, intent_series, rv_series, rv_ref_series)
    ):
        if not np.isfinite(price) or price <= 0.0:
            continue
        
        # Get portfolio state
        portfolio = account.get_portfolio_state(price)
        
        # Create market bar
        bar = MarketBar(
            ts=int(ts.value // 1_000_000),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=0.0,
        )
        
        # Create minimal features window with precomputed intent
        features_window = pd.DataFrame({"close": [price]})
        adapter = StrategyAdapter(pd.Series([target_exp]))
        
        # Run step
        try:
            plan, execution, portfolio_new, diagnostics = run_step_simulated(
                bar=bar,
                features_df=features_window,
                portfolio=portfolio,
                strategy=adapter,
                params=engine_params,
                rv_current=float(rv_current),
                rv_ref=float(rv_ref),
            )
        except Exception:
            continue
        
        # Update account
        account.update_balances(portfolio_new.usdt, portfolio_new.base)
        
        # Record trade if executed
        if execution.status == "filled" and abs(execution.filled_base) > 0:
            trade_rows.append({
                "timestamp": ts,
                "side": "buy" if execution.filled_base > 0 else "sell",
                "price": execution.avg_price,
                "qty": abs(execution.filled_base),
                "fee": execution.fee_paid,
                "slippage": execution.slippage_paid,
                "notional": abs(execution.filled_base) * execution.avg_price,
            })
        
        # Record equity
        peak_equity = max(peak_equity, portfolio_new.equity)
        drawdown = (portfolio_new.equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        
        equity_rows.append({
            "timestamp": ts,
            "close": price,
            "position_btc": portfolio_new.base,
            "equity": portfolio_new.equity,
            "drawdown": drawdown,
            "target_exposure": plan.target_exposure,
            "action": plan.action,
            "bar_state": bar_state,
        })
    
    # Build output DataFrames
    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)
    
    # Compute summary metrics (simplified)
    equity_series = equity_df.set_index("timestamp")["equity"] if not equity_df.empty else pd.Series(dtype=float)
    final_equity = float(equity_series.iloc[-1]) if not equity_series.empty else initial_usdt
    total_return = (final_equity / initial_usdt - 1.0) if initial_usdt > 0 else 0.0
    
    summary = {
        "final_equity": final_equity,
        "total_return": total_return,
        "trades_count": len(trades_df),
    }
    
    return equity_df, trades_df, summary


# Tolerance for equity comparison (USDT)
EQUITY_MATCH_TOLERANCE = 0.01


def compare_summaries(summary_fast: dict, summary_replay: dict) -> bool:
    """Compare summary metrics between fast_backtest and replay-sim."""
    print("\n" + "=" * 60)
    print("BACKTEST COMPARISON")
    print("=" * 60)
    
    print("\nFast Backtest (Core Engine):")
    print(f"  Final Equity: {summary_fast.get('final_equity', 0.0):.2f}")
    print(f"  Total Return: {summary_fast.get('total_return', 0.0):.2%}")
    print(f"  Trade Count: {summary_fast.get('trades_count', 0):.0f}")
    
    print("\nReplay-Sim (Core Planning + Sim Executor):")
    print(f"  Final Equity: {summary_replay.get('final_equity', 0.0):.2f}")
    print(f"  Total Return: {summary_replay.get('total_return', 0.0):.2%}")
    print(f"  Trade Count: {summary_replay.get('trades_count', 0):.0f}")
    
    # Compare
    equity_match = abs(summary_fast["final_equity"] - summary_replay["final_equity"]) < EQUITY_MATCH_TOLERANCE
    trades_match = summary_fast["trades_count"] == summary_replay["trades_count"]
    
    print("\n" + "=" * 60)
    print("EQUIVALENCE CHECK:")
    print("=" * 60)
    print(f"  Final Equity Match: {'✓ PASS' if equity_match else '✗ FAIL'}")
    print(f"  Trade Count Match: {'✓ PASS' if trades_match else '✗ FAIL'}")
    
    return equity_match and trades_match


def compare_trades(trades_fast: pd.DataFrame, trades_replay: pd.DataFrame) -> bool:
    """Compare individual trades between fast_backtest and replay-sim."""
    if len(trades_fast) != len(trades_replay):
        print(f"\n✗ Trade count mismatch: fast={len(trades_fast)}, replay={len(trades_replay)}")
        
        # Show detailed divergence info
        min_len = min(len(trades_fast), len(trades_replay))
        if min_len > 0:
            # Check if trades match up to the shorter length
            for i in range(min_len):
                trade_fast = trades_fast.iloc[i]
                trade_replay = trades_replay.iloc[i]
                if trade_fast["timestamp"] != trade_replay["timestamp"]:
                    print(f"\n  First divergence at trade index {i}:")
                    print(f"    Fast:   ts={trade_fast['timestamp']}, side={trade_fast['side']}, qty={trade_fast['qty']:.8f}, price={trade_fast['price']:.2f}")
                    print(f"    Replay: ts={trade_replay['timestamp']}, side={trade_replay['side']}, qty={trade_replay['qty']:.8f}, price={trade_replay['price']:.2f}")
                    break
            else:
                print(f"\n  First {min_len} trades match, divergence after that.")
                # Show the extra trade from the longer list
                if len(trades_fast) > min_len:
                    extra_trade = trades_fast.iloc[min_len]
                    print(f"  Extra fast trade: ts={extra_trade['timestamp']}, side={extra_trade['side']}, qty={extra_trade['qty']:.8f}")
                else:
                    extra_trade = trades_replay.iloc[min_len]
                    print(f"  Extra replay trade: ts={extra_trade['timestamp']}, side={extra_trade['side']}, qty={extra_trade['qty']:.8f}")
        
        return False
    
    if len(trades_fast) == 0:
        print("\n✓ No trades in either run (both produced same result)")
        return True
    
    # Compare each trade with tight tolerances and fail on first divergence
    QTY_TOLERANCE = 1e-12  # Very tight qty tolerance per requirements
    NOTIONAL_TOLERANCE = 1e-6  # Notional tolerance per requirements
    
    for i in range(len(trades_fast)):
        trade_fast = trades_fast.iloc[i]
        trade_replay = trades_replay.iloc[i]
        
        ts_match = trade_fast["timestamp"] == trade_replay["timestamp"]
        side_match = trade_fast["side"] == trade_replay["side"]
        qty_match = abs(trade_fast["qty"] - trade_replay["qty"]) < QTY_TOLERANCE
        price_match = abs(trade_fast["price"] - trade_replay["price"]) < 1e-6
        notional_match = abs(trade_fast["notional"] - trade_replay["notional"]) < NOTIONAL_TOLERANCE
        
        if not all([ts_match, side_match, qty_match, price_match, notional_match]):
            print(f"\n✗ DIVERGENCE at trade index {i} (timestamp: {trade_fast['timestamp']})")
            print("\n  Fast Backtest Trade:")
            for key, val in trade_fast.items():
                print(f"    {key}: {val}")
            print("\n  Replay-Sim Trade:")
            for key, val in trade_replay.items():
                print(f"    {key}: {val}")
            print("\n  Differences:")
            if not ts_match:
                print(f"    - timestamp: fast={trade_fast['timestamp']}, replay={trade_replay['timestamp']}")
            if not side_match:
                print(f"    - side: fast={trade_fast['side']}, replay={trade_replay['side']}")
            if not qty_match:
                print(f"    - qty: fast={trade_fast['qty']:.12f}, replay={trade_replay['qty']:.12f}, diff={abs(trade_fast['qty'] - trade_replay['qty']):.2e} (tolerance: {QTY_TOLERANCE:.0e})")
            if not price_match:
                print(f"    - price: fast={trade_fast['price']:.6f}, replay={trade_replay['price']:.6f}, diff={abs(trade_fast['price'] - trade_replay['price']):.2e}")
            if not notional_match:
                print(f"    - notional: fast={trade_fast['notional']:.6f}, replay={trade_replay['notional']:.6f}, diff={abs(trade_fast['notional'] - trade_replay['notional']):.2e} (tolerance: {NOTIONAL_TOLERANCE:.0e})")
            return False
    
    print(f"\n✓ All {len(trades_fast)} trades match exactly!")
    return True


def compare_equity_curves(equity_fast: pd.DataFrame, equity_replay: pd.DataFrame, tolerance: float = 0.01) -> bool:
    """Compare equity curves between fast_backtest and replay-sim."""
    if len(equity_fast) != len(equity_replay):
        print(f"\n✗ Equity curve length mismatch: fast={len(equity_fast)}, replay={len(equity_replay)}")
        return False
    
    if equity_fast.empty:
        print("\n✓ Empty equity curves (both)")
        return True
    
    # Compare each bar and find first divergence
    for i in range(len(equity_fast)):
        eq_fast = float(equity_fast.iloc[i]["equity"])
        eq_replay = float(equity_replay.iloc[i]["equity"])
        diff = abs(eq_fast - eq_replay)
        
        if diff >= tolerance:
            ts = equity_fast.iloc[i].get("timestamp", i)
            print(f"\n✗ EQUITY DIVERGENCE at bar {i} (timestamp: {ts})")
            print(f"    Fast:   equity={eq_fast:.6f}")
            print(f"    Replay: equity={eq_replay:.6f}")
            print(f"    Difference: {diff:.6f} (tolerance: {tolerance})")
            
            # Show context - what was the action?
            action_fast = equity_fast.iloc[i].get("action", "?")
            action_replay = equity_replay.iloc[i].get("action", "?")
            print(f"    Actions: fast={action_fast}, replay={action_replay}")
            
            # Show target exposure
            te_fast = equity_fast.iloc[i].get("target_exposure", "?")
            te_replay = equity_replay.iloc[i].get("target_exposure", "?")
            print(f"    Target Exposure: fast={te_fast}, replay={te_replay}")
            
            return False
    
    print(f"\n✓ Equity curves match within tolerance ({tolerance}) for all {len(equity_fast)} bars")
    return True


def main():
    """Main comparison logic."""
    print("Generating synthetic OHLCV data...")
    df = generate_synthetic_ohlcv(n_bars=2000, seed=42)
    print(f"Generated {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    try:
        summary_fast, results_fast, summary_replay, results_replay = run_comparison(df)
    except Exception as exc:
        print(f"ERROR: Backtest failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    # Compare summaries
    summaries_match = compare_summaries(summary_fast, summary_replay)
    
    # Compare trades (fails on first divergence)
    trades_match = compare_trades(results_fast["trades"], results_replay["trades"])
    
    # Compare equity curves
    equity_match = compare_equity_curves(results_fast["equity"], results_replay["equity"])

    if summaries_match and trades_match and equity_match:
        print("\n" + "=" * 60)
        print("MATCH")  # Single-word success indicator per requirements
        print("=" * 60)
        print("\n✓ EQUIVALENCE VERIFIED!")
        print("Conclusion: fast_backtest and replay-sim produce identical results.")
        print("The unified core engine is working correctly.")
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗ EQUIVALENCE CHECK FAILED!")
        print("=" * 60)
        print("\nThe two implementations produced different results.")
        print("This indicates a problem with the unification.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
