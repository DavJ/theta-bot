"""
Test equivalence between fast_backtest and replay-sim execution paths.

This test verifies that both paths produce identical results when given
the same OHLCV data and parameters, proving the unification is correct.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from spot_bot.backtest.fast_backtest import StrategyAdapter, run_backtest
from spot_bot.core import (
    EngineParams,
    MarketBar,
    PortfolioState,
    SimAccountProvider,
    compute_rv_ref_series,
    run_step_simulated,
)
from spot_bot.features import FeatureConfig, compute_features
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.strategies.mean_reversion import MeanReversionStrategy


def generate_synthetic_ohlcv(n_bars: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)

    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    base_price = 50000.0
    returns = np.random.normal(0, 0.01, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

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


def run_replay_sim(
    df: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run replay-sim backtest using core engine.
    
    This mimics the run_live replay mode but uses core simulation.
    """
    from spot_bot.backtest.fast_backtest import (
        _compute_intents_with_regime,
        _compute_risk_series,
    )

    # Normalize timestamps
    df = df.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = df.index
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute features ONCE (same as fast_backtest)
    feat_cfg = FeatureConfig(
        base=params["base"],
        rv_window=params["rv_window"],
        conc_window=params["conc_window"],
        psi_mode=params["psi_mode"],
        psi_window=params["psi_window"],
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
    regime_engine = RegimeEngine({})
    risk_state, risk_budget = _compute_risk_series(features, regime_engine)

    # Compute intent series (same as fast_backtest)
    strategy = MeanReversionStrategy()
    intent_series = _compute_intents_with_regime(
        features, strategy, risk_state, risk_budget, params["max_exposure"]
    )

    # Initialize engine params
    engine_params = EngineParams(
        fee_rate=params["fee_rate"],
        slippage_bps=params["slippage_bps"],
        spread_bps=params["spread_bps"],
        hyst_k=params["hyst_k"],
        hyst_floor=params["hyst_floor"],
        min_notional=params["min_notional"],
        step_size=params.get("step_size"),
        allow_short=False,
    )

    # Initialize account
    account = SimAccountProvider(initial_usdt=params["initial_usdt"], initial_base=0.0)

    # Run simulation
    equity_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []

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
            trade_rows.append(
                {
                    "timestamp": ts,
                    "side": "buy" if execution.filled_base > 0 else "sell",
                    "price": execution.avg_price,
                    "qty": abs(execution.filled_base),
                    "fee": execution.fee_paid,
                    "notional": abs(execution.filled_base) * execution.avg_price,
                }
            )

        # Record equity
        equity_rows.append(
            {
                "timestamp": ts,
                "close": price,
                "position_btc": portfolio_new.base,
                "equity": portfolio_new.equity,
                "target_exposure": plan.target_exposure,
                "action": plan.action,
            }
        )

    # Build output DataFrames
    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)

    # Compute summary metrics
    equity_series = (
        equity_df.set_index("timestamp")["equity"]
        if not equity_df.empty
        else pd.Series(dtype=float)
    )
    final_equity = (
        float(equity_series.iloc[-1]) if not equity_series.empty else params["initial_usdt"]
    )
    total_return = (final_equity / params["initial_usdt"] - 1.0) if params["initial_usdt"] > 0 else 0.0

    summary = {
        "final_equity": final_equity,
        "total_return": total_return,
        "trades_count": len(trades_df),
    }

    return equity_df, trades_df, summary


class TestEquivalenceFastVsReplaySim:
    """Test equivalence between fast_backtest and replay-sim."""

    @pytest.fixture
    def test_params(self):
        """Standard test parameters."""
        return {
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

    def test_trade_count_matches(self, test_params):
        """Both paths should produce the same number of trades."""
        df = generate_synthetic_ohlcv(n_bars=200, seed=42)

        equity_fast, trades_fast, summary_fast = run_backtest(df, **test_params)
        equity_replay, trades_replay, summary_replay = run_replay_sim(df, test_params)

        assert summary_fast["trades_count"] == summary_replay["trades_count"], (
            f"Trade count mismatch: fast={summary_fast['trades_count']}, "
            f"replay={summary_replay['trades_count']}"
        )

    def test_final_equity_matches(self, test_params):
        """Both paths should produce the same final equity."""
        df = generate_synthetic_ohlcv(n_bars=200, seed=42)

        equity_fast, trades_fast, summary_fast = run_backtest(df, **test_params)
        equity_replay, trades_replay, summary_replay = run_replay_sim(df, test_params)

        tolerance = 0.01  # Allow 1 cent difference
        equity_diff = abs(summary_fast["final_equity"] - summary_replay["final_equity"])
        assert equity_diff < tolerance, (
            f"Final equity mismatch: fast={summary_fast['final_equity']:.6f}, "
            f"replay={summary_replay['final_equity']:.6f}, diff={equity_diff:.6f}"
        )

    def test_trades_match_exactly(self, test_params):
        """Each trade should match exactly between both paths."""
        df = generate_synthetic_ohlcv(n_bars=200, seed=42)

        equity_fast, trades_fast, summary_fast = run_backtest(df, **test_params)
        equity_replay, trades_replay, summary_replay = run_replay_sim(df, test_params)

        if len(trades_fast) == 0:
            assert len(trades_replay) == 0
            return

        assert len(trades_fast) == len(trades_replay)

        for i in range(len(trades_fast)):
            trade_fast = trades_fast.iloc[i]
            trade_replay = trades_replay.iloc[i]

            # Check timestamp
            assert trade_fast["timestamp"] == trade_replay["timestamp"], (
                f"Trade {i} timestamp mismatch"
            )

            # Check side
            assert trade_fast["side"] == trade_replay["side"], f"Trade {i} side mismatch"

            # Check quantity with tolerance
            qty_diff = abs(trade_fast["qty"] - trade_replay["qty"])
            assert qty_diff < 1e-10, f"Trade {i} qty mismatch: {qty_diff}"

            # Check price with tolerance
            price_diff = abs(trade_fast["price"] - trade_replay["price"])
            assert price_diff < 1e-6, f"Trade {i} price mismatch: {price_diff}"

    def test_different_seeds_produce_different_results(self, test_params):
        """Sanity check that different data produces different results."""
        df1 = generate_synthetic_ohlcv(n_bars=200, seed=42)
        df2 = generate_synthetic_ohlcv(n_bars=200, seed=123)

        equity1, trades1, summary1 = run_backtest(df1, **test_params)
        equity2, trades2, summary2 = run_backtest(df2, **test_params)

        # Results should be different for different seeds
        assert summary1["final_equity"] != summary2["final_equity"], (
            "Different seeds should produce different results"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
