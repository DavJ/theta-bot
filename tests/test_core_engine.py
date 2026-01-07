"""
Tests for core trading engine modules.

These tests validate the unified trading math used across all execution modes.
"""

import math

import pandas as pd
import pytest

from spot_bot.core import (
    EngineParams,
    ExecutionResult,
    MarketBar,
    PortfolioState,
    StrategyOutput,
    TradePlan,
    apply_fill,
    apply_hysteresis,
    compute_cost_per_turnover,
    compute_equity,
    compute_exposure,
    compute_hysteresis_threshold,
    plan_trade,
    run_step,
    run_step_simulated,
    simulate_execution,
    target_base_from_exposure,
)


class TestCostModel:
    """Tests for cost_model.py"""

    def test_cost_model_matches_expected(self):
        """Verify cost computation matches live compute_step definition."""
        # Example from run_live.py line 324
        fee_rate = 0.001
        slippage_bps = 5.0
        spread_bps = 2.0

        cost = compute_cost_per_turnover(fee_rate, slippage_bps, spread_bps)

        # Expected: fee_rate + 2*(slippage_bps/10000) + (spread_bps/10000)
        # = 0.001 + 2*(5/10000) + (2/10000)
        # = 0.001 + 0.001 + 0.0002 = 0.0022
        expected = 0.0022
        assert abs(cost - expected) < 1e-10

    def test_cost_model_zero_slippage_spread(self):
        """Test cost with only fees."""
        cost = compute_cost_per_turnover(0.001, 0.0, 0.0)
        assert cost == 0.001

    def test_cost_model_high_slippage(self):
        """Test cost with high slippage."""
        cost = compute_cost_per_turnover(0.001, 50.0, 0.0)
        # 0.001 + 2*50/10000 = 0.001 + 0.01 = 0.011
        assert abs(cost - 0.011) < 1e-10


class TestHysteresis:
    """Tests for hysteresis.py"""

    def test_hysteresis_threshold_calculation(self):
        """Verify threshold computation."""
        hyst_k = 5.0
        hyst_floor = 0.02
        cost = 0.002
        rv_ref = 0.03
        rv_current = 0.02

        delta_e_min = compute_hysteresis_threshold(hyst_k, hyst_floor, cost, rv_ref, rv_current)

        # Expected: max(0.02, 5.0 * 0.002 * (0.03 / 0.02))
        # = max(0.02, 0.01 * 1.5) = max(0.02, 0.015) = 0.02
        assert abs(delta_e_min - 0.02) < 1e-10

    def test_hysteresis_threshold_high_vol(self):
        """When rv_current > rv_ref, threshold should be lower."""
        delta_e_min = compute_hysteresis_threshold(5.0, 0.01, 0.002, 0.02, 0.04)
        # max(0.01, 5.0 * 0.002 * (0.02 / 0.04)) = max(0.01, 0.01 * 0.5) = 0.01
        assert abs(delta_e_min - 0.01) < 1e-10

    def test_hysteresis_threshold_zero_rv_current(self):
        """Handle zero rv_current gracefully."""
        delta_e_min = compute_hysteresis_threshold(5.0, 0.02, 0.002, 0.03, 0.0)
        # rv_current_safe = 1e-8, so threshold will be very high
        # but should be at least hyst_floor
        assert delta_e_min >= 0.02

    def test_hysteresis_suppression(self):
        """Test trade suppression when delta is small."""
        current_exposure = 0.3
        target_exposure = 0.31
        delta_e_min = 0.02

        final_target, suppressed = apply_hysteresis(current_exposure, target_exposure, delta_e_min)

        # delta = 0.01 < 0.02, should suppress
        assert suppressed
        assert final_target == current_exposure

    def test_hysteresis_no_suppression(self):
        """Test no suppression when delta is large enough."""
        current_exposure = 0.3
        target_exposure = 0.35
        delta_e_min = 0.02

        final_target, suppressed = apply_hysteresis(current_exposure, target_exposure, delta_e_min)

        # delta = 0.05 >= 0.02, should not suppress
        assert not suppressed
        assert final_target == target_exposure


class TestPortfolio:
    """Tests for portfolio.py"""

    def test_compute_equity(self):
        """Test equity calculation."""
        usdt = 500.0
        base = 0.01
        price = 50000.0

        equity = compute_equity(usdt, base, price)
        # 500 + 0.01 * 50000 = 500 + 500 = 1000
        assert equity == 1000.0

    def test_compute_exposure(self):
        """Test exposure calculation."""
        base = 0.01
        price = 50000.0
        equity = 1000.0

        exposure = compute_exposure(base, price, equity)
        # (0.01 * 50000) / 1000 = 500 / 1000 = 0.5
        assert exposure == 0.5

    def test_compute_exposure_zero_equity(self):
        """Test exposure with zero equity."""
        exposure = compute_exposure(0.01, 50000.0, 0.0)
        assert exposure == 0.0

    def test_target_base_from_exposure(self):
        """Test converting exposure to base position."""
        equity = 1000.0
        target_exposure = 0.5
        price = 50000.0

        target_base = target_base_from_exposure(equity, target_exposure, price)
        # (1000 * 0.5) / 50000 = 500 / 50000 = 0.01
        assert abs(target_base - 0.01) < 1e-10

    def test_apply_fill_buy(self):
        """Test applying a BUY fill to portfolio."""
        portfolio = PortfolioState(usdt=1000.0, base=0.0, equity=1000.0, exposure=0.0)

        execution = ExecutionResult(
            filled_base=0.01,
            avg_price=50000.0,
            fee_paid=5.0,
            slippage_paid=2.5,
            status="filled",
        )

        updated = apply_fill(portfolio, execution)

        # USDT: 1000 - (0.01 * 50000) - 5 - 2.5 = 1000 - 500 - 7.5 = 492.5
        # Base: 0 + 0.01 = 0.01
        assert abs(updated.usdt - 492.5) < 1e-10
        assert abs(updated.base - 0.01) < 1e-10

    def test_apply_fill_sell(self):
        """Test applying a SELL fill to portfolio."""
        portfolio = PortfolioState(usdt=500.0, base=0.01, equity=1000.0, exposure=0.5)

        execution = ExecutionResult(
            filled_base=-0.005,  # Selling 0.005 BTC
            avg_price=50000.0,
            fee_paid=2.5,
            slippage_paid=1.25,
            status="filled",
        )

        updated = apply_fill(portfolio, execution)

        # USDT: 500 + (0.005 * 50000) - 2.5 - 1.25 = 500 + 250 - 3.75 = 746.25
        # Base: 0.01 - 0.005 = 0.005
        assert abs(updated.usdt - 746.25) < 1e-10
        assert abs(updated.base - 0.005) < 1e-10

    def test_apply_fill_skipped(self):
        """Test that SKIPPED execution doesn't change portfolio."""
        portfolio = PortfolioState(usdt=1000.0, base=0.01, equity=1500.0, exposure=0.333)

        execution = ExecutionResult(
            filled_base=0.0,
            avg_price=50000.0,
            fee_paid=0.0,
            slippage_paid=0.0,
            status="SKIPPED",
        )

        updated = apply_fill(portfolio, execution)

        assert updated.usdt == portfolio.usdt
        assert updated.base == portfolio.base


class TestTradePlanner:
    """Tests for trade_planner.py"""

    def test_rounding_step_size_floor_toward_zero(self):
        """Test that rounding floors toward zero."""
        portfolio = PortfolioState(usdt=1000.0, base=0.0, equity=1000.0, exposure=0.0)

        # Target: 0.006789 BTC, step_size: 0.001
        # Should round down to 0.006
        plan = plan_trade(
            portfolio=portfolio,
            price=50000.0,
            target_exposure=0.3394,  # Would give ~0.006788 BTC
            min_notional=10.0,
            step_size=0.001,
        )

        # Expected: floor(0.006788 / 0.001) * 0.001 = 6 * 0.001 = 0.006
        assert abs(plan.delta_base - 0.006) < 1e-10

    def test_rounding_negative_quantity(self):
        """Test rounding for SELL orders."""
        portfolio = PortfolioState(usdt=500.0, base=0.01, equity=1000.0, exposure=0.5)

        # Sell down to 0.003 with step_size 0.001
        # delta = 0.003 - 0.01 = -0.007
        plan = plan_trade(
            portfolio=portfolio,
            price=50000.0,
            target_exposure=0.15,  # 0.003 BTC
            min_notional=10.0,
            step_size=0.001,
        )

        assert abs(plan.delta_base - (-0.007)) < 1e-10
        assert plan.action == "SELL"

    def test_min_notional_guard(self):
        """Test that trades below min_notional are rejected."""
        portfolio = PortfolioState(usdt=1000.0, base=0.0, equity=1000.0, exposure=0.0)

        # Very small trade: 0.0001 BTC at 50000 = $5 notional
        plan = plan_trade(
            portfolio=portfolio,
            price=50000.0,
            target_exposure=0.00025,  # 0.00005 BTC
            min_notional=10.0,
        )

        assert plan.action == "HOLD"
        assert plan.reason == "min_notional"

    def test_reserve_guard(self):
        """Test USDT reserve guard prevents over-buying."""
        portfolio = PortfolioState(usdt=100.0, base=0.0, equity=100.0, exposure=0.0)

        # Try to buy 0.002 BTC at 50000 = $100, but need to keep 50 USDT reserve
        plan = plan_trade(
            portfolio=portfolio,
            price=50000.0,
            target_exposure=1.0,  # Max exposure
            min_notional=10.0,
            min_usdt_reserve=50.0,
        )

        # Should be able to spend at most 50 USDT (100 - 50)
        # 50 / 50000 = 0.001 BTC
        assert plan.action in ("BUY", "HOLD")
        if plan.action == "BUY":
            max_spend = 100.0 - 50.0
            notional = abs(plan.delta_base) * 50000.0
            assert notional <= max_spend

    def test_max_notional_per_trade_cap(self):
        """Test that max_notional_per_trade caps trade size."""
        portfolio = PortfolioState(usdt=10000.0, base=0.0, equity=10000.0, exposure=0.0)

        # Try to buy 0.1 BTC at 50000 = $5000, but cap at $1000
        plan = plan_trade(
            portfolio=portfolio,
            price=50000.0,
            target_exposure=0.5,  # 0.1 BTC
            min_notional=10.0,
            max_notional_per_trade=1000.0,
        )

        # Should be capped at 1000 / 50000 = 0.02 BTC
        assert plan.action == "BUY"
        assert plan.notional <= 1000.0 + 1e-6  # Allow small floating point error
        assert abs(plan.delta_base - 0.02) < 1e-6

    def test_clamp_exposure_for_spot(self):
        """Test that negative exposure is clamped to 0 for spot."""
        portfolio = PortfolioState(usdt=1000.0, base=0.01, equity=1500.0, exposure=0.333)

        plan = plan_trade(
            portfolio=portfolio,
            price=50000.0,
            target_exposure=-0.5,  # Negative exposure not allowed
            min_notional=10.0,
            allow_short=False,
        )

        # Should clamp to 0, meaning sell all base
        assert plan.target_exposure == 0.0


class TestEngine:
    """Tests for engine.py"""

    def test_simulate_execution_buy(self):
        """Test simulated BUY execution."""
        plan = TradePlan(
            action="BUY",
            target_exposure=0.5,
            target_base=0.01,
            delta_base=0.01,
            notional=500.0,
            exec_price_hint=50000.0,
            reason="test",
        )

        params = EngineParams(fee_rate=0.001, slippage_bps=5.0)

        execution = simulate_execution(plan, 50000.0, params)

        # Exec price: 50000 * (1 + 5/10000) = 50000 * 1.0005 = 50025
        assert abs(execution.avg_price - 50025.0) < 1e-6
        assert execution.filled_base == 0.01
        # Fee: 0.01 * 50025 * 0.001 = 0.50025
        assert abs(execution.fee_paid - 0.50025) < 1e-6
        # Slippage: abs(50025 - 50000) * 0.01 = 25 * 0.01 = 0.25
        assert abs(execution.slippage_paid - 0.25) < 1e-6
        assert execution.status == "filled"

    def test_simulate_execution_sell(self):
        """Test simulated SELL execution."""
        plan = TradePlan(
            action="SELL",
            target_exposure=0.0,
            target_base=0.0,
            delta_base=-0.01,
            notional=500.0,
            exec_price_hint=50000.0,
            reason="test",
        )

        params = EngineParams(fee_rate=0.001, slippage_bps=5.0)

        execution = simulate_execution(plan, 50000.0, params)

        # Exec price: 50000 * (1 - 5/10000) = 50000 * 0.9995 = 49975
        assert abs(execution.avg_price - 49975.0) < 1e-6
        assert execution.filled_base == -0.01
        assert execution.status == "filled"

    def test_simulate_execution_hold(self):
        """Test that HOLD plan returns SKIPPED execution."""
        plan = TradePlan(
            action="HOLD",
            target_exposure=0.5,
            target_base=0.01,
            delta_base=0.0,
            notional=0.0,
            exec_price_hint=50000.0,
            reason="hysteresis",
        )

        params = EngineParams()

        execution = simulate_execution(plan, 50000.0, params)

        assert execution.status == "SKIPPED"
        assert execution.filled_base == 0.0
        assert execution.fee_paid == 0.0

    def test_engine_step_consistency_smoke(self):
        """Smoke test for engine consistency."""

        # Mock strategy that always returns 0.5 exposure
        class FixedStrategy:
            def generate_intent(self, features_df):
                return StrategyOutput(target_exposure=0.5)

        portfolio = PortfolioState(usdt=1000.0, base=0.0, equity=1000.0, exposure=0.0)
        bar = MarketBar(ts=1000, open=50000.0, high=51000.0, low=49000.0, close=50000.0, volume=10.0)
        features_df = pd.DataFrame({"close": [50000.0]})
        strategy = FixedStrategy()
        params = EngineParams(fee_rate=0.001, slippage_bps=5.0, hyst_k=5.0, hyst_floor=0.02, min_notional=10.0)

        plan, strategy_output, diagnostics = run_step(
            bar=bar,
            features_df=features_df,
            portfolio=portfolio,
            strategy=strategy,
            params=params,
            rv_current=0.03,
            rv_ref=0.03,
        )

        # Should plan a BUY since target_exposure=0.5 and current=0
        assert plan.action == "BUY"
        assert plan.target_exposure == 0.5
        assert strategy_output.target_exposure == 0.5

    def test_run_step_simulated_full_cycle(self):
        """Test full simulated step including portfolio update."""

        class FixedStrategy:
            def generate_intent(self, features_df):
                return StrategyOutput(target_exposure=0.3)

        portfolio = PortfolioState(usdt=1000.0, base=0.0, equity=1000.0, exposure=0.0)
        bar = MarketBar(ts=1000, open=50000.0, high=51000.0, low=49000.0, close=50000.0, volume=10.0)
        features_df = pd.DataFrame({"close": [50000.0]})
        strategy = FixedStrategy()
        params = EngineParams(
            fee_rate=0.001,
            slippage_bps=5.0,
            min_notional=10.0,
            hyst_floor=0.0,  # Disable hysteresis for this test
        )

        plan, execution, updated_portfolio, diagnostics = run_step_simulated(
            bar=bar,
            features_df=features_df,
            portfolio=portfolio,
            strategy=strategy,
            params=params,
            rv_current=0.03,
            rv_ref=0.03,
        )

        # Should have bought some BTC
        assert plan.action == "BUY"
        assert execution.status == "filled"
        assert updated_portfolio.base > 0.0
        assert updated_portfolio.usdt < portfolio.usdt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
