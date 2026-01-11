"""
Tests for hysteresis diagnostics, return threshold, limit fill simulation, and sell guard.

These tests verify:
- Hysteresis cap binding diagnostics
- Limit fill simulation with OHLC
- Sell guard preventing sales below entry + threshold
"""
import pytest
from spot_bot.core.hysteresis import (
    compute_hysteresis_threshold,
    compute_return_threshold,
)
from spot_bot.core.engine import simulate_execution, EngineParams
from spot_bot.core.trade_planner import plan_trade
from spot_bot.core.types import MarketBar, TradePlan, PortfolioState


class TestHysteresisDiagnostics:
    """Test hysteresis diagnostics including cap/floor binding flags."""
    
    def test_cap_binding_when_raw_exceeds_max(self):
        """When raw hysteresis exceeds max_delta_e_min, cap should bind."""
        # Set parameters to make raw >> cap
        hyst_k = 50.0  # Large multiplier
        max_delta_e_min = 0.1  # Low cap
        hyst_floor = 0.01
        
        result = compute_hysteresis_threshold(
            rv_current=0.05,
            rv_ref=0.05,
            fee_rate=0.001,
            slippage_bps=5.0,
            spread_bps=5.0,
            hyst_k=hyst_k,
            hyst_floor=hyst_floor,
            k_vol=0.5,
            edge_bps=5.0,
            max_delta_e_min=max_delta_e_min,
            alpha_floor=6.0,
            alpha_cap=6.0,
            vol_hyst_mode="increase",
            return_diagnostics=True,
        )
        
        delta_e_min, diagnostics = result
        
        # Cap should be binding
        assert diagnostics["cap_binding"], "Cap should bind when raw >> max"
        # Result should be close to max (soft clamp allows some deviation)
        assert abs(delta_e_min - max_delta_e_min) < 0.03, "Result should be near cap"
        # Raw should be larger than final
        assert diagnostics["hyst_raw"] > delta_e_min, "Raw should exceed final"
    
    def test_floor_binding_when_raw_below_floor(self):
        """When raw hysteresis is below floor, floor should bind."""
        # Set parameters to make raw << floor
        hyst_k = 0.1  # Very small multiplier
        hyst_floor = 0.05  # High floor
        max_delta_e_min = 0.3
        
        result = compute_hysteresis_threshold(
            rv_current=0.01,
            rv_ref=0.05,
            fee_rate=0.0001,
            slippage_bps=0.0,
            spread_bps=0.0,
            hyst_k=hyst_k,
            hyst_floor=hyst_floor,
            k_vol=0.5,
            edge_bps=1.0,
            max_delta_e_min=max_delta_e_min,
            alpha_floor=6.0,
            alpha_cap=6.0,
            vol_hyst_mode="increase",
            return_diagnostics=True,
        )
        
        delta_e_min, diagnostics = result
        
        # Floor should be binding
        assert diagnostics["floor_binding"], "Floor should bind when raw << floor"
        # Result should be close to floor
        assert abs(delta_e_min - hyst_floor) < 0.01, "Result should be near floor"
        # Raw should be smaller than final
        assert diagnostics["hyst_raw"] < delta_e_min, "Raw should be below final"
    
    def test_neither_binding_in_middle(self):
        """When raw is between floor and cap, neither should bind."""
        # Moderate parameters
        hyst_k = 5.0
        hyst_floor = 0.02
        max_delta_e_min = 0.3
        
        result = compute_hysteresis_threshold(
            rv_current=0.05,
            rv_ref=0.05,
            fee_rate=0.001,
            slippage_bps=1.0,
            spread_bps=1.0,
            hyst_k=hyst_k,
            hyst_floor=hyst_floor,
            k_vol=0.5,
            edge_bps=5.0,
            max_delta_e_min=max_delta_e_min,
            alpha_floor=6.0,
            alpha_cap=6.0,
            vol_hyst_mode="increase",
            return_diagnostics=True,
        )
        
        delta_e_min, diagnostics = result
        
        # Neither should be binding (soft constraints allow some fuzziness)
        # We expect raw to be in a reasonable range
        assert hyst_floor - 0.01 < delta_e_min < max_delta_e_min + 0.01


class TestReturnThreshold:
    """Test return threshold computation."""
    
    def test_return_threshold_includes_all_costs(self):
        """Return threshold should include fees, spread, slippage, edge, and min profit."""
        rt = compute_return_threshold(
            fee_rate=0.001,  # 0.1% per side = 0.2% round trip
            spread_bps=5.0,
            slippage_bps=5.0,
            edge_bps=5.0,
            min_profit_bps=10.0,
            rv_current=0.05,
            rv_ref=0.05,
            k_vol=0.0,
            vol_hyst_mode="none",
        )
        
        # Expected: 2*0.001 + (5+5)*1e-4 + 5*1e-4 + 10*1e-4
        #         = 0.002 + 0.001 + 0.0005 + 0.001 = 0.0045
        expected = 0.0045
        assert abs(rt - expected) < 1e-6, f"Expected {expected}, got {rt}"
    
    def test_volatility_multiplier_increase_mode(self):
        """In increase mode, higher vol should increase threshold."""
        rt_low_vol = compute_return_threshold(
            fee_rate=0.001,
            spread_bps=5.0,
            slippage_bps=5.0,
            edge_bps=5.0,
            min_profit_bps=5.0,
            rv_current=0.02,
            rv_ref=0.05,
            k_vol=0.5,
            vol_hyst_mode="increase",
        )
        
        rt_high_vol = compute_return_threshold(
            fee_rate=0.001,
            spread_bps=5.0,
            slippage_bps=5.0,
            edge_bps=5.0,
            min_profit_bps=5.0,
            rv_current=0.10,
            rv_ref=0.05,
            k_vol=0.5,
            vol_hyst_mode="increase",
        )
        
        assert rt_high_vol > rt_low_vol, "Higher volatility should increase threshold in increase mode"


class TestLimitFillSimulation:
    """Test OHLC-based limit fill simulation."""
    
    def test_buy_fills_when_low_touches_limit(self):
        """BUY limit order should fill if bar.low <= limit_price."""
        params = EngineParams(fee_rate=0.001, slippage_bps=0.0)
        
        plan = TradePlan(
            action="BUY",
            target_exposure=0.5,
            target_base=0.01,
            delta_base=0.01,
            notional=500.0,
            exec_price_hint=50000.0,
            reason="test",
            limit_price=49900.0,  # Limit to buy at 49900
            order_type="limit",
        )
        
        bar = MarketBar(
            ts=1000000,
            open=50000.0,
            high=50100.0,
            low=49800.0,  # Low touches limit
            close=50000.0,
            volume=1000.0,
        )
        
        result = simulate_execution(plan, 50000.0, params, bar=bar)
        
        assert result.status == "filled", "Order should fill"
        assert result.filled_base == 0.01, "Should fill full quantity"
        assert result.avg_price == 49900.0, "Should fill at limit price"
        assert result.slippage_paid == 0.0, "Limit fill has no slippage"
    
    def test_buy_skipped_when_low_above_limit(self):
        """BUY limit order should be skipped if bar.low > limit_price."""
        params = EngineParams(fee_rate=0.001, slippage_bps=0.0)
        
        plan = TradePlan(
            action="BUY",
            target_exposure=0.5,
            target_base=0.01,
            delta_base=0.01,
            notional=500.0,
            exec_price_hint=50000.0,
            reason="test",
            limit_price=49900.0,  # Limit to buy at 49900
            order_type="limit",
        )
        
        bar = MarketBar(
            ts=1000000,
            open=50000.0,
            high=50100.0,
            low=49950.0,  # Low is above limit
            close=50000.0,
            volume=1000.0,
        )
        
        result = simulate_execution(plan, 50000.0, params, bar=bar)
        
        assert result.status == "SKIPPED", "Order should be skipped"
        assert result.filled_base == 0.0, "Should not fill"
    
    def test_sell_fills_when_high_touches_limit(self):
        """SELL limit order should fill if bar.high >= limit_price."""
        params = EngineParams(fee_rate=0.001, slippage_bps=0.0)
        
        plan = TradePlan(
            action="SELL",
            target_exposure=0.0,
            target_base=0.0,
            delta_base=-0.01,
            notional=500.0,
            exec_price_hint=50000.0,
            reason="test",
            limit_price=50100.0,  # Limit to sell at 50100
            order_type="limit",
        )
        
        bar = MarketBar(
            ts=1000000,
            open=50000.0,
            high=50150.0,  # High touches limit
            low=49900.0,
            close=50000.0,
            volume=1000.0,
        )
        
        result = simulate_execution(plan, 50000.0, params, bar=bar)
        
        assert result.status == "filled", "Order should fill"
        assert result.filled_base == -0.01, "Should fill full quantity"
        assert result.avg_price == 50100.0, "Should fill at limit price"
        assert result.slippage_paid == 0.0, "Limit fill has no slippage"
    
    def test_sell_skipped_when_high_below_limit(self):
        """SELL limit order should be skipped if bar.high < limit_price."""
        params = EngineParams(fee_rate=0.001, slippage_bps=0.0)
        
        plan = TradePlan(
            action="SELL",
            target_exposure=0.0,
            target_base=0.0,
            delta_base=-0.01,
            notional=500.0,
            exec_price_hint=50000.0,
            reason="test",
            limit_price=50100.0,  # Limit to sell at 50100
            order_type="limit",
        )
        
        bar = MarketBar(
            ts=1000000,
            open=50000.0,
            high=50050.0,  # High is below limit
            low=49900.0,
            close=50000.0,
            volume=1000.0,
        )
        
        result = simulate_execution(plan, 50000.0, params, bar=bar)
        
        assert result.status == "SKIPPED", "Order should be skipped"
        assert result.filled_base == 0.0, "Should not fill"


class TestSellGuard:
    """Test sell guard preventing sales below entry + threshold."""
    
    def test_sell_suppressed_below_threshold(self):
        """SELL should be suppressed if price < avg_entry * (1 + return_threshold)."""
        portfolio = PortfolioState(
            usdt=1000.0,
            base=0.02,  # Have position
            equity=2000.0,
            exposure=0.5,
            avg_entry_price=50000.0,  # Entry at 50000
        )
        
        return_threshold = 0.01  # 1%
        current_price = 50400.0  # Only 0.8% above entry, less than 1% threshold
        
        plan = plan_trade(
            portfolio=portfolio,
            price=current_price,
            target_exposure=0.0,  # Want to sell
            min_notional=10.0,
            return_threshold=return_threshold,
        )
        
        # Sell should be suppressed
        assert plan.action == "HOLD", "SELL should be suppressed by sell guard"
        assert plan.reason == "sell_guard", "Reason should indicate sell guard"
        assert plan.diagnostics["sell_guard"] == True
        min_sell_price = portfolio.avg_entry_price * (1.0 + return_threshold)
        assert plan.diagnostics["min_sell_price"] == min_sell_price
    
    def test_sell_allowed_above_threshold(self):
        """SELL should be allowed if price >= avg_entry * (1 + return_threshold)."""
        portfolio = PortfolioState(
            usdt=1000.0,
            base=0.02,  # Have position
            equity=2000.0,
            exposure=0.5,
            avg_entry_price=50000.0,  # Entry at 50000
        )
        
        return_threshold = 0.01  # 1%
        current_price = 50600.0  # 1.2% above entry, exceeds threshold
        
        plan = plan_trade(
            portfolio=portfolio,
            price=current_price,
            target_exposure=0.0,  # Want to sell
            min_notional=10.0,
            return_threshold=return_threshold,
        )
        
        # Sell should be allowed
        assert plan.action == "SELL", "SELL should be allowed above threshold"
        # Limit price should be at least min_sell_price
        min_sell_price = portfolio.avg_entry_price * (1.0 + return_threshold)
        assert plan.limit_price >= min_sell_price, "Limit price should respect min_sell_price"
    
    def test_sell_guard_not_triggered_without_position(self):
        """Sell guard should not trigger if no position (avg_entry_price is None)."""
        portfolio = PortfolioState(
            usdt=2000.0,
            base=0.0,  # No position
            equity=2000.0,
            exposure=0.0,
            avg_entry_price=None,  # No entry
        )
        
        return_threshold = 0.01
        current_price = 50000.0
        
        # This should not trigger sell guard (no position to protect)
        # But also should not create a SELL plan since delta_base would be 0
        plan = plan_trade(
            portfolio=portfolio,
            price=current_price,
            target_exposure=0.0,
            min_notional=10.0,
            return_threshold=return_threshold,
        )
        
        # Should be HOLD (no position to sell)
        assert plan.action == "HOLD"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
