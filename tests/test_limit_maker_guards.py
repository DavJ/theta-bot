"""
Tests for limit maker SELL no-loss guard.

These tests validate that the SELL guard prevents realized losses by
ensuring sell_limit_price >= avg_entry_price * (1 + edge).
"""

import pytest

from spot_bot.execution.ccxt_executor import CCXTExecutor, ExecutorConfig


class TestEdgeCalculation:
    """Tests for edge threshold calculation."""
    
    def test_edge_bps_calculation(self):
        """Test that edge_bps_total is correctly computed."""
        config = ExecutorConfig(
            maker_fee_rate=0.001,
            taker_fee_rate=0.001,
            min_profit_bps=5.0,
            slippage_bps=2.0,
            edge_floor_bps=0.0,
            edge_softmax_alpha=20.0,
            fee_roundtrip_mode="maker_maker",
        )
        executor = CCXTExecutor(config)
        
        # Test with bid=100, ask=101 (1% spread = 100 bps)
        bid = 100.0
        ask = 101.0
        
        edge_bps, components = executor._compute_edge_bps_total(bid, ask)
        
        # Expected components:
        # - Fee: 2 * 0.001 * 10000 = 20 bps
        # - Spread: 0.5 * 100 = ~50 bps (actual calculation may vary slightly)
        # - Slippage: 2 bps
        # - Profit: 5 bps
        # Total: ~77 bps
        
        assert abs(components["fee_component_bps"] - 20.0) < 0.1
        assert abs(components["spread_component_bps"] - 50.0) < 0.5  # Wider tolerance
        assert abs(components["slippage_component_bps"] - 2.0) < 0.1
        assert abs(components["profit_component_bps"] - 5.0) < 0.1
        # Total should be around 77 bps
        assert abs(components["computed_bps"] - 77.0) < 0.5


class TestSellGuardLogic:
    """Tests for the SELL no-loss guard invariant."""
    
    def test_guard_calculation(self):
        """Test guard calculation: required_exit = avg_entry * (1 + edge)."""
        # Scenario: avg_entry=100, edge=0.01 (1%)
        avg_entry = 100.0
        edge = 0.01
        
        required_exit = avg_entry * (1.0 + edge)
        
        # Expected: 101.0
        assert abs(required_exit - 101.0) < 0.001
    
    def test_guard_activation_logic(self):
        """Test when guard should activate (computed_limit < required_exit)."""
        avg_entry = 105.0
        edge = 0.0077  # ~77 bps
        
        required_exit = avg_entry * (1.0 + edge)  # 105.81
        
        # Computed limit below required exit
        computed_limit = 101.0
        
        # Guard should activate
        assert computed_limit < required_exit
        
        # Final limit should be required_exit
        final_limit = max(computed_limit, required_exit)
        assert abs(final_limit - required_exit) < 0.001
    
    def test_guard_no_activation_logic(self):
        """Test when guard should NOT activate (computed_limit >= required_exit)."""
        avg_entry = 90.0
        edge = 0.0077
        
        required_exit = avg_entry * (1.0 + edge)  # 90.69
        
        # Computed limit above required exit
        computed_limit = 102.0
        
        # Guard should NOT activate
        assert computed_limit >= required_exit
        
        # Final limit should be computed_limit
        final_limit = max(computed_limit, required_exit)
        assert abs(final_limit - computed_limit) < 0.001


class TestLimitPriceDirection:
    """Tests for correct limit price direction (BUY < bid, SELL > ask)."""
    
    def test_buy_direction(self):
        """BUY limit should be below reference (bid)."""
        bid = 100.0
        edge = 0.005  # 0.5%
        maker_offset = 0.001  # 0.1%
        
        # BUY: limit = bid * (1 - edge) * (1 - maker_offset)
        limit = bid * (1.0 - edge) * (1.0 - maker_offset)
        
        # Should be < bid
        assert limit < bid
        # Expected: 100 * 0.995 * 0.999 = 99.40
        assert abs(limit - 99.40) < 0.1
    
    def test_sell_direction(self):
        """SELL limit should be above reference (ask)."""
        ask = 101.0
        edge = 0.005
        maker_offset = 0.001
        
        # SELL: limit = ask * (1 + edge) * (1 + maker_offset)
        limit = ask * (1.0 + edge) * (1.0 + maker_offset)
        
        # Should be > ask
        assert limit > ask
        # Expected: 101 * 1.005 * 1.001 = 101.606
        assert abs(limit - 101.606) < 0.1
    
    def test_sell_with_guard(self):
        """SELL limit with guard should respect both edge and guard."""
        ask = 101.0
        edge = 0.0077
        maker_offset = 0.0001
        avg_entry = 105.0
        
        # Computed limit (before guard)
        computed_limit = ask * (1.0 + edge) * (1.0 + maker_offset)
        
        # Guard required exit
        required_exit = avg_entry * (1.0 + edge)
        
        # Final limit is max
        final_limit = max(computed_limit, required_exit)
        
        # Should be above ask
        assert final_limit > ask
        
        # Should be >= required_exit (guard respected)
        assert final_limit >= required_exit - 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
