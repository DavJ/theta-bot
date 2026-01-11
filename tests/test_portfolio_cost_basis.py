"""
Tests for portfolio cost basis tracking (avg_entry_price).

These tests validate that avg_entry_price is correctly tracked across
BUY and SELL fills according to the no-loss invariant requirements.
"""

import pytest

from spot_bot.core.portfolio import apply_fill
from spot_bot.core.types import ExecutionResult, PortfolioState


class TestAvgEntryPriceTracking:
    """Tests for avg_entry_price updates in apply_fill."""

    def test_initial_buy_sets_avg_entry(self):
        """First BUY should set avg_entry_price to buy price."""
        portfolio = PortfolioState(
            usdt=1000.0,
            base=0.0,
            equity=1000.0,
            exposure=0.0,
            avg_entry_price=None,
            realized_pnl_quote=0.0,
        )
        
        execution = ExecutionResult(
            filled_base=0.1,  # Buy 0.1 BTC
            avg_price=50000.0,
            fee_paid=5.0,
            slippage_paid=0.0,
            status="filled",
            raw=None,
        )
        
        updated = apply_fill(portfolio, execution)
        
        assert updated.avg_entry_price == 50000.0
        assert updated.base == 0.1
        assert updated.avg_entry_price is not None

    def test_buy_then_buy_weighted_average(self):
        """Multiple BUYs should compute weighted average entry price."""
        portfolio = PortfolioState(
            usdt=900.0,
            base=0.1,
            equity=5900.0,
            exposure=0.847,
            avg_entry_price=50000.0,
            realized_pnl_quote=0.0,
        )
        
        # Buy another 0.2 BTC at 60000
        execution = ExecutionResult(
            filled_base=0.2,
            avg_price=60000.0,
            fee_paid=12.0,
            slippage_paid=0.0,
            status="filled",
            raw=None,
        )
        
        updated = apply_fill(portfolio, execution)
        
        # Expected: (0.1 * 50000 + 0.2 * 60000) / (0.1 + 0.2)
        # = (5000 + 12000) / 0.3 = 17000 / 0.3 = 56666.67
        expected_avg = (0.1 * 50000.0 + 0.2 * 60000.0) / 0.3
        assert abs(updated.avg_entry_price - expected_avg) < 0.01
        assert abs(updated.base - 0.3) < 1e-10  # Floating point tolerance

    def test_buy_then_sell_partial_preserves_avg(self):
        """Partial SELL should preserve avg_entry_price."""
        portfolio = PortfolioState(
            usdt=500.0,
            base=0.1,
            equity=5500.0,
            exposure=0.909,
            avg_entry_price=50000.0,
            realized_pnl_quote=0.0,
        )
        
        # Sell 0.05 BTC at 55000 (profitable exit)
        execution = ExecutionResult(
            filled_base=-0.05,
            avg_price=55000.0,
            fee_paid=2.75,
            slippage_paid=0.0,
            status="filled",
            raw=None,
        )
        
        updated = apply_fill(portfolio, execution)
        
        # avg_entry_price should remain unchanged
        assert updated.avg_entry_price == 50000.0
        assert updated.base == 0.05
        
        # Should have computed realized PnL
        # PnL = (55000 - 50000) * 0.05 - 2.75 = 250 - 2.75 = 247.25
        expected_pnl = (55000.0 - 50000.0) * 0.05 - 2.75
        assert abs(updated.realized_pnl_quote - expected_pnl) < 0.01

    def test_sell_to_zero_resets_avg_entry(self):
        """Selling entire position should reset avg_entry_price to None."""
        portfolio = PortfolioState(
            usdt=500.0,
            base=0.1,
            equity=5500.0,
            exposure=0.909,
            avg_entry_price=50000.0,
            realized_pnl_quote=0.0,
        )
        
        # Sell all 0.1 BTC
        execution = ExecutionResult(
            filled_base=-0.1,
            avg_price=55000.0,
            fee_paid=5.5,
            slippage_paid=0.0,
            status="filled",
            raw=None,
        )
        
        updated = apply_fill(portfolio, execution)
        
        # avg_entry_price should be reset
        assert updated.avg_entry_price is None
        assert updated.base == 0.0
        
        # PnL = (55000 - 50000) * 0.1 - 5.5 = 500 - 5.5 = 494.5
        expected_pnl = (55000.0 - 50000.0) * 0.1 - 5.5
        assert abs(updated.realized_pnl_quote - expected_pnl) < 0.01

    def test_sell_more_than_position_clamps_to_zero(self):
        """Over-selling should clamp position to zero and reset avg_entry."""
        portfolio = PortfolioState(
            usdt=500.0,
            base=0.1,
            equity=5500.0,
            exposure=0.909,
            avg_entry_price=50000.0,
            realized_pnl_quote=0.0,
        )
        
        # Try to sell 0.15 BTC (more than we have)
        # In practice this shouldn't happen due to trade planner,
        # but we should handle it gracefully
        execution = ExecutionResult(
            filled_base=-0.15,
            avg_price=55000.0,
            fee_paid=8.25,
            slippage_paid=0.0,
            status="filled",
            raw=None,
        )
        
        updated = apply_fill(portfolio, execution)
        
        # Position should be clamped to 0
        assert updated.base <= 0.0
        assert updated.avg_entry_price is None

    def test_skipped_execution_preserves_state(self):
        """SKIPPED execution should not change portfolio."""
        portfolio = PortfolioState(
            usdt=500.0,
            base=0.1,
            equity=5500.0,
            exposure=0.909,
            avg_entry_price=50000.0,
            realized_pnl_quote=100.0,
        )
        
        execution = ExecutionResult(
            filled_base=0.0,
            avg_price=55000.0,
            fee_paid=0.0,
            slippage_paid=0.0,
            status="SKIPPED",
            raw=None,
        )
        
        updated = apply_fill(portfolio, execution)
        
        # Everything should remain unchanged
        assert updated.usdt == portfolio.usdt
        assert updated.base == portfolio.base
        assert updated.avg_entry_price == portfolio.avg_entry_price
        assert updated.realized_pnl_quote == portfolio.realized_pnl_quote

    def test_multiple_round_trips(self):
        """Test multiple complete round trips to verify PnL tracking."""
        # Start fresh
        portfolio = PortfolioState(
            usdt=10000.0,
            base=0.0,
            equity=10000.0,
            exposure=0.0,
            avg_entry_price=None,
            realized_pnl_quote=0.0,
        )
        
        # First round trip: BUY at 50k, SELL at 55k
        buy1 = ExecutionResult(
            filled_base=0.1,
            avg_price=50000.0,
            fee_paid=5.0,
            slippage_paid=0.0,
            status="filled",
            raw=None,
        )
        portfolio = apply_fill(portfolio, buy1)
        assert portfolio.avg_entry_price == 50000.0
        
        sell1 = ExecutionResult(
            filled_base=-0.1,
            avg_price=55000.0,
            fee_paid=5.5,
            slippage_paid=0.0,
            status="filled",
            raw=None,
        )
        portfolio = apply_fill(portfolio, sell1)
        assert portfolio.avg_entry_price is None
        
        # PnL from round trip 1: (55000 - 50000) * 0.1 - 5.5 = 500 - 5.5 = 494.5
        expected_pnl_1 = (55000.0 - 50000.0) * 0.1 - 5.5
        assert abs(portfolio.realized_pnl_quote - expected_pnl_1) < 0.01
        
        # Second round trip: BUY at 52k, SELL at 51k (loss)
        buy2 = ExecutionResult(
            filled_base=0.2,
            avg_price=52000.0,
            fee_paid=10.4,
            slippage_paid=0.0,
            status="filled",
            raw=None,
        )
        portfolio = apply_fill(portfolio, buy2)
        assert portfolio.avg_entry_price == 52000.0
        
        sell2 = ExecutionResult(
            filled_base=-0.2,
            avg_price=51000.0,
            fee_paid=10.2,
            slippage_paid=0.0,
            status="filled",
            raw=None,
        )
        portfolio = apply_fill(portfolio, sell2)
        assert portfolio.avg_entry_price is None
        
        # PnL from round trip 2: (51000 - 52000) * 0.2 - 10.2 = -200 - 10.2 = -210.2
        expected_pnl_2 = (51000.0 - 52000.0) * 0.2 - 10.2
        total_expected_pnl = expected_pnl_1 + expected_pnl_2
        assert abs(portfolio.realized_pnl_quote - total_expected_pnl) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
