"""
Tests for limit maker execution functionality.

These tests validate the limit maker order execution logic including
quantization, spread guards, and OPEN order status handling.
"""

import pytest
from unittest.mock import Mock, MagicMock

from spot_bot.execution.ccxt_executor import CCXTExecutor, ExecutorConfig, CCXTExecutionResult
from spot_bot.core.executor import LiveExecutor
from spot_bot.core.types import TradePlan


class TestQuantization:
    """Tests for price and amount quantization."""

    def test_quantize_price_decimal_precision(self):
        """Test price quantization with decimal precision."""
        config = ExecutorConfig(symbol="BTC/USDT")
        executor = CCXTExecutor(config)
        
        # Mock market info with decimal precision
        executor._market_info = {
            "precision": {"price": 2, "amount": 8},
            "limits": {},
        }
        
        # Test rounding
        assert executor.quantize_price(50000.123) == 50000.12
        assert executor.quantize_price(50000.126) == 50000.13
        assert executor.quantize_price(50000.125) == 50000.12  # Python rounds to even
        
    def test_quantize_price_tick_size(self):
        """Test price quantization with tick size."""
        config = ExecutorConfig(symbol="BTC/USDT")
        executor = CCXTExecutor(config)
        
        # Mock market info with tick size
        executor._market_info = {
            "precision": {"price": 0.01, "amount": 8},
            "limits": {},
        }
        
        # Test rounding to tick size (with floating point tolerance)
        assert abs(executor.quantize_price(50000.123) - 50000.12) < 1e-6
        assert abs(executor.quantize_price(50000.126) - 50000.13) < 1e-6
        
    def test_quantize_price_non_negative(self):
        """Test that quantize_price never returns negative values."""
        config = ExecutorConfig(symbol="BTC/USDT")
        executor = CCXTExecutor(config)
        
        executor._market_info = {
            "precision": {"price": 2, "amount": 8},
            "limits": {},
        }
        
        # Positive prices should remain positive
        assert executor.quantize_price(0.001) >= 0
        assert executor.quantize_price(100.0) >= 0
        assert executor.quantize_price(1e-10) >= 0

    def test_quantize_amount_decimal_precision(self):
        """Test amount quantization with decimal precision."""
        config = ExecutorConfig(symbol="BTC/USDT")
        executor = CCXTExecutor(config)
        
        executor._market_info = {
            "precision": {"price": 2, "amount": 5},
            "limits": {},
        }
        
        assert executor.quantize_amount(0.123456) == 0.12346
        assert executor.quantize_amount(0.123454) == 0.12345

    def test_quantize_amount_step_size(self):
        """Test amount quantization with step size."""
        config = ExecutorConfig(symbol="BTC/USDT")
        executor = CCXTExecutor(config)
        
        executor._market_info = {
            "precision": {"price": 2, "amount": 0.001},
            "limits": {},
        }
        
        assert executor.quantize_amount(0.1234) == 0.123
        assert executor.quantize_amount(0.1236) == 0.124

    def test_quantize_amount_non_negative(self):
        """Test that quantize_amount never returns negative values."""
        config = ExecutorConfig(symbol="BTC/USDT")
        executor = CCXTExecutor(config)
        
        executor._market_info = {
            "precision": {"price": 2, "amount": 8},
            "limits": {},
        }
        
        # Positive amounts should remain positive or zero
        assert executor.quantize_amount(0.00000001) >= 0
        assert executor.quantize_amount(1.0) >= 0


class TestSpreadGuard:
    """Tests for spread guard functionality."""

    def test_spread_guard_triggers_wide_spread(self):
        """Test that spread guard rejects orders when spread is too wide."""
        config = ExecutorConfig(
            symbol="BTC/USDT",
            max_spread_bps=20.0,
            max_notional_per_trade=10000.0,  # Set higher to pass notional check
            max_turnover_per_day=20000.0,  # Set higher to pass turnover check
            api_key="test_key",
            api_secret="test_secret",
        )
        executor = CCXTExecutor(config)
        
        # Mock exchange
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {
            "bid": 50000.0,
            "ask": 50200.0,  # Spread: 200 / 50100 * 10000 = 39.92 bps > 20 bps
        }
        mock_exchange.fetch_balance.return_value = {
            "free": {"USDT": 10000.0}
        }
        executor.exchange = mock_exchange
        executor._current_day = None
        
        # Mock market info
        executor._market_info = {
            "precision": {"price": 2, "amount": 8},
            "limits": {},
        }
        
        result = executor.place_limit_maker_order("buy", 0.1, 50100.0)
        
        assert result["status"] == "rejected"
        assert result["reason"] == "spread_guard"

    def test_spread_guard_accepts_narrow_spread(self):
        """Test that spread guard accepts orders when spread is acceptable."""
        config = ExecutorConfig(
            symbol="BTC/USDT",
            max_spread_bps=20.0,
            min_notional=10.0,
            api_key="test_key",
            api_secret="test_secret",
        )
        executor = CCXTExecutor(config)
        
        # Mock exchange
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {
            "bid": 50000.0,
            "ask": 50050.0,  # Spread: 50 / 50025 * 10000 = 9.99 bps < 20 bps
        }
        mock_exchange.create_order.return_value = {
            "id": "test_order_123",
            "status": "open",
            "filled": 0,
        }
        executor.exchange = mock_exchange
        executor._current_day = None
        
        # Mock market info
        executor._market_info = {
            "precision": {"price": 2, "amount": 8},
            "limits": {},
        }
        
        result = executor.place_limit_maker_order("buy", 0.1, 50025.0)
        
        # Should not be rejected by spread guard
        assert result["status"] != "rejected" or result["reason"] != "spread_guard"

    def test_spread_guard_no_bid_ask(self):
        """Test that spread guard rejects when bid/ask is unavailable."""
        config = ExecutorConfig(
            symbol="BTC/USDT",
            max_spread_bps=20.0,
            max_notional_per_trade=10000.0,  # Set higher to pass notional check
            max_turnover_per_day=20000.0,  # Set higher to pass turnover check
            api_key="test_key",
            api_secret="test_secret",
        )
        executor = CCXTExecutor(config)
        
        # Mock exchange with missing bid/ask
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {
            "bid": None,
            "ask": 50000.0,
        }
        mock_exchange.fetch_balance.return_value = {
            "free": {"USDT": 10000.0}
        }
        executor.exchange = mock_exchange
        executor._current_day = None
        
        result = executor.place_limit_maker_order("buy", 0.1, 50000.0)
        
        assert result["status"] == "rejected"
        assert result["reason"] == "no_bid_ask"


class TestLiveExecutorOpenStatus:
    """Tests for LiveExecutor OPEN status handling."""

    def test_live_executor_returns_open_for_limit_maker(self):
        """Test that LiveExecutor returns OPEN status for unfilled limit maker orders."""
        config = ExecutorConfig(
            symbol="BTC/USDT",
            order_type="limit_maker",
            api_key="test_key",
            api_secret="test_secret",
        )
        ccxt_executor = CCXTExecutor(config)
        
        # Mock the place_limit_maker_order to return open status
        ccxt_executor.place_limit_maker_order = Mock(return_value={
            "status": "open",
            "order_id": "test_order_123",
            "filled_qty": 0.0,
            "avg_price": 50000.0,
            "fee_est": 0.05,
            "limit_price": 50000.0,
        })
        ccxt_executor.cancel_stale_orders = Mock()
        
        live_executor = LiveExecutor(ccxt_executor)
        
        # Create a simple trade plan
        plan = TradePlan(
            action="BUY",
            target_exposure=0.3,
            target_base=0.1,
            delta_base=0.1,
            notional=5000.0,
            exec_price_hint=50000.0,
            reason="test",
        )
        
        result = live_executor.execute(plan, 50000.0)
        
        assert result.status == "OPEN"
        assert result.filled_base == 0.0
        assert result.fee_paid == 0.0
        assert result.slippage_paid == 0.0
        assert result.raw is not None

    def test_live_executor_calls_cancel_stale_orders(self):
        """Test that LiveExecutor calls cancel_stale_orders before placing limit maker."""
        config = ExecutorConfig(
            symbol="BTC/USDT",
            order_type="limit_maker",
            api_key="test_key",
            api_secret="test_secret",
        )
        ccxt_executor = CCXTExecutor(config)
        
        ccxt_executor.place_limit_maker_order = Mock(return_value={
            "status": "open",
            "order_id": "test_order_123",
            "filled_qty": 0.0,
            "avg_price": 50000.0,
            "fee_est": 0.05,
            "limit_price": 50000.0,
        })
        ccxt_executor.cancel_stale_orders = Mock()
        
        live_executor = LiveExecutor(ccxt_executor)
        
        plan = TradePlan(
            action="BUY",
            target_exposure=0.3,
            target_base=0.1,
            delta_base=0.1,
            notional=5000.0,
            exec_price_hint=50000.0,
            reason="test",
        )
        
        live_executor.execute(plan, 50000.0)
        
        # Verify cancel_stale_orders was called
        ccxt_executor.cancel_stale_orders.assert_called_once()

    def test_live_executor_market_order_unchanged(self):
        """Test that market orders still work as before."""
        config = ExecutorConfig(
            symbol="BTC/USDT",
            order_type="market",  # Market order type
            api_key="test_key",
            api_secret="test_secret",
        )
        ccxt_executor = CCXTExecutor(config)
        
        ccxt_executor.place_market_order = Mock(return_value={
            "status": "filled",
            "order_id": "test_order_123",
            "filled_qty": 0.1,
            "avg_price": 50005.0,
            "fee_est": 5.0,
        })
        
        live_executor = LiveExecutor(ccxt_executor)
        
        plan = TradePlan(
            action="BUY",
            target_exposure=0.3,
            target_base=0.1,
            delta_base=0.1,
            notional=5000.0,
            exec_price_hint=50000.0,
            reason="test",
        )
        
        result = live_executor.execute(plan, 50000.0)
        
        assert result.status == "filled"
        assert result.filled_base == 0.1
        assert result.fee_paid == 5.0
        # Verify market order was called, not limit maker
        ccxt_executor.place_market_order.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
