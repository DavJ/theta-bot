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


class TestEdgeCalculation:
    """Tests for edge/hysteresis calculation logic."""

    def test_smooth_max_basic(self):
        """Test smooth_max returns approximately max(a, b)."""
        config = ExecutorConfig(symbol="BTC/USDT")
        executor = CCXTExecutor(config)
        
        # When a >> b, should return approximately a
        result = executor._smooth_max(10.0, 2.0, 20.0)
        assert result > 9.5  # Should be close to 10
        assert result >= 10.0  # Should be at least max(a, b)
        
        # When b >> a, should return approximately b
        result = executor._smooth_max(2.0, 10.0, 20.0)
        assert result > 9.5  # Should be close to 10
        assert result >= 10.0  # Should be at least max(a, b)
        
    def test_smooth_max_always_exceeds_inputs(self):
        """Test that smooth_max(a, b) >= max(a, b)."""
        config = ExecutorConfig(symbol="BTC/USDT")
        executor = CCXTExecutor(config)
        
        # Test various combinations
        for a, b in [(5.0, 10.0), (10.0, 5.0), (7.5, 7.5), (0.0, 5.0), (5.0, 0.0)]:
            result = executor._smooth_max(a, b, 20.0)
            expected_max = max(a, b)
            assert result >= expected_max - 1e-10, f"smooth_max({a}, {b}) = {result} < max = {expected_max}"
    
    def test_smooth_max_symmetric(self):
        """Test that smooth_max is symmetric: smooth_max(a, b) == smooth_max(b, a)."""
        config = ExecutorConfig(symbol="BTC/USDT")
        executor = CCXTExecutor(config)
        
        result1 = executor._smooth_max(5.0, 10.0, 20.0)
        result2 = executor._smooth_max(10.0, 5.0, 20.0)
        
        assert abs(result1 - result2) < 1e-10

    def test_compute_edge_bps_maker_maker_mode(self):
        """Test edge calculation in maker_maker fee mode."""
        config = ExecutorConfig(
            symbol="BTC/USDT",
            maker_fee_rate=0.0003,  # 3 bps
            taker_fee_rate=0.001,   # 10 bps (not used in maker_maker)
            slippage_bps=2.0,
            min_profit_bps=5.0,
            edge_floor_bps=0.0,
            fee_roundtrip_mode="maker_maker",
        )
        executor = CCXTExecutor(config)
        
        bid = 50000.0
        ask = 50050.0  # Spread = 50 / 50025 * 10000 = 9.995 bps
        
        edge_bps, components = executor._compute_edge_bps_total(bid, ask)
        
        # Fee component: 2 * 3 = 6 bps
        assert abs(components["fee_component_bps"] - 6.0) < 0.1
        
        # Spread component: 0.5 * 9.995 = 4.9975 bps
        assert abs(components["spread_component_bps"] - 5.0) < 0.1
        
        # Slippage: 2.0 bps
        assert components["slippage_component_bps"] == 2.0
        
        # Profit: 5.0 bps
        assert components["profit_component_bps"] == 5.0
        
        # Total computed: 6 + 5 + 2 + 5 = 18 bps
        assert abs(components["computed_bps"] - 18.0) < 0.1
        
        # Edge should be at least computed_bps
        assert edge_bps >= 17.5

    def test_compute_edge_bps_maker_taker_mode(self):
        """Test edge calculation in maker_taker fee mode."""
        config = ExecutorConfig(
            symbol="BTC/USDT",
            maker_fee_rate=0.0003,  # 3 bps
            taker_fee_rate=0.001,   # 10 bps
            slippage_bps=0.0,
            min_profit_bps=5.0,
            edge_floor_bps=0.0,
            fee_roundtrip_mode="maker_taker",
        )
        executor = CCXTExecutor(config)
        
        bid = 50000.0
        ask = 50050.0
        
        edge_bps, components = executor._compute_edge_bps_total(bid, ask)
        
        # Fee component: 3 + 10 = 13 bps
        assert abs(components["fee_component_bps"] - 13.0) < 0.1

    def test_compute_edge_bps_respects_floor(self):
        """Test that edge calculation respects the floor threshold."""
        config = ExecutorConfig(
            symbol="BTC/USDT",
            maker_fee_rate=0.0001,  # 1 bps
            taker_fee_rate=0.0001,
            slippage_bps=0.0,
            min_profit_bps=1.0,
            edge_floor_bps=20.0,  # High floor
            fee_roundtrip_mode="maker_maker",
        )
        executor = CCXTExecutor(config)
        
        bid = 50000.0
        ask = 50010.0  # Narrow spread
        
        edge_bps, components = executor._compute_edge_bps_total(bid, ask)
        
        # Computed should be low (2 + 1 + 0 + 1 = 4 bps)
        # But edge should be close to floor (20 bps)
        assert components["computed_bps"] < 10.0
        assert edge_bps >= 19.0  # Should be close to floor


class TestLimitPriceDirection:
    """Tests for limit price direction (BUY < bid, SELL > ask)."""

    def test_buy_limit_below_bid(self):
        """Test that BUY limit price is below bid."""
        config = ExecutorConfig(
            symbol="BTC/USDT",
            order_type="limit_maker",
            maker_offset_bps=1.0,
            maker_fee_rate=0.0003,
            min_profit_bps=5.0,
            slippage_bps=2.0,
            max_notional_per_trade=10000.0,
            max_turnover_per_day=20000.0,
            api_key="test_key",
            api_secret="test_secret",
        )
        executor = CCXTExecutor(config)
        
        # Mock exchange
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {
            "bid": 50000.0,
            "ask": 50050.0,
        }
        
        # Mock order creation to capture the price
        captured_price = None
        def capture_order(*args, **kwargs):
            nonlocal captured_price
            captured_price = kwargs.get("price")
            return {
                "id": "test_order",
                "status": "open",
                "filled": 0,
            }
        
        mock_exchange.create_order = capture_order
        executor.exchange = mock_exchange
        executor._current_day = None
        executor._market_info = {
            "precision": {"price": 2, "amount": 8},
            "limits": {},
        }
        
        result = executor.place_limit_maker_order("buy", 0.1, 50025.0)
        
        # Verify limit price is below bid
        assert captured_price is not None
        assert captured_price < 50000.0, f"BUY limit price {captured_price} should be < bid 50000.0"

    def test_sell_limit_above_ask(self):
        """Test that SELL limit price is above ask."""
        config = ExecutorConfig(
            symbol="BTC/USDT",
            order_type="limit_maker",
            maker_offset_bps=1.0,
            maker_fee_rate=0.0003,
            min_profit_bps=5.0,
            slippage_bps=2.0,
            max_notional_per_trade=10000.0,
            max_turnover_per_day=20000.0,
            api_key="test_key",
            api_secret="test_secret",
        )
        executor = CCXTExecutor(config)
        
        # Mock exchange
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {
            "bid": 50000.0,
            "ask": 50050.0,
        }
        
        # Mock order creation to capture the price
        captured_price = None
        def capture_order(*args, **kwargs):
            nonlocal captured_price
            captured_price = kwargs.get("price")
            return {
                "id": "test_order",
                "status": "open",
                "filled": 0,
            }
        
        mock_exchange.create_order = capture_order
        executor.exchange = mock_exchange
        executor._current_day = None
        executor._market_info = {
            "precision": {"price": 2, "amount": 8},
            "limits": {},
        }
        
        result = executor.place_limit_maker_order("sell", 0.1, 50025.0)
        
        # Verify limit price is above ask
        assert captured_price is not None
        assert captured_price > 50050.0, f"SELL limit price {captured_price} should be > ask 50050.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
