from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict


class ExecutionResult(TypedDict, total=False):
    status: str
    order_id: str
    filled_qty: float
    avg_price: float
    fee_est: float
    reason: str


@dataclass
class ExecutorConfig:
    exchange_id: str = "binance"
    symbol: str = "BTC/USDT"
    max_notional_per_trade: float = 300.0
    max_trades_per_day: int = 10
    max_turnover_per_day: float = 2000.0
    slippage_bps_limit: float = 10.0
    min_balance_reserve_usdt: float = 50.0
    fee_rate: float = 0.001
    min_notional: float = 10.0
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    # Limit maker execution settings
    order_type: str = "market"  # "market" or "limit_maker"
    maker_offset_bps: float = 1.0  # how far from best bid/ask to place limit
    order_validity_seconds: int = 60  # cancel stale maker orders
    max_spread_bps: float = 20.0  # if spread too wide, reject placing maker order
    maker_fee_rate: float = 0.001  # default maker fee rate
    taker_fee_rate: float = 0.001  # default taker fee rate


class CCXTExecutor:
    """Spot-only execution helper with conservative safety limits."""

    def __init__(self, config: ExecutorConfig) -> None:
        self.config = config
        self.exchange = None
        self.trades_today = 0
        self.turnover_today = 0.0
        self._current_day = None
        self._market_info = None

    def _ensure_exchange(self) -> None:
        if self.exchange:
            return
        try:
            import ccxt  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("ccxt is required for live trading") from exc
        if not hasattr(ccxt, self.config.exchange_id):
            raise ValueError(f"Exchange '{self.config.exchange_id}' not found in ccxt.")
        cls = getattr(ccxt, self.config.exchange_id)
        self.exchange = cls({"enableRateLimit": True})
        if self.config.api_key and self.config.api_secret:
            self.exchange.apiKey = self.config.api_key
            self.exchange.secret = self.config.api_secret

    def _reset_counters(self) -> None:
        today = datetime.date.today()
        if self._current_day != today:
            self._current_day = today
            self.trades_today = 0
            self.turnover_today = 0.0

    def _slippage_guard(self, last_close: float) -> bool:
        try:
            ticker = self.exchange.fetch_ticker(self.config.symbol)
            bid = ticker.get("bid")
            ask = ticker.get("ask")
            if bid is None or ask is None:
                return True
            mid = (bid + ask) / 2.0
            deviation_bps = abs(mid - last_close) / max(last_close, 1e-12) * 10000.0
            return deviation_bps <= self.config.slippage_bps_limit
        except Exception:
            return True

    def _has_reserve(self) -> bool:
        if not self.config.min_balance_reserve_usdt:
            return True
        try:
            balances = self.exchange.fetch_balance()
            free_usdt = None
            free_block = balances.get("free")
            if isinstance(free_block, dict):
                free_usdt = free_block.get("USDT")
            if free_usdt is None:
                spot_entry = balances.get("USDT")
                if isinstance(spot_entry, dict):
                    free_usdt = spot_entry.get("free")
            return free_usdt is None or float(free_usdt) >= self.config.min_balance_reserve_usdt
        except Exception:
            return True

    def fetch_market_rules(self) -> None:
        """Fetch and cache market rules (precision, min notional, etc)."""
        if self._market_info is not None:
            return
        try:
            self.exchange.load_markets()
            market = self.exchange.market(self.config.symbol)
            self._market_info = {
                "precision": market.get("precision", {}),
                "limits": market.get("limits", {}),
            }
        except Exception:
            # Fallback to defaults if market info unavailable
            self._market_info = {
                "precision": {"price": 2, "amount": 8},
                "limits": {"cost": {"min": self.config.min_notional}},
            }

    def quantize_price(self, price: float) -> float:
        """Round price to exchange precision."""
        self.fetch_market_rules()
        precision = self._market_info.get("precision", {})
        price_precision = precision.get("price")
        
        if price_precision is None:
            return price
        
        if isinstance(price_precision, int):
            # Decimal places
            return round(price, price_precision)
        else:
            # Tick size
            tick_size = float(price_precision)
            if tick_size > 0:
                return round(price / tick_size) * tick_size
        return price

    def quantize_amount(self, amount: float) -> float:
        """Round amount to exchange precision."""
        self.fetch_market_rules()
        precision = self._market_info.get("precision", {})
        amount_precision = precision.get("amount")
        
        if amount_precision is None:
            return amount
        
        if isinstance(amount_precision, int):
            # Decimal places
            return round(amount, amount_precision)
        else:
            # Step size
            step_size = float(amount_precision)
            if step_size > 0:
                return round(amount / step_size) * step_size
        return amount

    def cancel_stale_orders(self) -> None:
        """Cancel open orders older than order_validity_seconds."""
        try:
            open_orders = self.exchange.fetch_open_orders(self.config.symbol)
            now = datetime.datetime.now(datetime.timezone.utc)
            
            for order in open_orders:
                order_time = order.get("timestamp")
                if order_time:
                    order_dt = datetime.datetime.fromtimestamp(order_time / 1000.0, datetime.timezone.utc)
                    age_seconds = (now - order_dt).total_seconds()
                    
                    if age_seconds > self.config.order_validity_seconds:
                        try:
                            self.exchange.cancel_order(order["id"], self.config.symbol)
                        except Exception:
                            # Fail safe - if cancel fails, continue
                            pass
        except Exception:
            # Fail safe - if fetching orders fails, do nothing
            pass

    def place_limit_maker_order(self, side: str, qty: float, last_close: float) -> ExecutionResult:
        """
        Place a limit maker (post-only) order.
        
        Args:
            side: "buy" or "sell"
            qty: Quantity to trade (positive)
            last_close: Last close price for reference
            
        Returns:
            ExecutionResult with status "open", "filled", "rejected", or "error"
        """
        self._ensure_exchange()
        self._reset_counters()
        
        notional = abs(qty) * last_close

        # Basic checks
        if qty <= 0:
            return {"status": "rejected", "reason": "qty_non_positive"}
        if notional < self.config.min_notional:
            return {"status": "rejected", "reason": "min_notional"}
        if notional > self.config.max_notional_per_trade:
            return {"status": "rejected", "reason": "max_notional_per_trade"}
        if self.trades_today >= self.config.max_trades_per_day:
            return {"status": "rejected", "reason": "max_trades_per_day"}
        if (self.turnover_today + notional) > self.config.max_turnover_per_day:
            return {"status": "rejected", "reason": "max_turnover_per_day"}
        if not self._has_reserve():
            return {"status": "rejected", "reason": "reserve_guard"}

        try:
            # Fetch orderbook to get bid/ask
            ticker = self.exchange.fetch_ticker(self.config.symbol)
            bid = ticker.get("bid")
            ask = ticker.get("ask")
            
            if bid is None or ask is None:
                return {"status": "rejected", "reason": "no_bid_ask"}
            
            # Compute spread and check spread guard
            mid = (bid + ask) / 2.0
            spread_bps = (ask - bid) / max(mid, 1e-12) * 10000.0
            
            if spread_bps > self.config.max_spread_bps:
                return {"status": "rejected", "reason": "spread_guard"}
            
            # Compute limit price
            if side == "buy":
                # Place bid below current bid
                limit_price = bid * (1.0 - self.config.maker_offset_bps * 1e-4)
            else:  # sell
                # Place ask above current ask
                limit_price = ask * (1.0 + self.config.maker_offset_bps * 1e-4)
            
            # Quantize price
            limit_price = self.quantize_price(limit_price)
            
            # Quantize amount
            qty_quantized = self.quantize_amount(qty)
            
            # Check if quantization resulted in zero or too small notional
            if qty_quantized <= 0:
                return {"status": "rejected", "reason": "qty_too_small"}
            
            notional_quantized = qty_quantized * limit_price
            if notional_quantized < self.config.min_notional:
                return {"status": "rejected", "reason": "min_notional_after_quantize"}
            
            # Create limit order with postOnly
            # Binance supports both postOnly and timeInForce:GTX
            # Try postOnly first, fallback to GTX if needed
            params = {"postOnly": True}
            
            try:
                order = self.exchange.create_order(
                    symbol=self.config.symbol,
                    type="limit",
                    side=side,
                    amount=qty_quantized,
                    price=limit_price,
                    params=params
                )
            except Exception as e:
                # Try with GTX as fallback for Binance
                if "binance" in self.config.exchange_id.lower():
                    params = {"timeInForce": "GTX"}
                    order = self.exchange.create_order(
                        symbol=self.config.symbol,
                        type="limit",
                        side=side,
                        amount=qty_quantized,
                        price=limit_price,
                        params=params
                    )
                else:
                    raise
            
            # Update counters
            self.trades_today += 1
            self.turnover_today += notional_quantized
            
            # Check order status
            order_status = order.get("status", "open")
            order_id = order.get("id", "")
            filled = float(order.get("filled", 0))
            
            if order_status == "closed" or filled > 0:
                # Order was filled immediately (shouldn't happen with post-only, but handle it)
                avg_price = float(order.get("average") or limit_price)
                fee_est = notional_quantized * self.config.maker_fee_rate
                return {
                    "status": "filled",
                    "order_id": order_id,
                    "filled_qty": filled,
                    "avg_price": avg_price,
                    "fee_est": fee_est,
                }
            else:
                # Order is open, waiting to be filled
                fee_est = notional_quantized * self.config.maker_fee_rate
                return {
                    "status": "open",
                    "order_id": order_id,
                    "filled_qty": 0.0,
                    "avg_price": limit_price,
                    "fee_est": fee_est,
                    "limit_price": limit_price,
                }
            
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

    def place_market_order(self, side: str, qty: float, last_close: float) -> ExecutionResult:
        self._ensure_exchange()
        self._reset_counters()
        notional = abs(qty) * last_close

        if qty <= 0:
            return {"status": "rejected", "reason": "qty_non_positive"}
        if notional < self.config.min_notional:
            return {"status": "rejected", "reason": "min_notional"}
        if notional > self.config.max_notional_per_trade:
            return {"status": "rejected", "reason": "max_notional_per_trade"}
        if self.trades_today >= self.config.max_trades_per_day:
            return {"status": "rejected", "reason": "max_trades_per_day"}
        if (self.turnover_today + notional) > self.config.max_turnover_per_day:
            return {"status": "rejected", "reason": "max_turnover_per_day"}
        if not self._slippage_guard(last_close):
            return {"status": "rejected", "reason": "slippage_guard"}
        if not self._has_reserve():
            return {"status": "rejected", "reason": "reserve_guard"}

        try:
            order = self.exchange.create_order(
                symbol=self.config.symbol, type="market", side=side, amount=qty
            )
            self.trades_today += 1
            self.turnover_today += notional
            fee_est = notional * self.config.taker_fee_rate
            order_id = order.get("id", "")
            filled = float(order.get("filled") or qty)
            avg_price = float(order.get("average") or last_close)
            return {
                "status": "filled",
                "order_id": order_id,
                "filled_qty": filled,
                "avg_price": avg_price,
                "fee_est": fee_est,
            }
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}


__all__ = ["CCXTExecutor", "ExecutorConfig", "ExecutionResult"]
