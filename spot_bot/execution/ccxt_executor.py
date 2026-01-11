from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict


# Numerical stability constant for division guards
_EPSILON = 1e-12


class CCXTExecutionResult(TypedDict, total=False):
    """
    CCXT-layer execution result (dict format for exchange API compatibility).
    
    This is distinct from core.types.ExecutionResult (dataclass) which is used
    by the core engine. LiveExecutor converts between these types.
    """
    status: str
    order_id: str
    filled_qty: float
    avg_price: float
    fee_est: float
    reason: str
    limit_price: float  # Only for limit orders


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
    # Edge/hysteresis calculation settings
    slippage_bps: float = 0.0  # expected slippage in bps
    min_profit_bps: float = 5.0  # minimum profit target in bps
    edge_softmax_alpha: float = 20.0  # smoothness for soft max
    edge_floor_bps: float = 0.0  # minimum edge threshold
    fee_roundtrip_mode: str = "maker_maker"  # "maker_maker" or "maker_taker"


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
            # Fallback to conservative defaults if market info unavailable
            # These defaults work for most major pairs (BTC/USDT, ETH/USDT, etc.)
            # but may need adjustment for exotic pairs
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
        """
        Cancel open orders older than order_validity_seconds.
        
        Uses fail-safe approach: if API errors occur, does nothing rather than failing.
        This prevents order cancellation issues from blocking new order placement.
        """
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
                            # Fail safe - individual order cancellation errors are non-fatal
                            # Order may have already been filled or cancelled
                            pass
        except Exception:
            # Fail safe - if fetching orders fails entirely, do nothing
            # This could happen due to network issues or exchange API errors
            pass

    def _smooth_max(self, a: float, b: float, alpha: float) -> float:
        """
        Smooth maximum function using tanh for differentiability.
        
        Returns approximately max(a, b) but smoothly transitions.
        Formula: 0.5*(a+b) + 0.5*(a-b)*tanh(alpha*(a-b))
        
        Args:
            a: First value
            b: Second value
            alpha: Smoothness parameter (higher = sharper transition)
            
        Returns:
            Smooth maximum of a and b
        """
        diff = a - b
        # Use tanh for smooth transition
        return 0.5 * (a + b) + 0.5 * diff * math.tanh(alpha * diff)

    def _compute_edge_bps_total(
        self,
        bid: float,
        ask: float,
    ) -> tuple[float, Dict[str, float]]:
        """
        Compute total edge threshold in basis points for limit order pricing.
        
        The edge includes:
        - Fee component (roundtrip fees)
        - Spread component (half-spread for maker model)
        - Slippage component
        - Minimum profit requirement
        - Floor threshold (via smooth max)
        
        Args:
            bid: Best bid price
            ask: Best ask price
            
        Returns:
            Tuple of (edge_bps_total, components_dict)
            components_dict contains breakdown of all components for logging
        """
        # Compute spread in bps
        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / max(mid, _EPSILON) * 10000.0
        
        # Fee component
        fee_bps_maker = self.config.maker_fee_rate * 10000.0
        fee_bps_taker = self.config.taker_fee_rate * 10000.0
        
        if self.config.fee_roundtrip_mode == "maker_maker":
            fee_component_bps = 2.0 * fee_bps_maker
        else:  # maker_taker
            fee_component_bps = fee_bps_maker + fee_bps_taker
        
        # Spread component (maker model: we capture half the spread)
        spread_component_bps = 0.5 * spread_bps
        
        # Slippage component
        slippage_component_bps = self.config.slippage_bps
        
        # Profit component
        profit_component_bps = self.config.min_profit_bps
        
        # Compute total (sum of all components)
        computed_bps = (
            fee_component_bps
            + spread_component_bps
            + slippage_component_bps
            + profit_component_bps
        )
        
        # Apply smooth max with floor
        edge_bps_total = self._smooth_max(
            self.config.edge_floor_bps,
            computed_bps,
            self.config.edge_softmax_alpha
        )
        
        # Return total and components for logging
        components = {
            "spread_bps": spread_bps,
            "fee_component_bps": fee_component_bps,
            "spread_component_bps": spread_component_bps,
            "slippage_component_bps": slippage_component_bps,
            "profit_component_bps": profit_component_bps,
            "computed_bps": computed_bps,
            "edge_floor_bps": self.config.edge_floor_bps,
            "edge_bps_total": edge_bps_total,
        }
        
        return edge_bps_total, components

    def place_limit_maker_order(
        self, 
        side: str, 
        qty: float, 
        last_close: float,
        portfolio_avg_entry_price: Optional[float] = None,
    ) -> CCXTExecutionResult:
        """
        Place a limit maker (post-only) order.
        
        Args:
            side: "buy" or "sell"
            qty: Quantity to trade (positive)
            last_close: Last close price for reference
            portfolio_avg_entry_price: Average entry price for SELL guard (optional)
            
        Returns:
            CCXTExecutionResult with status "open", "filled", "rejected", or "error"
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
            
            # Compute edge threshold and components
            edge_bps_total, components = self._compute_edge_bps_total(bid, ask)
            
            # Check spread guard
            if components["spread_bps"] > self.config.max_spread_bps:
                return {"status": "rejected", "reason": "spread_guard"}
            
            # Convert edge and maker offset from bps to fraction
            edge = edge_bps_total * 1e-4
            maker_offset = self.config.maker_offset_bps * 1e-4
            
            # Compute limit price with correct direction
            # BUY: cheaper than bid (bid * (1 - edge))
            # SELL: more expensive than ask (ask * (1 + edge))
            if side == "buy":
                # Reference price is bid for buying
                reference_price = bid
                # Apply edge: buy cheaper
                limit_price = reference_price * (1.0 - edge)
                # Apply additional maker offset (move further away from book)
                limit_price *= (1.0 - maker_offset)
            else:  # sell
                # Reference price is ask for selling
                reference_price = ask
                # Apply edge: sell higher
                limit_price = reference_price * (1.0 + edge)
                # Apply additional maker offset (move further away from book)
                limit_price *= (1.0 + maker_offset)
                
                # SELL NO-LOSS GUARD: ensure we don't sell below avg_entry_price + edge
                if portfolio_avg_entry_price is not None:
                    required_exit_price = portfolio_avg_entry_price * (1.0 + edge)
                    
                    if limit_price < required_exit_price:
                        # Guard activated: raise limit price to prevent loss
                        computed_limit = limit_price
                        limit_price = required_exit_price
                        
                        print(
                            f"SELL_GUARDED: avg_entry={portfolio_avg_entry_price:.2f} "
                            f"edge={edge:.4f} required_exit={required_exit_price:.2f} "
                            f"computed_limit={computed_limit:.2f} final_limit={limit_price:.2f}"
                        )
            
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
            # Different exchanges support different parameters for post-only orders:
            # - Most exchanges: {"postOnly": True}
            # - Binance: {"timeInForce": "GTX"} (Good-Til-Crossing)
            params = {"postOnly": True}
            
            # Log order placement details
            print(
                f"Placing limit maker order: side={side} qty={qty_quantized:.8f} "
                f"bid={bid:.2f} ask={ask:.2f} "
                f"spread_bps={components['spread_bps']:.2f} "
                f"fee_bps={components['fee_component_bps']:.2f} "
                f"spread_comp_bps={components['spread_component_bps']:.2f} "
                f"slip_bps={components['slippage_component_bps']:.2f} "
                f"profit_bps={components['profit_component_bps']:.2f} "
                f"edge_total_bps={edge_bps_total:.2f} "
                f"limit_price={limit_price:.2f}"
            )
            
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
                # Fallback for exchanges that don't support postOnly parameter
                # Binance and similar exchanges may require timeInForce=GTX instead
                error_msg = str(e).lower()
                if "postonly" in error_msg or "timeInForce" in error_msg or "binance" in self.config.exchange_id.lower():
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

    def place_market_order(self, side: str, qty: float, last_close: float) -> CCXTExecutionResult:
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


__all__ = ["CCXTExecutor", "ExecutorConfig", "CCXTExecutionResult"]
