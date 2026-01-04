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


class CCXTExecutor:
    """Spot-only execution helper with conservative safety limits."""

    def __init__(self, config: ExecutorConfig) -> None:
        self.config = config
        self.exchange = None
        self.trades_today = 0
        self.turnover_today = 0.0
        self._current_day = None

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
            free_usdt = balances.get("free", {}).get("USDT") or balances.get("USDT", {}).get("free")
            return free_usdt is None or float(free_usdt) >= self.config.min_balance_reserve_usdt
        except Exception:
            return True

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
            fee_est = notional * self.config.fee_rate
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
