from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from spot_bot.persist import SQLiteLogger


@dataclass
class BrokerConfig:
    fee_rate: float = 0.001
    min_notional: float = 10.0
    initial_usdt: float = 1000.0
    initial_btc: float = 0.0
    step_size: float | None = None


class PaperBroker:
    """Simulated spot broker that fills at provided prices."""

    def __init__(
        self,
        initial_usdt: float,
        fee_rate: float,
        min_notional: float = 10.0,
        initial_btc: float = 0.0,
        step_size: float | None = None,
    ) -> None:
        self.usdt = float(initial_usdt)
        self.btc = float(initial_btc)
        self.fee_rate = float(fee_rate)
        self.min_notional = float(min_notional)
        self.step_size = float(step_size) if step_size else None

    @classmethod
    def from_logger(
        cls,
        logger: Optional[SQLiteLogger],
        fee_rate: float,
        min_notional: float = 10.0,
        fallback_usdt: float = 1000.0,
        step_size: float | None = None,
    ) -> "PaperBroker":
        equity = logger.get_latest_equity() if logger else None
        initial_usdt = equity["usdt"] if equity else fallback_usdt
        initial_btc = equity["btc"] if equity else 0.0
        return cls(
            initial_usdt=initial_usdt,
            initial_btc=initial_btc,
            fee_rate=fee_rate,
            min_notional=min_notional,
            step_size=step_size,
        )

    def balances(self) -> Dict[str, float]:
        return {"usdt": float(self.usdt), "btc": float(self.btc)}

    def set_balances(self, usdt: float, btc: float) -> None:
        """Set balances directly (used by core simulation for state sync)."""
        self.usdt = float(usdt)
        self.btc = float(btc)

    def equity(self, price: float) -> float:
        return float(self.usdt + self.btc * float(price))

    def _round_qty(self, qty: float) -> float:
        if qty <= 0:
            return 0.0
        if self.step_size and self.step_size > 0:
            return max(0.0, round(qty / self.step_size) * self.step_size)
        return round(qty, 8)

    def trade_to_target_btc(self, target_btc: float, price: float) -> Dict[str, Any]:
        price = float(price)
        current_btc = float(self.btc)
        delta = float(target_btc) - current_btc
        side = "buy" if delta > 0 else "sell"
        qty_desired = self._round_qty(abs(delta))

        if qty_desired <= 0:
            return {"status": "noop", "side": side, "qty": 0.0, "price": price, "fee": 0.0}

        if side == "buy":
            max_affordable_qty = self._round_qty(self.usdt / (price * (1 + self.fee_rate)))
            qty = min(qty_desired, max_affordable_qty)
        else:
            qty = min(qty_desired, self.btc)

        if qty <= 0:
            return {"status": "rejected", "reason": "insufficient_balance", "side": side, "qty": 0.0, "price": price, "fee": 0.0}

        notional = qty * price
        if notional < self.min_notional:
            return {"status": "rejected", "reason": "min_notional", "side": side, "qty": qty, "price": price, "fee": 0.0}

        fee = notional * self.fee_rate
        if side == "buy":
            cost = notional + fee
            if cost > self.usdt:
                return {"status": "rejected", "reason": "insufficient_balance", "side": side, "qty": qty, "price": price, "fee": fee}
            self.usdt -= cost
            self.btc += qty
        else:
            if qty > self.btc:
                qty = self.btc
                notional = qty * price
                fee = notional * self.fee_rate
            self.usdt += notional - fee
            self.btc -= qty

        status = "filled"
        if qty < qty_desired:
            status = "partial"

        return {"status": status, "side": side, "qty": qty, "price": price, "fee": fee}


__all__ = ["PaperBroker", "BrokerConfig"]
