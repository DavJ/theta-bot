from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class BrokerConfig:
    fee_rate: float = 0.001
    min_notional: float = 10.0
    max_exposure: float = 0.3
    starting_usdt: float = 1000.0
    starting_btc: float = 0.0


class PaperBroker:
    """Minimal paper broker that simulates spot fills at the provided price."""

    def __init__(self, config: BrokerConfig | None = None) -> None:
        cfg = config or BrokerConfig()
        self.fee_rate = float(cfg.fee_rate)
        self.min_notional = float(cfg.min_notional)
        self.max_exposure = float(cfg.max_exposure)
        self.usdt = float(cfg.starting_usdt)
        self.btc = float(cfg.starting_btc)

    def get_balances(self) -> Dict[str, float]:
        return {"usdt": self.usdt, "btc": self.btc}

    def _equity(self, price: float) -> float:
        return self.usdt + self.btc * price

    def place_order(self, side: str, qty: float, price: float) -> Dict[str, float | str]:
        qty = float(qty)
        notional = qty * price
        if qty <= 0 or notional < self.min_notional:
            return {"status": "rejected", "reason": "min_notional", "action": side, "qty": qty, "price": price, "fee": 0.0}

        equity = self._equity(price)
        target_btc = self.btc + qty if side.lower() == "buy" else self.btc - qty
        exposure = abs(target_btc * price) / equity if equity > 0 else 0.0
        if exposure > self.max_exposure:
            return {"status": "rejected", "reason": "max_exposure", "action": side, "qty": qty, "price": price, "fee": 0.0}

        fee = notional * self.fee_rate
        if side.lower() == "buy":
            total_cost = notional + fee
            if total_cost > self.usdt:
                return {"status": "rejected", "reason": "insufficient_usdt", "action": side, "qty": qty, "price": price, "fee": fee}
            self.usdt -= total_cost
            self.btc += qty
        else:
            if qty > self.btc:
                qty = self.btc
                notional = qty * price
                fee = notional * self.fee_rate
            self.usdt += notional - fee
            self.btc -= qty

        return {"status": "filled", "action": side, "qty": qty, "price": price, "fee": fee}

    def mark_to_market(self, price: float) -> Dict[str, float]:
        equity = self._equity(price)
        return {"equity_usdt": equity, "btc": self.btc, "usdt": self.usdt}
