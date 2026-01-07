"""
Account provider abstraction for portfolio state management.

Provides unified interface for getting portfolio state in live vs simulated modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol

from spot_bot.core.portfolio import compute_equity, compute_exposure
from spot_bot.core.types import PortfolioState


class AccountProvider(ABC):
    """Abstract interface for portfolio state providers."""

    @abstractmethod
    def get_portfolio_state(self, price: float) -> PortfolioState:
        """
        Get current portfolio state.

        Args:
            price: Current market price for equity/exposure calculation

        Returns:
            PortfolioState with usdt, base, equity, exposure
        """
        ...


class SimAccountProvider(AccountProvider):
    """Simulated account provider using local state."""

    def __init__(self, initial_usdt: float, initial_base: float = 0.0) -> None:
        """
        Initialize simulated account.

        Args:
            initial_usdt: Starting USDT balance
            initial_base: Starting base position (default 0.0)
        """
        self._usdt = float(initial_usdt)
        self._base = float(initial_base)

    def get_portfolio_state(self, price: float) -> PortfolioState:
        """Get current simulated portfolio state."""
        equity = compute_equity(self._usdt, self._base, price)
        exposure = compute_exposure(self._base, price, equity)
        return PortfolioState(
            usdt=self._usdt,
            base=self._base,
            equity=equity,
            exposure=exposure,
        )

    def update_balances(self, usdt: float, base: float) -> None:
        """
        Update balances directly (for simulated fills).

        Args:
            usdt: New USDT balance
            base: New base position
        """
        self._usdt = float(usdt)
        self._base = float(base)


class LiveAccountProvider(AccountProvider):
    """Live account provider fetching real balances from exchange."""

    def __init__(self, exchange: Optional[object] = None, symbol: str = "BTC/USDT") -> None:
        """
        Initialize live account provider.

        Args:
            exchange: CCXT exchange instance
            symbol: Trading pair symbol (e.g., "BTC/USDT")
        """
        self.exchange = exchange
        self.symbol = symbol
        # Parse base and quote from symbol
        parts = symbol.split("/")
        if len(parts) == 2:
            self.base_currency = parts[0]
            self.quote_currency = parts[1]
        else:
            self.base_currency = "BTC"
            self.quote_currency = "USDT"

    def get_portfolio_state(self, price: float) -> PortfolioState:
        """
        Fetch current portfolio state from exchange.

        Args:
            price: Current market price

        Returns:
            PortfolioState with real balances

        Raises:
            RuntimeError: If exchange is not set or fetch fails
        """
        if self.exchange is None:
            raise RuntimeError("Exchange not configured for LiveAccountProvider")

        try:
            balance = self.exchange.fetch_balance()
            free_balances = balance.get("free", {})

            usdt = float(free_balances.get(self.quote_currency, 0.0))
            base = float(free_balances.get(self.base_currency, 0.0))

            equity = compute_equity(usdt, base, price)
            exposure = compute_exposure(base, price, equity)

            return PortfolioState(
                usdt=usdt,
                base=base,
                equity=equity,
                exposure=exposure,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch balance from exchange: {exc}") from exc


__all__ = [
    "AccountProvider",
    "SimAccountProvider",
    "LiveAccountProvider",
]
