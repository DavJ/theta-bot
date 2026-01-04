"""
Data provider interfaces for Spot Bot 2.0.

The providers separate historical batch access from live streaming so runners
can swap implementations without altering downstream logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional


class DataProvider(ABC):
    """Abstract source of market data for long/flat strategies."""

    @abstractmethod
    def fetch(self, start: Optional[Any] = None, end: Optional[Any] = None) -> Any:
        """
        Retrieve market data between optional boundaries.
        Concrete implementations can return DataFrames, iterables, or domain objects.
        """


class HistoricalDataProvider(DataProvider):
    """Loads historical data for offline feature generation and backtests."""

    def fetch(self, start: Optional[Any] = None, end: Optional[Any] = None) -> Any:
        """Return a dataset spanning the requested window for replay."""
        raise NotImplementedError("Historical data retrieval is not implemented yet.")


class LiveDataProvider(DataProvider):
    """Streams or polls live market data for the live runner."""

    def fetch(self, start: Optional[Any] = None, end: Optional[Any] = None) -> Any:
        """Return the latest snapshot when polled."""
        raise NotImplementedError("Live polling is not implemented yet.")

    def stream(self) -> Iterable[Any]:
        """Yield live updates suitable for incremental feature updates."""
        raise NotImplementedError("Live streaming is not implemented yet.")
