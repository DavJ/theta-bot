from __future__ import annotations

from typing import Optional

import pandas as pd


def fetch_ohlcv_ccxt(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit_total: int = 6000,
    since: Optional[int] = None,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """
    Download OHLCV using ccxt with pagination.

    The import of ccxt is deferred to runtime so that this module can be imported
    without ccxt being installed.
    """
    import ccxt  # type: ignore

    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Exchange '{exchange_id}' is not available in ccxt.")
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    if since is None:
        since = exchange.milliseconds() - (limit_total + 10) * tf_ms
    data = []

    while len(data) < limit_total:
        limit = min(1000, limit_total - len(data))
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        data.extend(batch)
        since = batch[-1][0] + tf_ms

    df = pd.DataFrame(
        data, columns=["ts", "open", "high", "low", "close", "volume"]
    ).drop_duplicates(subset="ts")
    df = df.sort_values("ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df
