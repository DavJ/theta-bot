from binance.client import Client
import pandas as pd
import datetime

def get_historical_klines(symbol="BTCUSDT", interval="1h", start_str="1 Jan 2022", end_str=None, api_key=None, api_secret=None):
    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float, "volume": float
    })
    return df[["open", "high", "low", "close", "volume"]]

if __name__ == "__main__":
    # Testovací běh bez klíče, pouze pro ověření struktury (zde by se normálně vložily API klíče)
    import os
    df = get_historical_klines(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_1HOUR,
        start_str="1 Jan 2024",
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET")
    )
    print(df.head())
