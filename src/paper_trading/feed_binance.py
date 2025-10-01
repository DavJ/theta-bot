from binance.client import Client
import numpy as np

class BinanceFeed:
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)

    def fetch_closes(self, symbols, interval="1m", lookback="2 day ago UTC"):
        data = {}
        for sym in symbols:
            kl = self.client.get_historical_klines(sym, interval, lookback)
            closes = [float(k[4]) for k in kl]
            if len(closes) > 100:
                data[sym] = closes
        return data
