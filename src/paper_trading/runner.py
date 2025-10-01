import os, time, json, math
from datetime import datetime
from paper_trading.theta_kalman import ThetaKalman
from paper_trading.broker_paper import PaperBroker
from paper_trading.feed_binance import BinanceFeed
from paper_trading.feed_csv import CSVFeed

class PaperRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.symbols = cfg["symbols"]
        self.interval = cfg["interval"]
        self.feeder_live = BinanceFeed(cfg["binance_api_key"], cfg["binance_api_secret"])
        self.model = ThetaKalman()
        self.broker = PaperBroker(
            fee_bps=cfg["fee_bps"], slip_bps=cfg["slip_bps"],
            base_equity=cfg["base_equity"], max_w_per_symbol=cfg["max_w_per_symbol"],
            hedge_ratio=cfg["hedge_ratio"], cooldown_steps=cfg["cooldown_steps"],
            trades_path=cfg["trades_path"], equity_path=cfg["equity_path"]
        )

    def _signals_from_prices(self, closes_dict):
        # closes_dict: {symbol: [closes...]}
        signals = {}
        innovs = {}
        for sym, closes in closes_dict.items():
            res = self.model.process(closes)
            if res is None:
                continue
            innov = res["innov"]
            if len(innov) < 2:
                continue
            sig = 1 if (innov[-1] - innov[-2]) > 0 else -1
            signals[sym] = sig
            innovs[sym] = float(innov[-1])
        # pair-hedge: if two symbols, hedge second against first
        if len(self.symbols) >= 2:
            s1, s2 = self.symbols[:2]
            if s1 in signals and s2 in signals:
                # net exposure control
                signals[s2] = -int(self.cfg["hedge_ratio"] * signals[s1])
        return signals, innovs

    def run_live(self):
        loop_s = self.cfg["loop_seconds"]
        print(f"[paper] live mode | symbols={self.symbols} interval={self.interval} loop={loop_s}s")
        while True:
            closes = self.feeder_live.fetch_closes(self.symbols, self.interval)
            if closes:
                signals, innovs = self._signals_from_prices(closes)
                prices = {sym: closes[sym][-1] for sym in closes}
                self.broker.step(signals, prices)
            time.sleep(loop_s)

    def run_historical(self):
        csvs = [p.strip() for p in self.cfg["csv_path"].split(",") if p.strip()]
        feeders = []
        if not csvs:
            raise RuntimeError("CSV_PATH not provided for historical mode")
        for i, sym in enumerate(self.symbols):
            path = csvs[min(i, len(csvs)-1)]
            feeders.append(CSVFeed(sym, path))
        # iterate in lockstep over rows
        while True:
            closes = {}
            prices = {}
            for f in feeders:
                row = f.next_row()
                if row is None:
                    return  # done
                closes.setdefault(f.symbol, []).append(row["close"])
                prices[f.symbol] = row["close"]
            signals, innovs = self._signals_from_prices({k: v for k, v in closes.items()})
            self.broker.step(signals, prices)
