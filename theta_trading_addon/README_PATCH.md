# Theta Trading Patch (auto CSV + robustní evaluace)

Co je uvnitř:
- `make_prices_csv.py` – stáhne OHLCV z Binance (ccxt) a uloží do `prices/<SYMBOL>_<interval>.csv`
- `theta_trading_addon/robustness_suite_v4_trade.py` – opraveno parsování `--symbols` (mix CSV/tickerů), automatické stažení CSV, evaluátor volán bez uppercasu cest
- `theta_trading_addon/theta_trade_loop.py` – přijme ticker nebo CSV; pro ticker auto-stáhne CSV
- `theta_trading_addon/scripts/trade_sim.sh` – jednoduchý wrapper

Rychlý start:
```bash
pip install ccxt pandas
python make_prices_csv.py --symbols BTCUSDT,ETHUSDT --interval 1h --limit 1000

python theta_trading_addon/robustness_suite_v4_trade.py \
  --symbols BTCUSDT,ETHUSDT \
  --interval 1h --window 256 --horizon 4 \
  --minP 24 --maxP 480 --nP 16 --sigma 0.8 --lambda 1e-3 \
  --pred-ensemble avg --max-by transform \
  --fees-bps 5 --slippage-bps 2 \
  --z-entry 0.8 --z-window 96 --position-cap 1.0 \
  --tp-sigma 1.25 --sl-sigma 1.5 \
  --out theta_trading_addon/results/summary_trading_custom.csv
```
