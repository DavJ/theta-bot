# Binance live fetch patch

Tento patch přidává:
- `make_prices_csv.py` (Binance/ccxt OHLCV -> CSV)
- podporu pro **tickery** v `theta_trading_addon/robustness_suite_v4_trade.py` a `theta_trading_addon/theta_trade_loop.py` (auto-fetch, parent lookup)
- skript `scripts/trade_sim.sh` pro rychlé spuštění z rootu

## Instalace závislostí
```
pip install ccxt pandas
```

## Použití
### Batch trading evaluace (tickery, auto-fetch)
```
python theta_trading_addon/robustness_suite_v4_trade.py   --symbols BTCUSDT,ETHUSDT   --interval 1h --window 256 --horizon 4   --minP 24 --maxP 480 --nP 16 --sigma 0.8 --lambda 1e-3   --pred-ensemble avg --max-by transform   --fees-bps 5 --slippage-bps 2   --z-entry 0.8 --z-window 96 --position-cap 1.0   --tp-sigma 1.25 --sl-sigma 1.5   --out theta_trading_addon/results/summary_trading_custom.csv
```

### Jednorázový signál (dry-run)
```
python theta_trading_addon/theta_trade_loop.py --symbol BTCUSDT --interval 1h --dry-run
```

### Spouštěcí skript
```
bash scripts/trade_sim.sh
```

### Konfigurace přes proměnné prostředí
- `EVAL_PATH` — nastav explicitní cestu k `theta_eval_hbatch_biquat_max.py`
- `MAKER_PATH` — nastav explicitní cestu k `make_prices_csv.py`
