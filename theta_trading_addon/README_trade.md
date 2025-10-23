
# Trading Add-on (position sizing, TP/SL, costs)

Tento balíček rozšiřuje backtest o realistickou **obchodní vrstvu**:
- dynamické řízení pozice pomocí z-score predikce,
- TP/SL podle rolling volatility,
- model poplatků a skluzu (bps),
- výstupy: equity křivka, trade log a shrnutí.

## Rychlé spuštění (na datech z evaluátoru)
```
bash scripts/trade_sim.sh
```

Výstupy najdeš v `results/`:
- `summary_trading.csv` – přehled za všechny páry (OOS část),
- `trades_<SYMBOL>.csv` – seznam obchodů,
- `equity_<SYMBOL>.csv` – equity křivka.

## Parametry (konfig)
Konfigurace je v `configs/trade_params.json`. Dá se přepsat CLI argumenty:
- `--fees-bps`, `--slippage-bps`
- `--z-entry` a `--z-window` pro z-score gating
- `--position-cap` pro omezení páky pozice
- `--tp-sigma`, `--sl-sigma` – násobky rolling sigma pro TP/SL

## Live / Paper trading (skeleton)
`theta_trade_loop.py` spustí jednorázový výpočet signálu (dry-run):
```
python theta_trade_loop.py --symbol BTCUSDT --interval 1h --dry-run
```
Loguje do `results/live_signals.csv`. Napojení na API je oddělené (bezpečné).
