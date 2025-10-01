# Theta-Bot Paper Trading Patch

This patch adds a **functional paper-trading bot** using Theta transform + Kalman.
It supports **both historical backtests (CSV)** and **real-time paper trading** using Binance klines.
It **does not execute real orders** – it simulates fills, fees, slippage, and logs trades.

## How to install (unzip at your project root)
```bash
unzip ~/Downloads/theta-bot-paper-trading-patch.zip
```

This adds:
- `src/cli/paper_trade.py`
- `src/paper_trading/*`
- `compose/.env.paper.example`
- (No existing files overwritten unless you already have files at these exact paths)

## Quick start (real-time paper on BTCUSDT & ETHUSDT)
1) Set environment variables:
```bash
cp compose/.env.paper.example compose/.env.paper
# Edit compose/.env.paper to set SYMBOLS, INTERVAL etc.
```
2) Run with your Python env (or inside your container image):
```bash
python -m cli.paper_trade
```
The bot will:
- fetch klines every LOOP_SECONDS (default 30s),
- compute theta residuals + Kalman smoothing,
- generate a hedged pair signal,
- simulate fills & PnL, and log trades to `paper_trades.csv`, equity to `equity_curve.csv`.

## Backtest on historical CSV
Prepare a CSV with columns: `timestamp,open,high,low,close,volume` (UNIX ms or ISO8601 timestamp).
Then run:
```bash
MODE=historical CSV_PATH=/path/to/data.csv python -m cli.paper_trade
```
You can also pass two CSVs (comma-separated) to backtest two symbols simultaneously:
```bash
MODE=historical CSV_PATH="/path/to/BTC.csv,/path/to/ETH.csv" python -m cli.paper_trade
```

## Strategy (v1)
- **ThetaBasis** (fast approximation) → residuals
- **Kalman smoother** on residuals → innovations
- **Signal**:
  - For each symbol: `sign(innovation_t - innovation_{t-1})` (momentum)
  - Pair-hedge: net exposure limited; if trading two symbols, the second leg hedges 50% of the first
- **Risk**: max position size per symbol (as % of equity), per-trade fee + slippage, cooldown

## Outputs
- `paper_trades.csv`: timestamp,symbol,side,qty,price,fee,reason
- `equity_curve.csv`: timestamp,equity,positions_json

## Notes
- This is a minimal, safe baseline. It **won't send real orders**.
- You can later replace `theta_kalman.py` with your legacy implementations via the legacy bridge.
- Tune hyper-parameters in `compose/.env.paper` or via environment variables when running.

