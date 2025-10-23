#!/usr/bin/env bash
set -euo pipefail

SYMBOLS="${1:-BTCUSDT,ETHUSDT}"
INTERVAL="${2:-1h}"

python theta_trading_addon/robustness_suite_v4_trade.py       --symbols "$SYMBOLS"       --interval "$INTERVAL"       --window 256 --horizon 4       --minP 24 --maxP 480 --nP 16       --sigma 0.8 --lam 1e-3       --pred-ensemble avg --max-by transform       --fees-bps 5 --slippage-bps 2       --z-entry 0.8 --z-window 96 --position-cap 1.0       --tp-sigma 1.25 --sl-sigma 1.5       --out theta_trading_addon/results/summary_trading_custom.csv
