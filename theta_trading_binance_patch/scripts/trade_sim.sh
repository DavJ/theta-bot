#!/usr/bin/env bash
set -euo pipefail

# Spouštěj z rootu repozitáře
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$THIS_DIR/.." && pwd )"

cd "$ROOT_DIR"

PY=python

$PY theta_trading_addon/robustness_suite_v4_trade.py   --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT   --interval 1h --window 256 --horizon 4   --minP 24 --maxP 480 --nP 16 --sigma 0.8 --lambda 1e-3   --pred-ensemble avg --max-by transform   --fees-bps 5 --slippage-bps 2   --z-entry 0.8 --z-window 96 --position-cap 1.0   --tp-sigma 1.25 --sl-sigma 1.5   --out theta_trading_addon/results/summary_trading_batch.csv
