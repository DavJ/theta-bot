#!/bin/bash
# Confidence benchmark script
# Tests how conf_power affects trading behavior across multiple symbols

set -e

echo "========================================="
echo "CONFIDENCE BENCHMARK - PART 1"
echo "========================================="
echo ""

# 1) confidence OFF (conf_power=0)
echo ">>> SWEEP 1: conf_power=0 (confidence disabled)"
for s in BTCUSDT ETHUSDT XRPUSDT DOTUSDT; do
  echo "=== $s conf_power=0 ==="
  PYTHONPATH=. python scripts/run_backtest.py --csv data/ohlcv_1h/${s}.csv --timeframe 1h --print_metrics --conf_power 0
  echo ""
done

# 2) default confidence (conf_power=1)
echo ">>> SWEEP 2: conf_power=1 (default confidence)"
for s in BTCUSDT ETHUSDT XRPUSDT DOTUSDT; do
  echo "=== $s conf_power=1 ==="
  PYTHONPATH=. python scripts/run_backtest.py --csv data/ohlcv_1h/${s}.csv --timeframe 1h --print_metrics --conf_power 1
  echo ""
done

# 3) stronger confidence gating (conf_power=2)
echo ">>> SWEEP 3: conf_power=2 (stronger confidence gating)"
for s in BTCUSDT ETHUSDT XRPUSDT DOTUSDT; do
  echo "=== $s conf_power=2 ==="
  PYTHONPATH=. python scripts/run_backtest.py --csv data/ohlcv_1h/${s}.csv --timeframe 1h --print_metrics --conf_power 2
  echo ""
done

# 4) optional: confidence + hysteresis penalty (conf_power=1, hyst_conf_k=0.5)
echo ">>> SWEEP 4: conf_power=1 hyst_conf_k=0.5 (confidence-based hysteresis)"
for s in BTCUSDT ETHUSDT XRPUSDT DOTUSDT; do
  echo "=== $s conf_power=1 hyst_conf_k=0.5 ==="
  PYTHONPATH=. python scripts/run_backtest.py --csv data/ohlcv_1h/${s}.csv --timeframe 1h --print_metrics --conf_power 1 --hyst_conf_k 0.5
  echo ""
done

echo "========================================="
echo "CONFIDENCE BENCHMARK COMPLETE"
echo "========================================="
