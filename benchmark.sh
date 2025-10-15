#!/usr/bin/env bash
set -euo pipefail

OUTDIR="reports_cmp_theta_new"
SYMBOL="${1:-BTCUSDT}"
INTERVAL="${2:-5m}"
LIMIT="${3:-20000}"

echo "[i] RAW & WAVELET grid..."
python -m tests_backtest.cli.run_theta_benchmark   --symbol "$SYMBOL" --interval "$INTERVAL" --limit "$LIMIT"   --variants raw wavelet   --models logreg   --upper-grid "0.504,0.508,0.512"   --lower-grid "0.496,0.492,0.488"   --outdir "${OUTDIR}_raw_wavelet"

echo "[i] THETA 1D/2D/3D (ckalman+logreg fallback) ..."
python -m tests_backtest.cli.run_theta_benchmark   --symbol "$SYMBOL" --interval "$INTERVAL" --limit "$LIMIT"   --variants theta1D theta2D theta3D   --models ckalman logreg   --theta-K 48 --theta-tau-re 0.03   --upper-grid "0.504,0.508,0.512"   --lower-grid "0.496,0.492,0.488"   --outdir "${OUTDIR}_theta"

echo "[i] DONE. See ${OUTDIR}_raw_wavelet and ${OUTDIR}_theta"
