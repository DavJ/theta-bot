#!/usr/bin/env bash
set -euo pipefail

SYMBOLS="${1:-BTCUSDT,ETHUSDT}"
START="${2:-2024-01-01}"
END="${3:-2024-10-01}"
INTERVAL="${INTERVAL:-1h}"
Q="${Q:-0.85}"
FORCE_FLAG=""

if [[ "${FORCE:-0}" == "1" || "${FORCE:-false}" == "true" ]]; then
  FORCE_FLAG="--force"
fi

echo "Running real-data pipeline for symbols=${SYMBOLS}, start=${START}, end=${END}, interval=${INTERVAL}"

python scripts/data/binance_fetch_metadata.py ${FORCE_FLAG}
sleep 1

python scripts/data/binance_download_spot_klines.py --symbols "${SYMBOLS}" --interval "${INTERVAL}" --start "${START}" --end "${END}" ${FORCE_FLAG}
sleep 1

python scripts/data/binance_download_funding.py --symbols "${SYMBOLS}" --start "${START}" --end "${END}" ${FORCE_FLAG}
sleep 1

python scripts/data/binance_download_mark_klines.py --symbols "${SYMBOLS}" --interval "${INTERVAL}" --start "${START}" --end "${END}" ${FORCE_FLAG}
sleep 1

python scripts/data/binance_download_open_interest_hist.py --symbols "${SYMBOLS}" --start "${START}" --end "${END}" ${FORCE_FLAG}
sleep 1

python scripts/data/build_basis.py --symbols "${SYMBOLS}" --interval "${INTERVAL}" ${FORCE_FLAG}
sleep 1

python scripts/data/check_derivatives_sanity.py --symbols ${SYMBOLS//,/ } --interval "${INTERVAL}" --start "${START}" --end "${END}"

python scripts/derivatives_sde/run_decompose.py --symbols "${SYMBOLS}" --start "${START}" --end "${END}" --q "${Q}" --interval "${INTERVAL}"
python scripts/derivatives_sde/run_eval.py --symbols "${SYMBOLS}" --start "${START}" --end "${END}" --q "${Q}" --interval "${INTERVAL}" --horizons "1,3,6,12,24"

echo "Pipeline complete. Reports written to reports/DERIVATIVES_SDE_DECOMPOSITION.md and reports/DERIVATIVES_SDE_EVAL.md"
