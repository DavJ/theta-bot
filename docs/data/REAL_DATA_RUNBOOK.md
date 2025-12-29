# Real Data Runbook (Spot + Derivatives)

This runbook describes a **one-command** pipeline to fetch real Binance SPOT + USD-M futures data for BTCUSDT and ETHUSDT, run sanity checks, and execute the derivatives SDE evaluation. No API keys are required.

## Quickstart

```bash
# Default window can be overridden with env vars
START=2024-01-01
END=2024-10-01
SYMBOLS=BTCUSDT,ETHUSDT

./scripts/run_real_data_pipeline.sh "$SYMBOLS" "$START" "$END"
```

The pipeline:
1. Downloads metadata and raw datasets (skips existing files unless `FORCE=1`).
2. Builds basis from spot + mark closes.
3. Runs sanity checks.
4. Runs derivatives SDE decomposition and evaluation.
5. Writes reports to `reports/DERIVATIVES_SDE_DECOMPOSITION.md` and `reports/DERIVATIVES_SDE_EVAL.md`.

## Manual Commands

```bash
python scripts/data/binance_fetch_metadata.py
python scripts/data/binance_download_spot_klines.py --symbols BTCUSDT,ETHUSDT --interval 1h --start 2024-01-01 --end 2024-10-01
python scripts/data/binance_download_funding.py     --symbols BTCUSDT,ETHUSDT --start 2024-01-01 --end 2024-10-01
python scripts/data/binance_download_mark_klines.py --symbols BTCUSDT,ETHUSDT --interval 1h --start 2024-01-01 --end 2024-10-01
python scripts/data/binance_download_open_interest_hist.py --symbols BTCUSDT,ETHUSDT --start 2024-01-01 --end 2024-10-01
python scripts/data/build_basis.py --symbols BTCUSDT,ETHUSDT --interval 1h
python scripts/data/check_derivatives_sanity.py --symbols BTCUSDT,ETHUSDT --interval 1h --start 2024-01-01 --end 2024-10-01

python scripts/derivatives_sde/run_decompose.py --symbols BTCUSDT,ETHUSDT --start 2024-01-01 --end 2024-10-01 --q 0.85
python scripts/derivatives_sde/run_eval.py       --symbols BTCUSDT,ETHUSDT --start 2024-01-01 --end 2024-10-01 --q 0.85 --horizons 1,3,6,12,24
```

## Troubleshooting

- **HTTP 429 / 418 (rate limit)**  
  - Scripts pace requests to ≤2 req/s and retry with exponential backoff.  
  - If limits persist: rerun after waiting, reduce `--max-requests-per-second`, or use `FORCE=1` to refresh selectively.

- **Open interest API unavailable**  
  - The downloader automatically falls back to the official bulk archive (`https://data.binance.vision`).  
  - You can pre-download monthly/daily archives and place the CSVs in `data/raw/futures/` to be reused (cache honored unless `--force`).

- **Partial datasets**  
  - Re-run the specific downloader with `--force` for that symbol.  
  - Use `check_derivatives_sanity.py` to verify monotonic timestamps, 1h step size, NaN-free required columns, and overlap windows.

- **Reproducibility**  
  - All timestamps are UTC close-times.  
  - Outputs are gzip CSVs with deterministic ordering; reruns with the same inputs produce identical files.

## Paths & Outputs
- Raw data: `data/raw/spot/`, `data/raw/futures/`
- Processed: `data/processed/derivatives_sde/`
- Metadata: `data/metadata/futures_exchangeInfo.json` (+ symbols table)
- Reports: `reports/DERIVATIVES_SDE_DECOMPOSITION.md`, `reports/DERIVATIVES_SDE_EVAL.md`

Be polite to Binance: keep request rates low, rely on caching, and prefer the official bulk archive when available.
