Hotfix #2
---------
- Updated make_prices_csv.py to accept `--outdir` (default: prices). It also auto-creates the folder.
Usage:
  python make_prices_csv.py --symbols BTCUSDT,ETHUSDT --interval 1h --limit 1000 --outdir prices
