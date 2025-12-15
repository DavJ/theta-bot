# Running the Profit-Oriented Pipeline

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Walk-forward evaluation

```bash
python scripts/run_walkforward.py --config configs/btc_1h.yaml
```

Outputs are written to `runs/<timestamp>/<config_name>/` and include:
- `metrics.json`
- `predictions.parquet`
- `trades.csv`
- `equity_curve.csv`

## Manual backtest on existing predictions

```bash
python scripts/run_backtest.py --predictions path/to/predictions.parquet
```
