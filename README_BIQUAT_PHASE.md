
# theta_eval_hbatch_biquat — biquaternion phase evaluator

This drop-in tool extends `theta_eval_hbatch.py` functionality with a **biquaternion phase** basis.
It can fetch OHLCV from Binance via `ccxt` or read from a local CSV (`time,close`).

## Install

```bash
pip install ccxt pandas numpy
```

## Run (Binance symbols)

```bash
python theta_eval_hbatch_biquat.py \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT \
  --interval 1h --window 256 --horizon 4 \
  --minP 24 --maxP 480 --nP 12 \
  --sigma 0.8 --lambda 1e-3 --limit 2000 \
  --phase biquat \
  --out hbatch_summary.csv
```

## Run (local CSV)

`mydata.csv` must contain columns: `time` (ISO or epoch ms) and `close`.

```bash
python theta_eval_hbatch_biquat.py \
  --symbols mydata.csv \
  --interval 1h --window 256 --horizon 4 \
  --minP 24 --maxP 480 --nP 12 \
  --sigma 0.8 --lambda 1e-3 --limit 2000 \
  --phase biquat \
  --out hbatch_from_csv.csv
```

## Phases

- `simple`  → Fourier (cos/sin) with QR-orthonormalization
- `complex` → complex-exponential expanded to (cos,sin) real basis
- `biquat`  → **biquaternion phase** basis with 4 real channels per period (scalar + 3 vector parts)

The biquat basis uses a unit vector `u` derived from a smoothed carrier to expand
`exp((a i + b j + c k) θ) = cos(|v|θ) + (u_i i + u_j j + u_k k) sin(|v|θ)`
and stacks all periods, then QR-orthonormalizes the resulting design matrix.
Ridge regression (λ) fits levels over the sliding window; forecast is produced
by evaluating the same basis at `t = (W-1)+H`.

## Outputs

For each symbol:
- `eval_h_<SYMBOL>.csv`
- `sum_h_<SYMBOL>.json`

And the combined `--out` CSV.

