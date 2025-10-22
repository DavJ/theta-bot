# Theta Predictor Patch

Obsah balíčku:
- `theta_predictor.py` — aktuální prediktor (numerická predikce + LONG/SHORT/FLAT).
- `theta_backtest.py` — rolling walk-forward backtest, metriky a equity křivka.
- `theta_batch.py` — dávkové spuštění pro více symbolů s CSV výstupem.

## Rychlé příklady

### 1) Jednorázová predikce
```bash
python theta_predictor.py --symbol BTCUSDT --interval 1h \
  --horizons 1,4,24 --window 256 --fee-bps 5 --out out.csv
```

### 2) Rolling backtest
```bash
python theta_backtest.py --symbol BTCUSDT --interval 1h \
  --window 256 --horizon 4 --minP 24 --maxP 480 --nP 8 \
  --sigma 0.8 --lambda 1e-3 --fee-bps 5 --limit 2000 \
  --out equity.csv --summary summary.json
```

### 3) Batch přes více symbolů
```bash
python theta_batch.py --symbols BTCUSDT,ETHUSDT,BNBUSDT \
  --interval 1h --horizons 1,4,24 --window 256 --out batch.csv
```

## Poznámky
- Parametr `sigma` určuje \( q = \exp(-(\pi \sigma)^2) \). Doporučené `sigma≈0.8` ⇒ `q≈0.081`.
- QR ortonormalizace a jednoduchý Kalman lze vypnout přepínači `--no-qr`, `--no-kalman`.
- U backtestu se obchoduje na close vstupního baru a vystupuje na close o `horizon` barů později, se započtenými poplatky (round-trip).


### 4) RAW vyhodnocení (bez poplatků a exekuce)
```bash
python theta_eval_raw.py --symbol BTCUSDT --interval 1h \
  --window 256 --horizon 4 --minP 24 --maxP 480 --nP 8 \
  --sigma 0.8 --lambda 1e-3 --limit 2000 \
  --out eval_raw.csv --summary raw_summary.json
```
- Měří shodu směru a korelaci mezi predikovanou a skutečnou změnou ceny.
- Backtest nově umí `--exec open|close`.


### 5) RAW prediktor vs. HSTRATEGY baseline (HOLD)
```bash
python theta_eval_hstrategy.py --symbol BTCUSDT --interval 1h \
  --window 256 --horizon 4 --minP 24 --maxP 480 --nP 8 \
  --sigma 0.8 --lambda 1e-3 --limit 2000 \
  --out eval_hstrategy.csv --summary hstrategy_summary.json
```
- Porovnává směrovou přesnost prediktoru s baseline HOLD (vždy long).


### 6) Batch RAW vs HOLD přes více symbolů
```bash
python theta_eval_hbatch.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT \
  --interval 1h --window 256 --horizon 4 --minP 24 --maxP 480 --nP 8 \
  --sigma 0.8 --lambda 1e-3 --limit 2000 --out hbatch_summary.csv
```
