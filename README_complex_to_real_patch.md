# Komplexní → reálné featury (FFT)

Patch přidává do FFT tři režimy převodu komplexních koeficientů na reálné featury:
- `abs` – pouze magnitudy (nejjednodušší)
- `reim` – reálná + imaginární část
- `magphase` – magnituda + cos/sin fáze (výchozí, stabilní vůči skokům fáze)

Použití:
```bash
python -m tests_backtest.cli.run_theta_benchmark   --symbol BTCUSDT --interval 5m --limit 10000   --variants raw fft wavelet theta theta_gs   --models logreg kalman   --fft-mode magphase   --upper 0.52 --lower 0.48   --ema-prob 0.2   --outdir reports_theta_benchmark_v5
```
