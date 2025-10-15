# Komplexní výpočet + Kalman → reálná pravděpodobnost

Tento patch přidává:
- `fft_complex` – komplexní výběr top-N FFT koeficientů (normalizovaných per okno)
- `ComplexKalmanDiag` – 2D (Re,Im) Kalman filtr per-bin, který odhaduje drift v komplexní rovině
- `ckalman` – nový model v CLI, který používá výhradně komplexní FFT path a vrací reálné `p` v [0,1]

## Spuštění (doporučeno)
```bash
python -m tests_backtest.cli.run_theta_benchmark   --symbol BTCUSDT --interval 5m --limit 20000   --variants raw fft wavelet   --models logreg ckalman   --fft-topn 12   --ema-prob 0.2   --upper 0.52 --lower 0.48   --outdir reports_theta_benchmark_v6
```
> Pokud zadáš `--models ... ckalman` a zároveň `--variants ... fft`, skript interně přidá `fft_complex` tak, aby bylo co krmit komplexním Kalmanem.

Další kroky: analogicky zavedeme `theta_complex` a Morlet CWT (mag+phase).
