# Theta Forecast Benchmark – Patch

Adds a **predictive evaluation** CLI to compare feature transforms (RAW / FFT / Theta-1D/2D/3D) by *forecast accuracy*.

See `examples/forecast_bench.sh` or run:

```
python -m tests_backtest.cli.run_theta_forecast   --symbol BTCUSDT --interval 5m --limit 10000   --variants raw fft theta1D theta2D theta3D   --window 256 --horizon 12 --horizon-alpha 0.85   --fft-topn 24   --tau 0.12 --tau-re 0.02 --psi 0.0   --ridge 1e-3   --outdir reports_forecast
```
