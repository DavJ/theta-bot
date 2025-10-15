# Benchmark v2 – stabilita a férové WF škálování

Co se mění:
- Numericky stabilní sigmoid (bez `overflow` warningů)
- L2 regularizace + gradient clipping (stabilnější u vysokých hodnot featur)
- Standardizace featur pouze podle tréninkové části v každém WF okně
- Volitelné EMA vyhlazení pravděpodobností (`--ema-prob 0.2`)

Použití:
```bash
python -m tests_backtest.cli.run_theta_benchmark   --symbol BTCUSDT --interval 5m --limit 10000   --variants raw fft wavelet theta theta_gs   --models logreg kalman   --ema-prob 0.2   --outdir reports_theta_benchmark_v2
```
