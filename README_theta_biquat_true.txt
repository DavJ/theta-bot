Theta Biquaternionic TRUE (BCH-2) — Full Patch
==================================================

Nový runner: tests_backtest/cli/run_theta_biquat_true.py

Spuštění (příklad):
  python -m tests_backtest.cli.run_theta_biquat_true     --symbol BTCUSDT --interval 15m --limit 10000     --variants raw thetaBiquatTrue     --horizons 1h 1d     --window 256 --horizon-alpha 1.0     --biq-zero-pad 4 --biq-min-period-bars 24 --biq-lam 1e-5     --biq-mode ema3 --biq-const 0,0,0 --biq-scale 1.0,0.5,0.5 --biq-ema 32,64,32     --biq-w-alpha 0.5 --biq-w-ema 32     --biq-topn 2 --biq-scan-every 8     --biq-max-coherence 0.25 --biq-max-cond 10000     --log-steps     --outdir reports_forecast

Vyhodnocení:
  python -m tests_backtest.cli.eval_theta_forecast --csv <soubor.csv> --metric MSE --outdir reports_forecast
  python -m tests_backtest.cli.eval_theta_directional --csv <soubor.csv> --fee-bps 1.0 --outdir reports_forecast
