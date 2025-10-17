Kalman & Post-processing Patch for thetaBiquatTrue
====================================================

Tento patch přidává stabilizační post-procesing do runneru:
- --post-kalman : jednoduchý 1D Kalman (random-walk) přes sekvenci predikcí (per-horizon)
- --post-kalman-r-mult, --post-kalman-q-mult : nastavení R a Q (Q = q_mult * R), R škáluje lokální varianci
- --post-scale : dodatečný shrink predikce relativně k aktuálnímu close

Použití (příklad):
  patch -p0 < patch_1.patch && patch -p0 < patch_2.patch && patch -p0 < patch_3.patch
  # nebo ručně aplikujte změny do tests_backtest/cli/run_theta_biquat_true.py

  python -m tests_backtest.cli.run_theta_biquat_true     --symbol BTCUSDT --interval 15m --limit 10000     --variants raw thetaBiquatTrue     --horizons 1h     --window 256 --horizon-alpha 1.0     --biq-zero-pad 4 --biq-min-period-bars 48 --biq-lam 1e-4     --biq-mode ema3 --biq-const 0,0,0 --biq-scale 0.5,0.25,0.25 --biq-ema 32,64,32     --biq-w-alpha 0.3 --biq-w-ema 32     --biq-topn 1 --biq-scan-every 8     --biq-max-coherence 0.20 --biq-max-cond 8000     --biq-shrink 0.25 --biq-damping 0.01     --post-scale 0.75     --post-kalman --post-kalman-r-mult 1.0 --post-kalman-q-mult 0.1     --log-steps     --outdir reports_forecast
