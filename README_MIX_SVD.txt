README — thetaMixSVD (SVD-ortho) + thetaBiquatTrue
====================================================
Nový runner: tests_backtest/cli/run_theta_mix_svd.py

Spuštění (příklad 15m, 1h):
--------------------------
python -m tests_backtest.cli.run_theta_mix_svd \
  --symbol BTCUSDT --interval 15m --limit 10000 \
  --variants raw thetaBiquatTrue thetaMixSVD \
  --horizons 1h \
  --window 256 --horizon-alpha 1.0 \
  --biq-zero-pad 4 --biq-min-period-bars 48 --biq-lam 2e-4 \
  --biq-mode ema3 --biq-const 0,0,0 --biq-scale 0.5,0.25,0.25 --biq-ema 32,64,32 \
  --biq-w-alpha 0.3 --biq-w-ema 32 \
  --biq-topn 1 --biq-scan-every 4 \
  --biq-max-coherence 0.18 --biq-max-cond 7000 \
  --biq-shrink 0.30 --biq-damping 0.015 \
  --mix-arg-kinds time,lin_psi,lin_phi,lin_xi \
  --mix-phases 0,1.57079632679 \
  --mix-topk-max 6 --mix-use-bic \
  --post-scale 0.65 --post-kalman --post-kalman-r-mult 2.0 --post-kalman-q-mult 0.02 \
  --outdir reports_forecast

Vyhodnocení (stejné nástroje):
------------------------------
python -m tests_backtest.cli.eval_theta_forecast \
  --csv reports_forecast/forecast_BTCUSDT_15m_win256_h4_pure1.csv \
  --metric MSE --outdir reports_forecast

python -m tests_backtest.cli.eval_theta_directional \
  --csv reports_forecast/forecast_BTCUSDT_15m_win256_h4_pure1.csv \
  --outdir reports_forecast --fee-bps 1.0 --thr 75
