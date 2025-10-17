README — thetaBiquatTrue (Kalman + post-scale)
=============================================

Co je to?
---------
Hotový runner `tests_backtest/cli/run_theta_biquat_true.py` s:
- pravou bikvaternionovou thetou (BCH-2 bloky),
- block-OMP + vážený QR + guardy (koherence μ, kondice κ),
- **post-scale** (dodatečný shrink amplitudy),
- **Kalman** (1D RW) napříč rolling predikcemi pro každý horizont.

Instalace
---------
Rozbal do kořene projektu:
    unzip -o theta_biquat_true_kalman_ready.zip -d .

Příklad běhu (navazuje na tvůj setup)
-------------------------------------
python -m tests_backtest.cli.run_theta_biquat_true \
  --symbol BTCUSDT --interval 15m --limit 10000 \
  --variants raw thetaBiquatTrue \
  --horizons 1h \
  --window 256 --horizon-alpha 1.0 \
  --biq-zero-pad 4 --biq-min-period-bars 48 --biq-lam 1e-4 \
  --biq-mode ema3 --biq-const 0,0,0 --biq-scale 0.5,0.25,0.25 --biq-ema 32,64,32 \
  --biq-w-alpha 0.3 --biq-w-ema 32 \
  --biq-topn 1 --biq-scan-every 8 \
  --biq-max-coherence 0.20 --biq-max-cond 8000 \
  --biq-shrink 0.25 --biq-damping 0.01 \
  --post-scale 0.75 \
  --post-kalman --post-kalman-r-mult 1.0 --post-kalman-q-mult 0.1 \
  --log-steps \
  --outdir reports_forecast

Vyhodnocení
-----------
python -m tests_backtest.cli.eval_theta_forecast \
  --csv reports_forecast/forecast_BTCUSDT_15m_win256_h4_pure1.csv \
  --metric MSE --outdir reports_forecast

python -m tests_backtest.cli.eval_theta_directional \
  --csv reports_forecast/forecast_BTCUSDT_15m_win256_h4_pure1.csv \
  --outdir reports_forecast --fee-bps 1.0

Poznámky
--------
- `post_scale` a `post-kalman` snižují MSE bez ztráty směrové informace.
- Guardy μ/κ udržují nízkou koherenci báze → stabilnější extrapolace.
- Log JSON obsahuje aktivní bloky a parametry post-procesingu.
