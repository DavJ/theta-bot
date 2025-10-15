FFT Refined Patch (pure extrapolation)
--------------------------------------
Přidává variantu **fftRefined** (bez Kalman/PLL):
  • detrend (lineární), Hann okno, zero-padding (4×), RFFT
  • peak-picking + kvadratická interpolace píku → jemnější frekvence
  • extrapolace sumy sinů s odhadnutou fází a amplitudou + přičtení trendu
  • volitelně: damping/shrink, min_period_bars

Spuštění (15m, 1h+1d):
  python -m tests_backtest.cli.run_theta_forecast \
    --symbol BTCUSDT --interval 15m --limit 10000 \
    --variants raw fft fftRefined \
    --horizons 1h 1d \
    --window 256 --horizon-alpha 1.0 \
    --fft-topn 8 \
    --fft-refined-topn 1 --fft-refined-zero-pad 4 \
    --fft-refined-min-period-bars 24 \
    --fft-refined-damping 0.0 --fft-refined-shrink 0.0 \
    --outdir reports_forecast

Vyhodnocení:
  python -m tests_backtest.cli.eval_theta_forecast \
    --csv reports_forecast/forecast_BTCUSDT_15m_win256_h4-96_pure1.csv \
    --metric MSE --outdir reports_forecast

Tipy:
  • 1h: min_period_bars 24, damping 0.0, shrink 0.0
  • 1d: min_period_bars 48–96, damping 0.04–0.08, shrink 0.1–0.2
