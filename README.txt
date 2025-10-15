Theta PLL (EKF) Patch
----------------------
Nová varianta: **thetaPLL** (EKF/PLL nad rezidui + Kalman trend).

Model:
  • Trend (reálná část): Kalman s lokálním lineárním trendem [level,slope].
  • Oscilace (imaginární část): pro top-N frekvencí běží EKF nad stavem
      x_i = [A_i, B_i, phi_i, omega_i, domega_i]
    s nelineární měřením y = Σ (A_i cos(phi_i) + B_i sin(phi_i)).
  • Inicializace (FFT): ω z FFT píků, φ z arg(Y_k), A/B z |Y_k|.

Spuštění (příklad 15m, 1h + 1d):
  python -m tests_backtest.cli.run_theta_forecast \
    --symbol BTCUSDT --interval 15m --limit 10000 \
    --variants raw fft thetaPLL \
    --horizons 1h 1d \
    --window 256 --horizon-alpha 1.0 \
    --fft-topn 8 \
    --pll-topn 1 --pll-min-period-bars 24 \
    --pll-qL 1e-4 --pll-qS 1e-5 --pll-r 2e-2 \
    --pll-qA 1e-5 --pll-qB 1e-5 --pll-qphi 1e-6 --pll-qomega 1e-6 --pll-qdomega 1e-8 --pll-R 1e-2 \
    --pll-gamma 0.0 --pll-shrink 0.0 \
    --outdir reports_forecast

Vyhodnocení:
  python -m tests_backtest.cli.eval_theta_forecast \
    --csv reports_forecast/forecast_BTCUSDT_15m_win256_h4-96_pure1.csv \
    --metric MSE --outdir reports_forecast

Tipy:
  • 1h: min_period_bars 24, pll-gamma 0.0, pll-shrink 0.0 (reaktivní).
  • 1d: min_period_bars 48–96, pll-gamma 0.05–0.08, pll-shrink 0.1–0.2 (stabilní).
  • Zpřísnit fázovou stabilitu: snižte pll-qdomega (např. 1e-9) a pll-qomega.
