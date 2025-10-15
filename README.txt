Theta PLL Guard Patch
----------------------
Nová varianta: **thetaPLLGuard** – PLL s ochranami proti rozjezdu fáze:
  • Gated update: když je inovace > k·σ, zvýší se dočasně měřicí šum (R×boost) a update je jemnější
  • Limity na |Δω| a |ω̇| (kvadratika fáze se drží na uzdě)
  • Auto-FFT reseed po N po sobě jdoucích outlierech (krátké okno reziduí)
  • Post-reset tlumení (gamma/shrink) na pár barů
  • Trend: Kalman (level+slope)

Spuštění (15m, 1h):
  python -m tests_backtest.cli.run_theta_forecast \
    --symbol BTCUSDT --interval 15m --limit 10000 \
    --variants raw thetaPLLGuard \
    --horizons 1h \
    --window 256 --horizon-alpha 1.0 \
    --pll-topn 1 --pll-min-period-bars 24 \
    --pll-qL 2e-5 --pll-qS 5e-7 --pll-r 2e-2 \
    --pll-qA 2e-6 --pll-qB 2e-6 --pll-qphi 1e-7 --pll-qomega 5e-8 --pll-qdomega 1e-10 --pll-R 1e-2 \
    --pll-guard-k 3.0 --pll-guard-boost 10.0 --pll-guard-max-misses 3 \
    --pll-reseed-window 96 --pll-reseed-topn 1 \
    --pll-dw-max 0.02 --pll-d2w-max 0.001 \
    --pll-gamma 0.0 --pll-shrink 0.0 --pll-post-gamma 0.05 --pll-post-shrink 0.1 --pll-post-len 16 \
    --outdir reports_forecast

Vyhodnocení:
  python -m tests_backtest.cli.eval_theta_forecast \
    --csv reports_forecast/forecast_BTCUSDT_15m_win256_h4_pure1.csv \
    --metric MSE --outdir reports_forecast

Poznámky:
  • Pro 1d zvyšte pll-min-period-bars (48–96), snižte qomega/qdomega, zapněte gamma/shrink.
  • Pokud reseed nechcete, dejte --pll-reseed-window 0.
