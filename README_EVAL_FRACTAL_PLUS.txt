\
    README — eval_theta_fractal_plus
    ================================

    Co umí navíc:
    - --weighted-sign-acc  : vážená směrová přesnost (váhy = |true_delta|^p)
    - --wexp p             : exponent p (default 1.0)
    - --multi-horizons ... : vybere jen vybrané horizonty, např. "4,8,12" nebo "15m,30m,1h,4h"
                             (časové značky převádí na bary podle intervalu detekovaného z názvu CSV, např. ..._15m_...)
    - --buckets N          : report per-kvantil přes |true_delta| (např. N=10 = decily)

    Příklad:
    --------
    python -m tests_backtest.cli.eval_theta_fractal_plus \
      --csv reports_forecast/forecast_BTCUSDT_15m_win256_h4_pure1.csv \
      --outdir reports_forecast \
      --fee-bps 0.5 \
      --thr 350 \
      --confirm-k 3 \
      --deadband-sigma 0.7 \
      --weighted-sign-acc --wexp 1.5 \
      --multi-horizons 15m,30m,1h \
      --buckets 10
