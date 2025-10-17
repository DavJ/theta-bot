Theta OrthoQR Patch
---------------------
Přidává variantu **thetaOrthoQR** – ortogonální theta-fit s **váženým QR** a **globálním refitem** po každém kroku (OMP+QR).
Báze: \(\cos(\omega\,\tau(t)),\ \sin(\omega\,\tau(t))\) s \(\tau(t)=t+i\psi(t)\).

Psi(t):
  • `--theta-psi-mode const` (default) s `--theta-psi-const`,
  • `--theta-psi-mode ema_absret` → \(\psi(t)=\text{scale}\cdot\text{EMA}(|\Delta p|)\).

Váhy W:
  • `--theta-w-alpha 0` → jednotkové váhy,
  • >0 → W ~ 1 + alpha*EMA(|ret|) (praktický proxy na |dτ/dt|).

Doporučený běh (15m, 1h + 1d):
  python -m tests_backtest.cli.run_theta_forecast \
    --symbol BTCUSDT --interval 15m --limit 10000 \
    --variants raw thetaOrthoQR \
    --horizons 1h 1d \
    --window 256 --horizon-alpha 1.0 \
    --ortho-topn 1 --ortho-zero-pad 4 --ortho-min-period-bars 24 --ortho-lam 1e-5 \
    --theta-psi-mode ema_absret --theta-psi-scale 1.0 --theta-psi-ema 32 --theta-w-alpha 0.5 \
    --ortho-damping 0.0 --ortho-shrink 0.0 \
    --outdir reports_forecast

Vyhodnocení:
  python -m tests_backtest.cli.eval_theta_forecast \
    --csv reports_forecast/forecast_BTCUSDT_15m_win256_h4-96_pure1.csv \
    --metric MSE --outdir reports_forecast

Poznámky:
  • OrthoQR dělá **globální** refit (všech dosud vybraných atomů) přes QR. Je to numericky stabilní a férová rekonstrukce (pseudoinverze).
  • Výběr ω začíná z mřížky (zero-pad FFT), ale fit je v theta-prostoru (cos/sin(ωτ)).
  • Psi lze držet konstantní nebo „vědomou“: EMA(|ret|). Pro další verzi lze přidat i vlastní ψ(t).
