README — eval_theta_directional_plus
======================================
Použití:
---------
python -m tests_backtest.cli.eval_theta_directional_plus \
  --csv reports_forecast/forecast_BTCUSDT_15m_win256_h4_pure1.csv \
  --outdir reports_forecast \
  --fee-bps 0.5 \
  --thr 350 \
  --confirm-k 3 \
  --deadband-sigma 0.7

Poznámky:
- --confirm-k N: vyžaduje, aby posledních N predikovaných směrů bylo shodných (omezuje přepínání).
- --deadband-sigma S: mrtvé pásmo kolem nuly na základě sigma(pred_delta). Pokud je |pred - close| menší než S*sigma, trade se nebere.
- sign_acc a ostatní směrové metriky počítáme přes všechny řádky, PnL a trades jen pro ty, které prošly filtry (thr/deadband/confirm-k).
- PnL model: 1 jednotka podkladu; poplatek je round-trip: 2 * fee_bps/1e4 * close[t].
