# Spot Bot 2.0 Go-Live Checklist

1. **Dry run sanity**
   - Install deps: `pip install -r requirements.txt`
   - Update `spot_bot/config.yaml` (symbol, timeframe, max_exposure, fee_rate, min_notional).
   - Run once in dryrun mode (no trades, optional DB for logs):  
     `python -m spot_bot.run_live --mode dryrun --symbol BTC/USDT --timeframe 1h --limit-total 2000 --db bot.db --cache data/latest.csv`
   - Confirm summary prints risk state, intent exposure, target exposure, and equity.
   - Note: If running as a script instead of module mode, set `PYTHONPATH=.`.

2. **Paper trading rehearsal**
   - Start with `--initial-usdt` or reuse balances loaded from `bot.db`.
   - Run paper mode (fills at close, fees applied):  
     `python -m spot_bot.run_live --mode paper --symbol BTC/USDT --timeframe 1h --limit-total 2000 --db bot.db --fee-rate 0.001 --max-exposure 0.3 --min-notional 10`
   - Verify `bot.db` contains `bars`, `features`, `decisions`, `intents`, `executions`, and `equity`.
   - Re-run with the same data; it should exit with “No new closed bar.”

3. **Safety parameters**
   - Spot sizing only: `max_exposure` clamp, `min_notional`, `fee_rate`, optional `slippage_bps`.
   - Live-only guards come from executor config: `max_notional_per_trade`, `max_trades_per_day`, `max_turnover_per_day`, `slippage_bps_limit`, `min_usdt_reserve`.
   - Keep DB on durable storage; WAL mode enabled by default.

4. **Monitoring**
   - Inspect the latest rows in `bars`, `features`, `decisions`, `intents`, `executions`, and `equity` after each run.
   - Track equity drift and risk_state timeline; paper mode should never create negative balances.
   - Use the single-line summary for quick checks (ts, price, S/C, psi, rv, risk, exposures, equity).

5. **Going live (explicit opt-in)**
   - Ensure API keys are configured for ccxt outside source control.
   - Run with acknowledgement flag:  
     `python -m spot_bot.run_live --mode live --i-understand-live-risk --db bot.db --symbol BTC/USDT --timeframe 1h --limit-total 2000`
   - Confirm ccxt connectivity; slippage/reserve guards should block abnormal fills.

6. **Stopping safely**
   - Stop the loop/process gracefully.
   - Run one final dryrun to record a flat target exposure if needed.
   - Archive the SQLite DB snapshot for audit.
