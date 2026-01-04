# Spot Bot 2.0 Go-Live Checklist

1. **Dry run sanity**
   - Install deps: `pip install -r requirements.txt`
   - Update `spot_bot/config.yaml` with symbol/timeframe limits.
   - Run once in dryrun mode:  
     `python spot_bot/run_live.py --mode dryrun --config spot_bot/config.yaml --cache data/latest.csv --db bot.db`
   - Confirm single-line summary prints risk state, target exposure, and equity.

2. **Paper trading rehearsal**
   - Start with fresh balances in `config.yaml` (initial_equity) or carry forward from `bot.db`.
   - Run paper mode:  
     `python spot_bot/run_live.py --mode paper --config spot_bot/config.yaml --cache data/latest.csv --db bot.db`
   - Verify `bot.db` has `bars`, `features`, `decisions`, `intents`, `executions`, and `equity` entries.
   - Inspect balances from the summary; ensure exposure < `max_exposure`.

3. **Safety parameters**
   - `max_exposure`, `min_notional`, `fee_rate` in `config.yaml`.
   - Live-only guards: `max_notional_per_trade`, `max_trades_per_day`, `max_turnover_per_day`, `slippage_bps_limit`, `min_usdt_reserve`.
   - Adjust `state-file`/`--cache` to avoid duplicate bar processing.

4. **Monitoring**
   - Watch latest row in `bot.db` tables for anomalies.
   - Track equity drift, executed qty, and risk_state timeline.
   - Keep an eye on log output; dryrun/paper should never raise.

5. **Going live (explicit opt-in)**
   - Ensure API keys are configured for ccxt outside of source control.
   - Run with acknowledgement:  
     `python spot_bot/run_live.py --mode live --i-understand-live-risk --config spot_bot/config.yaml --db bot.db`
   - Confirm ccxt connectivity and that slippage/reserve guards block abnormal fills.

6. **Stopping safely**
   - Stop the loop/process.
   - Run one final dryrun to record a flat target exposure.
   - Archive the SQLite DB snapshot for audit.
