# Prospective validation of fixed entry policies — 2026-10-25

The entry, fixed-risk and omission studies reuse January 2022–September 2026.
Their positive outcomes do not establish new-data profitability. This separate
cohort fixes all four policies before its future UTC midnight start. It preserves
the original October 16 forward protocol and every earlier frozen source file.

## Fixed accounts and data boundary

- `control_15`: original guarded edge at 15% requested annual volatility.
- `entry_15`: the fixed 0.64% entry hurdle at 15% requested volatility.
- `entry_175`: the same entry rule at 17.5%, diagnostic only.
- `entry_20`: the same entry rule at 20%, fixed candidate.

Do not replace the candidate or select assets after results. BTCUSDT, ETHUSDT
and BNBUSDT remain fixed. Each policy/scenario starts independently with 10,000
USDT and zero inventory at the future start; warmup gains/positions are excluded.
Sixteen continuous model accounts cover nominal, doubled fee/slippage, one extra
4h signal/covariance lag and combined stress. No invocation/year/window resets
cash, inventory or historical peak.

Retain 50% aggregate/25% asset target caps, Monday ordinary rebalancing, 1pp
band-edge execution, 10-USDT minimum and all cap/inactivity exits. Keep every-4h
actual-holdings risk reductions, 60 completed daily returns and 25% diagonal
covariance shrinkage. Entry eligibility follows the fixed research state machine;
nonpositive momentum resets eligibility. There is no leverage, borrowing,
financial short, option sleeve or loss-based sizing.

Use completed public spot `/api/v3/klines`, UTC 4h, with full raw 12-field arrays.
History begins on **2022-01-01**, as in the research, because eligibility memory
can persist longer than a short rolling warmup. That history supports features
only; scored returns begin at the newly frozen future start. Validate exact
aligned grids, OHLCV/taker quantities, source hashes and closed candles. Reject
missing/duplicate/unfinished data and any change to previously accepted prefixes.
Maximum observation staleness remains eight hours.

The bounded explicit collector uses at most 20 pages of 1,000 rows per symbol.
The connected Binance spot-kline tool can supply the same full arrays to the
validated import route, with transport recorded. No other market type or source
is silently substituted. Official interface references:
[spot klines](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)
and [public market-data documentation](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md).

## Publication, integrity and evidence

Freeze the settings, complete source fingerprints and runtime before the future
start. Publish the protocol and source to GitHub before that start. An explicit
publication receipt verifies a GitHub Actions run's server creation timestamp,
repository/workflow, source commit/tree, every pinned source blob and the exact
protocol blob. Record the public run/commit references and receipt hash. The
receipt is an inspected API record, not an unforgeable local clock or proof of
financial performance. Evaluation requires that bound publication receipt.

Each evaluation replays the full funded period. It validates the full prior
receipt sequence, all saved ledgers, the state pointer and source-prefix hashes.
Changing an anchor, rewinding time, resetting an existing account, orphaned runs,
modified receipts/ledgers or historical source edits fail closed. Replayed ledger
bytes must extend the previously accepted CSV prefixes. Commit a new state pointer
atomically only after complete immutable output and receipt writes under a lock.
An interrupted/orphaned run requires inspection; do not silently reset it.

Reconcile owned cash/inventory/NAV in every funded path. Independently audit each
saved CSV account against source-open fills, fees/impact, price/drawdown bounds,
lagged direct-NumPy covariance and its configured risk request. Before any closed
forward bar, results and economic gates are unavailable, not zero-profit evidence.

## Fixed preliminary screen and scope

Require **180 full days of closed forward bars**, not elapsed wall-clock time.
Require positive candidate net under all four scenarios, candidate drawdown bounds
within the historical 20% selection budget, candidate net above both 15% controls
under nominal/combined stress, and reconciled/independently audited accounts.
An early candidate drawdown breach rejects the screen. A preliminary pass never
sets live eligibility true. Preserve all diagnostics and failures.

This is prospective **model replay**, not observed exchange or executable paper
orders. Open fills, fixed fees/slippage and terminal marks remain hypotheses.
Timing stress delays signals/covariance, not emergency order execution. Actual
account/EEA access, commission tier, depth/queue, tax, operating costs, custody and
quote-FX/depeg risks are unverified. Annual-volatility requests and historical
drawdown screens do not cap future loss.

All commands are explicit. No scheduler, daemon, trading credentials, exchange
orders, live/default changes, deployment or merge are introduced.
