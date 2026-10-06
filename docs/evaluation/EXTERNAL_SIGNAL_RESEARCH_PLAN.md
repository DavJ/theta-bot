# Fixed external-signal / five-day forecast experiment

Recorded 2026-10-06 before model fitting or economic comparison. The account
drawdown research budget remains **20%**. All BTC/ETH/BNB prices from 2022 through
September 2026 and the earlier momentum choice are already known. New external
datasets are retrospective snapshots, not an independent market lockbox.

## Sources and availability

Use ECB EUR/USD and USD/JPY reference rates; FRED NASDAQ Composite, VIX, US
10-year yield and WTI oil; Alternative.me Fear & Greed; checksum-verified Binance
spot daily taker-buy share and actual settled perpetual funding for BTC/ETH/BNB.
The bot still trades spot only: funding is an explanatory signal, not income.

Record event time, modeled availability, snapshot retrieval time and hashes.
ECB availability is observation date +18 hours UTC. NASDAQ/VIX/yield snapshots
get a conservative two-US-federal-business-day lag; WTI gets ten calendar days because daily labels
are not publication timestamps. Fear & Greed gets one full day. Spot flow is
available at the following midnight, and daily funding aggregates only after
the latest included actual settlement plus five minutes. Use backward as-of
joins; never backward-fill unavailable features. Stale source values expire.
FRED/ECB/Alternative snapshots lack historical first-release vintages: these
lags do not eliminate possible revisions or prove historical availability.
Mark that restriction in every interpretation. Do not substitute weekly FRED
H.10 observation dates for immediately tradable forex closes.

Pre-fit availability correction: inspecting the official DGS10 page showed
Friday 2026-10-02 observations published on Monday 2026-10-05. The original
two-calendar-day draft would admit a Friday observation prematurely on Sunday.
It was corrected to two US federal business days before any model was fitted;
holidays are excluded using the USFederalHolidayCalendar.

Pre-fit join correction: exchange observations can occur on federal holidays;
the modeled business-day availability can therefore place several different
observation dates in one release batch. Preserve unique source/event grain,
but permit shared availability. At a tied availability choose the latest event
deterministically, after computing source features on all observations.

Fear & Greed attribution must sit beside its displayed values/results. It is a
composite partly built from price/volume/volatility, not independent NLP news.
Long historical NLP sentiment is not claimed without an actual timestamped
corpus. GDELT's recent DOC history is insufficient for 2022-2026 model training.
Open-interest, ETF/on-chain flows and news first-seen records remain additional
research candidates, not proven useful signals.

## Nine fixed candidates

1. `control_momentum`: unchanged 30-day positive momentum/inverse volatility,
   50% aggregate /25% individual cap, no new model or risk scaler.
2. `ridge_price`: regularized five-day return forecast from crypto price,
   range, volume, market-relative momentum and the existing normalized dual
   Kalman/theta regime features.
3. `ridge_macro`: same algorithm plus forex, equity, volatility, rate and oil.
4. `ridge_sentiment`: same algorithm plus Fear & Greed level/change.
5. `ridge_crypto`: same algorithm plus settled funding and taker-buy share.
6. `ridge_all`: all three external groups.
7. `boost_price`: shallow gradient boosting on price/regime features.
8. `boost_all`: the same boosting plus all external groups.
9. `blend_all`: equal average of momentum, ridge_all and boost_all allocations.

Ridge alpha=10; scaler fitted only to training data. HistGradientBoosting uses
100 iterations, learning rate=.05, max depth=2, maximum seven leaves, minimum
30 samples per leaf, L2=10, no early-stopping split, fixed random seed 20261006.
Refit every 20 calendar observations with at most 504 and at least 126 completed
training outcomes per asset. Target = close[t+5]/close[t]-1; rows train only
when their fifth subsequent daily close is already complete. Training outcomes
are clipped to ±3 training standard deviations; no global clipping/scaling.
No hyperparameter search or additional variants after results are inspected.

Pre-fit amendment on 2026-10-06, following the user's variable-delay/jitter
question: after publication-time alignment, every external base feature is
represented by three nonoverlapping backward calendar-day means: lag 0-2,
3-9 and 10-20. A bucket requires all of its observations; no forward value is
used. This is a coarse distributed-lag response, separate from publication
latency. Refit coefficients/boost interactions can change the lag response
over time, but do not identify an exact causal delay for each event.

Both algorithms weight completed training outcomes exponentially by age, with
a fixed 126-calendar-day half-life. Normalize weights to mean one to preserve
the Ridge regularization scale. Fit both StandardScaler and Ridge with those
weights; boosting also receives sample weights. Keep the existing unweighted
training-only target clipping rule. No additional Kalman/EKF weight filter is
used in this experiment. This amendment precedes all real-data model fitting.

Price inputs use returns at 1/5/20/30/60 days, 20/60-day volatility, daily range,
volume relative to its 20-day mean, BTC-relative momentum, and the existing
dual Kalman source using causal 20-day RV and 60-day phase windows. External
market returns use 5/20 observed market sessions; rates use level and five-session
change; Fear & Greed uses level and five-observation change; funding and flow
use current/5/20-day summaries, followed by the three lag buckets above.
Required missing or stale features keep a model
in cash; they are not silently filled with future observations.

Forecast conviction = clip((predicted five-day return - .0032) /
(0.5 * trailing daily volatility * sqrt(5)), 0, 1). Multiply by normalized
inverse-volatility asset weights. Model allocations have a 100% aggregate,
50% individual target cap and a fixed 20% annual full-covariance volatility
target (60 days, 25% diagonal shrinkage). No leverage. The blend uses these same
caps/risk rule. Monday ordinary rebalances, daily inactive/risk reductions,
1% NAV band, 10 USDT minimum and core market-fill accounting stay unchanged.
The fixed cost hurdle is not increased/reoptimized in the doubled-cost replay.

## Selection, incremental value and checks

Freeze maximum positive net-return candidate from a continuous 2022-2024 account
whose conservative intraday drawdown bound is <=20%. Save the selection before
later-period economic evaluation. Then report all candidates on independently
restarted 2025 and Jan-Sep 2026 accounts, and the continuous 2022-Sep 2026 account.
Repeat all three at double fees/slippage: base fee=.001, slippage=5 bps, full
spread=2 bps; stress fee=.002, slippage=10 bps, spread unchanged. Screen the
frozen candidate only, never reselect on newer-period profit.

Compare each feature addition to ridge_price and boost_all to boost_price on
the same available dates: five-day forecast MSE, Spearman information coefficient,
and incremental paired squared-error differences. A 20-day moving-block
bootstrap of daily asset-averaged loss preserves cross-asset and overlapping
target dependence (1,000 replications, fixed seed). Report uncertainty; never
call a source significant just because contemporaneous prices correlate.
Apply a Bonferroni-adjusted threshold across the five feature-addition tests.
Positive forecasts or predictive significance alone do not establish net profit.

For an external-source frozen candidate, repeat forecasts and the later nominal
economic replays with all external modeled availability delayed another three
days. This tests sensitivity to availability assumptions, not historical vintages.
Do not imply that this removes source revision risk.

Check raw source grain, duplicates, nulls, bounds, archive checksums and overlap
with existing OHLCV; save profiles/provenance. Test future-price/label/source
prefix invariance, publication delays, staleness, malformed joins, training-only
scaling and matured outcomes. Reconcile every replay's actual cash, inventory,
NAV and cost totals. No live deployment or exchange orders.

Sources: [ECB](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html),
[FRED](https://fred.stlouisfed.org/),
[Alternative.me Fear & Greed](https://alternative.me/crypto/fear-and-greed-index/),
[Binance public data](https://github.com/binance/binance-public-data).
