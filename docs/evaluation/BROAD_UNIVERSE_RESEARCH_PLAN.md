# Fixed broad spot-universe experiment — 2026-10-09

Written before computing new economic results. The goal is a material increase
in net profit, using more contemporaneously available assets, rather than more
variants on the same three prices. Owned spot funds only: no leverage, loans,
margin, shorts, derivatives, leveraged tokens or martingale. No live/default
changes, deployment or PR merge follow from this experiment.

## Cohort and source rules

Use the top 20 market-cap assets in the dated [2021-12-26 snapshot](https://coinmarketcap.com/historical/20211226/),
excluding USDT/USDC/BUSD stablecoins and wrapped BTC. Require a checksum-verified
December 2021 Binance spot-USDT archive, before any model result. CRO fails that
source rule (404); do not replace it with a later winner. The fixed 15 underlying
assets, in snapshot order, are BTC, ETH, BNB, SOL, ADA, XRP, old Terra, DOT, AVAX,
DOGE, SHIB, Polygon, UNI, LTC and LINK. This is a reconstructed historical cohort,
not a current-survivor list or the complete cryptocurrency market. New post-2021
entrants and other exchanges are outside the experiment.

Download official monthly spot 4h klines, January 2022–September 2026, verifying
each published SHA-256, timestamp units, ordered unique in-month grid, OHLCV,
quote/taker volumes and trade counts. Keep genuine absent candles absent; never
forward-fill a price, interpolate or turn a later quote into an earlier fill.

Identity matters: old LUNA/USDT ends on 2022-05-13 at 00:40 UTC. Exclude later
Terra 2.0 LUNA quotes, which are another token. Resume the same old underlying
as LUNC/USDT only from its documented 2022-09-09 08:00 listing; do not credit
Terra 2.0 airdrops. Keep Polygon quantities unchanged at the documented 1:1
MATIC/POL swap: MATIC until 2024-09-10 03:00, POL from 2024-09-13 10:00.
Exclude a listing candle whose label precedes actual listing time (POL 08:00).
No base-quantity or price rescaling is learned from future quotes.

Common event rule: after the completed-day close containing the published
2024-08-28 09:00 Polygon migration announcement, targets for Polygon are zero
until POL resumes and ordinary history/liquidity warmup is ready. Thus owned
positions can exit at a valid next-day price before the known trading pause.
For old Terra's sudden 2022-05-13 00:28 notice, there is no fictitious earlier
exit: the next daily decision is too late for its USDT halt.

In replay, missing-market rows have an explicit unquoted flag and zero sentinel
marks, never executable zero-price orders. Held unquoted inventory is valued at
zero conservatively until actual quotes resume; ownership and cash remain
unchanged. Count every such asset/bar. A candidate with any held unquoted asset
fails the data-quality gate, even if its apparent final profit is positive.
The zero mark is a pessimistic missing-quote assumption, not an observed price.
Only complete six-candle days enter daily indicators. Missing days reset required
continuous-history windows. Archive/schema failures stop the study, not remove
poorly performing assets from the cohort.

Source-validation amendment, before any new economic replay: the initial text's
SHA-256 was `2b089149a2ab9ea13c4c5a8e8504c1d435070448bac19836dc9fbd0bbbd3c0f8`.
The source probe found the documented final LUNA/MATIC candles close at the
actual halt time minus 1ms; accept only those two exact terminal timestamps.
It also found an identical repeated AVAX row on 2026-05-06 in both published
monthly and daily archives. An identical duplicate may collapse only after
SHA-256 verification of the separate official daily archive and exact agreement
of every field for the entire six-candle day. Preserve both raw source hashes,
published/canonical row counts and removed-row audit. This is corroboration by
another published artifact, not an independent market measurement. Conflicting,
uncorroborated, off-grid or out-of-order rows still stop the study. No strategy,
cohort, threshold or economic gate changes in this amendment.

Primary identity sources:
- [old LUNA spot halt](https://www.binance.com/en/support/announcement/detail/514e0ba636e843bda47d5d740b7dadf4)
- [old/new Terra distinction](https://www.binance.com/en/support/announcement/detail/d044a6742e484b77a170111460b0eed3)
- [LUNC/USDT listing](https://www.binance.com/en/support/announcement/detail/1e237033f63945d6b6acaf39f058d152)
- [MATIC/POL 1:1 migration and publication time](https://www.binance.com/en/support/announcement/detail/6a6de383727f4659a3050f7982e1620f)
- [archive schema](https://github.com/binance/binance-public-data).

## Eight fixed policies and known information

M(h) is the h-calendar-day completed close return; V60 is trailing population
standard deviation of 60 completed daily returns, floored at 0.0001. New-cohort
eligibility requires 90 continuous complete days, trailing 90-day mean quote
volume >=10 million USDT/day, and membership in the ten highest such liquidity
scores. All liquidity/rank inputs are completed-day values. Stable cohort order
resolves ties. Controls keep their original three-asset eligibility unchanged.

| Policy | Fixed rule |
|---|---|
| `spot_control` | Original BTC/ETH/BNB positive M(30), inverse V60, clip 25% without redistributing. |
| `capped_control` | Same three assets, redistribute inverse V60 within 50% aggregate/25% asset caps. |
| `broad_inverse_vol` | Liquid cohort, positive M(30), capped proportional inverse V60 across all eligible assets. |
| `broad_top3` | Select at most three eligible positive-M(30) assets by raw M(30), each target 1/6. |
| `broad_top3_vol` | Same, rank by M(30)/(V60*sqrt(30)), each target 1/6. |
| `broad_top5_vol` | Same risk-adjusted rank, at most five, each target 10%. |
| `broad_top5_consensus` | At least two positive M(14), M(30), M(60); rank by median standardized returns over those horizons, at most five, each target 10%. |
| `broad_top5_vol20` | Top-five risk-adjusted targets reduced, never increased, to a 20% annual ex-ante portfolio-volatility target. |

Rank membership is selected at completed Sunday closes, held through the week;
daily loss of eligibility exits, replacements wait for next Sunday. Vol20 uses
60-day completed-return covariance on the selected assets, with 25% diagonal
shrinkage, annualized by 365. Missing selected covariance means zero target.
No fitted forecast, later-period calibration, extra grid or post-result policy.

Publish targets with the last completed 4h candle and execute the next open.
One 1,000-USDT account, 50% aggregate/25% asset caps, Monday ordinary rebalance,
daily inactive/cap reductions, 1pt band and 10-USDT minimum fill. Sell before
buy; no debt/unowned sales. Positions can drift between decisions. Keep open
inventory marked at period ends and preserve peaks/compounding continuously.

## Freeze, economic tests and material improvement

First replay all eight continuous 2022–2024 nominal accounts. Among positive
profits with conservative 4h high/low drawdown >=-20% AND zero held-unquoted
bars, freeze the largest net return; otherwise cash. Persist protocol hash,
cohort/source identities and frozen choice before later economic replay. A
control cannot count as a new improvement; later leaders cannot be reselected.

Report each policy on fresh 2025 and Jan–Sep 2026 accounts, continuous Jan
2025–Sep 2026, and continuous full history. Nominal fills pay 0.1% fee +5bps
slippage +half a 2bps spread. Repeat all later paths with 0.2% fee +10bps
slippage, spread unchanged, and separately with one extra 4h target delay at
nominal costs (daily decisions 04:00 UTC). No policy is exempt from these checks.

Historical improvement requires a frozen NEW policy to beat both controls on
all four nominal periods, positive continuous later profit under cost/time
stress, full conservative DD within 20% in all three scenarios, independently
reconciled cash/inventory/NAV, and no held-unquoted bar on ANY path for that
policy. A material improvement additionally requires full nominal CAGR >=1.5
times capped control's CAGR AND continuous later nominal net return >=1.5 times
capped control's. Disclose modest improvements separately if they do not meet
that material bar. Historical DD is not a guaranteed maximum future loss.

For the frozen policy versus capped control on continuous later nominal data,
report a descriptive paired circular 14-day-block bootstrap of daily log-return
differences: 2,000 replicates, seed 23812, 99% percentile interval for annualized
mean log excess return. A positive lower bound is an additional uncertainty
gate. It is NOT confirmatory significance: periods were already inspected,
several rounds/strategies have been tried, and selection/universe uncertainty
is not captured by this interval. Subsequent unseen/paper evidence is needed.

Disclose every rejected policy, NAV/net/CAGR, close-only and conservative DD,
costs, turnover, fill frequency, held-unquoted counts, source/missing-bar audits,
rolling 7/30/90-day gains/doubling counts (overlapping, not future probabilities).
Reconcile every 4h close from signed cash flows/owned quantities. Reproduce
both controls against the previous report and test missing-market execution,
identity separation, publication clocks and future-prefix invariance.
