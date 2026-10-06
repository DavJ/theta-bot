# Frozen research plan — 6 October 2026

The April–September 2026 result has already been inspected. It cannot be an
untouched holdout for subsequent changes. No returns from the new 2025 validation
period were inspected before this plan and candidate definitions were written.

## Question

Does the dollar-denominated dual Kalman filter suppress exposure because its
fixed innovation covariance is in incompatible price units? Does correcting
that measurement actually improve returns after transaction costs?

## Predeclared candidates

1. `legacy_dual`: existing dollar-price filter, confidence power 1.
2. `log_vol_dual`: log-price filter, innovation covariance estimated from the
   preceding 120 hourly log returns, confidence power 1.
3. `log_vol_no_conf`: the same corrected filter with confidence power 0. This is
   an ablation of the confidence rule, not permission to increase leverage.

For the corrected filter, process-noise/measurement-noise ratios retain the
legacy defaults (0.1 for level, 0.001 for slope). Volatility has a 1 bp floor.
No hyperparameter grid or further candidate will be added after validation.

## Chronology and selection

- Development: January 2022–December 2024, spanning falling and rising markets.
- Validation: January–December 2025, fresh portfolio, prior feature/filter warmup.
- Diagnostic replay: January–September 2026, explicitly already partly seen.
- Freeze the best development net-P&L candidate before running validation.
- Abstain to cash if its development return is nonpositive or its drawdown
  exceeds 15%. Report rejected candidates as well; no holdout-driven replacement.
- Evaluate every predeclared candidate in validation and with doubled fees and
  slippage for transparent comparison. These comparisons do not reselect it.
- Split the selected validation into three independently reset subperiods.

## Constant assumptions

Data-handling amendment before any validation returns were calculated: older
official archives can contain hourly gaps. Strict download remains the default.
Research may explicitly preserve and report up to 24 missing timestamps across
the whole experiment, provided each is identified in the verified manifest.
No candle is synthesized or filled. Rolling windows count observed candles;
trading during missing bars is not simulated. This limits execution fidelity.

BTCUSDT spot, 1h candles, 1000 USDT starting capital, 30% target-exposure cap,
0.1% fee per fill, 5 bp market slippage, 2 bp full spread. Existing causal theta
regime gate, hysteresis, limit-fill assumptions and sell guard stay fixed.
No exchange order is sent. Provenance comes from checksum-verified official
Binance monthly archives: https://github.com/binance/binance-public-data.

## Interpretation

Report realized mean exposure, fees, execution costs, turnover, net P&L and
drawdown. Compare capital returns to cash and to 30% initial-allocation hold.
Also show a hindsight mean-exposure benchmark, labelled as descriptive rather
than investable or fully risk matched. A larger result caused by more exposure
does not establish better alpha. Historical screening cannot enable live mode.
The archive and OHLC fill model do not measure order-book queues or live latency.
