# Frozen multi-strategy protocol — 6 October 2026

David expanded the scope to established approaches and suggested a Kalman/EKF
fusion layer. The objective is maximum net trading profit subject to the same
1000 USDT capital, 30% target exposure, spot-only constraints and 15% development
drawdown screen. These constraints prevent different risk budgets from deciding
the comparison. This is a finite experiment, not a claim to cover every best
strategy in existence.

## Fixed candidates

1. Legacy dollar-price dual Kalman/theta.
2. Volatility-normalized log-price dual Kalman/theta.
3. The corrected theta with confidence power zero, as a confidence ablation.
4. 30-day time-series momentum; enter above a round-trip cost buffer and exit
   below its negative, using completed UTC daily closes.
5. EMA(20)/EMA(60) trend, with the same cost buffer.
6. Close-channel breakout: enter above the previous 20-day highest close, exit
   below the previous 10-day lowest close. This is an adaptation, not a full
   reproduction of the Turtle system's futures sizing and stop rules.
7. Range reversion: 20-day mean/std, enter below -2 sigma only with EMA gap
   between -1% and +3%, exit at the mean, at a 5% close-based stop, or outside
   that range regime. Stops are decisions at closed daily bars, not guaranteed
   intraday execution prices.
8. Equal-weight exposure combination of theta plus the four classical signals.
9. Regime-weighted combination: theta 20%, range 20%*(1-trend strength), and
   the three trend signals equally share 60%+20%*trend strength. Strength is
   the absolute EMA gap divided by 20-day daily volatility times sqrt(20),
   clipped to [0,1].
10. Arithmetic mean of causally calibrated next-day return forecasts.
11. Correlation-aware Kalman fusion of those same forecasts.

No hyperparameter search follows validation. Every candidate is reported,
including failed variants. Strategy-specific regime rules are part of the
candidate; classical signals do not inherit the theta gate. All candidates in
this expanded comparison allow exit below cost basis, so a risk-off signal can
actually reduce a losing position. Legacy strategies retain their guard by
default. The new strategies own their risk rules and allow loss exits by
default; no live strategy is activated here.

Structural implementation correction after the first development run: all
candidates now explicitly enforce the 30% exposure cap at each known decision
price, overriding turnover hysteresis and the profit guard for a necessary
reduction. The first run's validation results were discarded without choosing
or tuning a model from them. The complete suite is rerun. Market drift during
the execution candle and exchange minimum notional can still leave a residual
deviation. This corrects execution of the fixed risk constraint, not the signal
definitions or hyperparameters.

## Forecast calibration and fusion

The five source scores (theta, momentum, EMA trend, channel position, inverse
range z-score) share a next-day close-return horizon. Each receives an individual
rolling linear calibration on at most 252 completed daily observations, at least
126 required, slope ridge 0.01. At time t the newest training pair is score(t-1)
and realized return(t); no future return is available.

The latent state is expected next-day return. F=0.95, H is a column of ones.
Q=(0.05*daily volatility)^2. R uses at most 60 previously realized out-of-sample
forecast-error vectors, with 50% diagonal shrinkage and a numerical variance
floor. Initial R includes shared return noise; correlated predictions do not
count as independent votes. Joseph-form covariance keeps numerical positivity.
This is linear, so an EKF adds no necessary nonlinear measurement relationship.

The forecast mean and fused mean use identical entry/exit hysteresis around
round-trip cost/20, corresponding to a predeclared 20-day cost-amortization
assumption. This assumption does not guarantee any holding period or return.
All trading stays long/cash, with one net target and one execution path.

## Chronology

- Development January 2022–December 2024. Freeze the winner by development net
  P&L among candidates with positive development P&L and drawdown at most 15%,
  before comparing any expanded-suite validation result. Abstain to cash when
  no candidate meets those requirements.
- Historical validation January–December 2025 with a fresh cash portfolio and
  prior filter history. Online forecast calibration may update using outcomes
  already observed in that period; hyperparameters remain fixed.
- January–September 2026 is diagnostic replay, already partly examined.
- The earlier three-candidate scale experiment used 2025 first. Therefore this
  expanded session does not claim 2025 as a fresh confirmatory lockbox. Future
  paper observations are still needed for confirmation.
- Report all 2025 candidates, doubled fee/slippage stress, and independently
  reset quarterly returns for the frozen winner and Kalman fusion.

Costs: 0.1% per fill, 5 bp market slippage, 2 bp full spread. One declared missing
source hour, 2023-03-24 13:00 UTC, remains absent rather than synthesized. No order
is simulated during that hour. OHLC fill/timeout assumptions and terminal
mark-to-market remain limitations. No live mode is enabled by a historical pass.

## Primary sources and scope

- Moskowitz, Ooi, Pedersen (2012), [Time Series Momentum](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum).
- Liu, Tsyvinski (2018), [Risks and Returns of Cryptocurrency](https://www.nber.org/papers/w24877).
- John Bollinger, [official band rules](https://www.bollingerbands.com/bollinger-band-rules).
- Curtis Faith (2003), [Original Turtle Trading Rules, authored primary document](https://therobusttrader.com/wp-content/uploads/OriginalTurtle-Trading-Rules.pdf).
- Roger Labbe, [linear Kalman](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)
  and [EKF](https://filterpy.readthedocs.io/en/latest/kalman/ExtendedKalmanFilter.html) documentation.

These sources motivate families and filtering equations. The parameterized BTC
spot adaptations and calibration model above are our research choices, not
published promises of profitability or exact replications of those papers.
