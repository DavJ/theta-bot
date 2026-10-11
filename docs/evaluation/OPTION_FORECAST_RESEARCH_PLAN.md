# Causal option purchase selection: fixed exploratory follow-up — 2026-10-12

User instruction: continue improving the bot. Freeze this file's SHA-256 before
computing the new variant payoffs. All source dates/outcomes have already been
studied; this is exploratory refinement, not a new holdout or confirmatory test.
The paid-long-option exception stays offline, optional, and disabled by default.

## Fixed funding, contracts and comparisons

- Reuse the original verified spot/option observations, contracts, monthly clock,
  actual trade proxies, expiry delivery prices, fees, lot floors and synthetic marks.
- Independently funded accounts: 1,000 and 10,000 USD-equivalent quote units.
- Initial allocation 90% spot / 10% isolated option cash. Purchase debit is capped
  by the smaller of 20% of current option cash and 2% of funded current account
  opening equity. No borrowing, written options or replenishment from spot.
- Controls: full-capital spot; idle option cash; the old call-only 1% budget policy;
  call-only 2%; momentum-direction 2%. Preserve archived economics exactly.
- Add warmup-only controls for call-only and momentum-direction, with the same
  252-completed-daily-close minimum as the forecast filters. This separates a
  forecast effect from simply excluding early trades.
- Six filters on the existing OTM signals: empirical expected edge >=10% and
  >=25%, each on call-only and momentum-direction; Gaussian-RV edge >=10%, each
  on call-only and momentum-direction. No contract/calendar/strike retuning.
- Missing, rejected or unaffordable contracts leave cash idle without redistribution.

## Two fixed causal payoff forecasts

- At execution use current spot open and only daily closes completed before the
  execution day's UTC midnight. Require at least 252 contiguous finite positive
  daily closes; retain at most 730. Do not use partial-day closes, future inventory,
  official future delivery prices or future option marks to accept a purchase.
- Forecast horizon H = ceiling(calendar days from execution to expiry). This
  integer-day approximation is explicit; execution/settlement still use actual 4h times.
- Empirical model: collect all completed overlapping H-day log returns inside
  the trailing close window. Apply those log returns to current spot open with
  equal weights; no signal-conditioned sample fitting or forward drift tuning.
- Gaussian-RV model: use sample volatility of the last 60 completed daily log
  returns. Forecast a zero-drift-in-price lognormal terminal distribution, with
  log mean -0.5 * daily_variance * H and variance daily_variance * H. Integrate
  with 128-point deterministic Gauss-Hermite quadrature, not random trials.
- For each terminal scenario calculate intrinsic and net delivery proceeds after
  scenario-specific delivery/conversion fees. Assume delivery-index/spot parity
  for this prediction only; actual backtest payments still use official delivery
  index and actual spot conversion prices.
- Compare expected net delivery receipt per underlying unit with the actual
  modeled buy debit (marked-up trade premium, tick rounding and all purchase costs).
  Expected edge = expected net receipt / buy debit - 1. It is a model estimate,
  not an independently calibrated probability or guaranteed expected profit.
- Record input cutoff, horizon, history/sample counts, forecast receipt, debit,
  estimated edge/win probability, decision and reason for every evaluated quote.

## Stresses, chronology and selection

- Evaluate six scenarios: nominal; cost stress; +4h delay; joint cost-plus-delay;
  forecast volatility scale 0.75; forecast volatility scale 1.25.
- Cost/delay assumptions remain the previous study's assumptions. Volatility-scale
  scenarios use nominal costs/time. For empirical returns retain their historical
  log-return mean and scale deviations around it; for Gaussian-RV scale sigma.
  These stresses change entry decisions and profits, unlike held-mark IV sensitivity.
- Development 2022–2024: positive whole-account profit and modeled conservative
  4h drawdown <=20% in all six scenarios. Freeze the policy with greatest minimum
  scenario net profit; tie by nominal profit then policy name. Controls may win;
  if no policy qualifies freeze cash. Persist before any new later-period payoffs.
- Report every policy on separately funded 2025, Jan–Sep2026, continuous 2025–Sep2026
  and continuous 2022–Sep2026 accounts under all six scenarios. Do not replace a
  frozen choice with a later winner or call reused periods independent validation.
- Recompute full nominal paths omitting each variant's three largest positive
  fills, explicitly an ex-post fragility diagnostic. Retain unspent cash and
  resize later entries. Also repeat half/double held-mark IV; expiry cash profit
  must not change because purchase forecasts are independent of those marks.
- Independently reconcile spot, option, funded account, monthly debit and forecast
  decision ledgers; verify default-off equivalence and unchanged archived controls.

## Limits on interpretation

Overlapping empirical windows are not independent observations. Both forecasting
models may be wrong after a regime change and omit index/venue basis, stablecoin
depegs and venue failure. Positive estimated edge is not established market alpha.
Historical trades are not executable asks or depth; drawdown uses synthetic marks.
No live execution, deployment, default change or PR merge is authorized.

Primary source specifications and pricing assumptions: [original protocol](LONG_OPTIONS_RESEARCH_PLAN.md).
