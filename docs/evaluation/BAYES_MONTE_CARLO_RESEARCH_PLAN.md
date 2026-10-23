# Fixed Bayesian / Monte Carlo signal experiment — 2026-10-12

Freeze this protocol before computing any new candidate's trading payoff.
This is offline, owned-cash spot research under [the binding policy](SPOT_ONLY_POLICY.md).
The optional long-option experiment is not changed in this round.

## Inputs, target and clock

Use the already verified BTCUSDT, ETHUSDT and BNBUSDT spot 4h archive, January
2022–September 2026. Reject gaps or changed SHA-256 checksums. Derive complete
daily bars from these data. Features at a day labelled d become available at
d+1 UTC midnight: publish with the completed 20:00 4h bar, execute at the next
open. Stress adds one further 4h bar. No outcome ending after that daily close
may enter training, analogue selection, standardization or fusion weights.

Both new experts estimate the same **five-calendar-day log return**, not an
option payout or an incompatible mixture of daily and monthly horizons. Use
the existing causal price/theta feature function and this fixed nine-column
subset: r1, r5, r20, r60, vol20, relative_volume, market_r5,
phase_concentration, theta. No external data or information is added by merely
changing the model. All price/volume features use the completed day.

## Bayesian expert

Training window: at most 504 daily feature/outcome pairs, at least 252 valid
completed pairs. Standardize using this training window alone, with scale
floor 1e-8. Fit only every fifth eligible chronological row, counting backwards
from the latest completed row, to avoid overlapping five-day outcome blocks.
Missing feature rows are excluded; selected actual row locations must remain
at least five days apart. Refit each completed day.

Conjugate normal / inverse-gamma linear regression with intercept, zero prior
coefficient mean, prior precision 1 on intercept and 20 on each standardized
slope, inverse-gamma shape 3 and scale .01. Predictive distribution is Student t;
retain predictive variance and the separate uncertainty of the latent mean.
Use expected **log** return and probability of exceeding the log cost hurdle.
Do not exponentiate a Student-t mean into an expected cash payoff. Conditional
Gaussian noise, a fixed linear relationship and the prior are modeling
assumptions, not an assurance of calibrated probabilities or independent data.

Primary methodological references:
https://arxiv.org/abs/2406.01819 and
https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf .

## Conditional historical Monte Carlo expert

On the same completed training pairs and standardization, take the 64 closest
historical feature states by Euclidean distance; resolve ties chronologically.
Each has an already completed, contiguous five-day log-return block. Draw 4,096
whole blocks uniformly with replacement. Its fitted stochastic model is a
conditional historical distribution; Monte Carlo is numerical sampling from
that model, not independent market information. Whole blocks preserve dependence
inside the observed five-day path, but do not model future regime changes or
joint cross-asset paths. Analogues can overlap and are not independent samples.

Primary seed 20261012; sensitivity seeds 20261013 and 20261014. Derive a separate
generator from seed, asset identity and decision date, so a longer input prefix,
asset order or an extra calculation does not change an earlier simulation.
Archive sample mean/variance/probability, quantiles, exact analogue moments,
numerical Monte Carlo standard errors and analogue count. Those numerical errors
shrink with draws; epistemic uncertainty and historical sample size do not.

Block-resampling context, not a theorem proving this conditional analogue model:
https://mathweb.ucsd.edu/~politis/impactBOOT.pdf .

## Fusion and eight fixed policies

Fuse only the two five-day log forecasts. Record errors after their actual
five-day outcomes mature. Use at most 60 **nonoverlapping**, fixed-calendar
forecast error pairs, with at least 20 before learning weights. Minimize the
uncentered error second moment with 50% diagonal shrinkage and ridge 1e-10,
subject to nonnegative weights summing to one; use equal weights during warmup.
The fused predictive variance is no smaller than either the weighted empirical
error second moment or the smaller individual predictive variance. Duplicating
a forecast must not create artificial predictive precision. The fused gain
probability uses a Gaussian approximation and is explicitly uncalibrated.

Keep the existing `capped_allocation` daily momentum target, Monday rebalances,
daily inactivity/cap exits, 50% aggregate and 25% per-asset target caps, 1% band,
10-USDT minimum fills and one shared cash account. Released weight remains cash;
no new model may increase the original target or redistribute rejected weight.

- `spot_control`: exactly the archived current spot target and execution clock.
- `ready_control`: same target, but wait for both experts' valid forecasts.
- `bayes_buy_gate`, `mc_buy_gate`, `fused_buy_gate`: the ready target; permit only
  buys/increases when the named expected log return exceeds the hurdle AND its
  estimated probability of a five-day gain after the hurdle is at least .55.
  A rejected buy does not itself liquidate held inventory. Existing base exits
  and required cap reductions remain executable.
- `bayes_blend`, `mc_blend`, `fused_blend`: ready target multiplied by
  .5 + .5*c, where c=clip((P(gain after hurdle)-.5)/.1, 0, 1), and c=0 if mean
  log return does not exceed the hurdle. This keeps a 50% share of the original
  target even on a pessimistic forecast. Monday sizing and base exits persist.

The hurdle is log1p of the core's first-order round-trip fee/impact cost; use the
scenario's own fees. It is a signal heuristic, not a guarantee that an actual
five-day round trip will occur. The unchanged core charges each actual fill.
Disabled integration returns the exact original target without inspecting
forecast inputs. No new live strategy choice or default is enabled.

## Frozen selection and evaluation

Independently funded 1,000 and 10,000 USDT accounts. Development 2022–2024 in
nominal, doubled fee/slippage, one-bar delay and joint cost/delay scenarios.
Nominal: .1% fee, 5bps slippage, half of a 2bps spread per fill. Stress: .2% fee,
10bps slippage and unchanged spread. Reconcile every close independently from
cash flows/inventory. Require nonnegative cash and inventory.

Eligible candidates must have positive net development P&L and conservative
4h high/low drawdown no worse than 20% in all four scenarios. Select the highest
worst-scenario development return, then nominal return, then name. Controls may
win. Otherwise choose cash. Write that choice before later economic replays.

Report all eight policies for independent 2025 and January–September 2026
accounts, continuous 2025–September 2026 and full continuous 2022–September 2026,
in all four execution scenarios. Repeat six new model policies on the two
continuous periods at nominal costs with both alternative MC seeds. Never
reselect a winner or a seed using these later results. Check every old spot
control field against the archived bounded-option study on identical periods.

Archive full nominal equity/trades, daily forecasts, decision inputs, model
weights and frozen selections. Report net profit, CAGR, conservative drawdown,
fees, turnover, fills, exposure and 7/30/90-day rolling gains/doublings. Compare
forecast log-return MSE and gain-probability Brier loss against zero and the
same-window unconditional historical forecast on identical matured rows.
Scores on overlapping evaluation horizons are descriptive, not p-values.

A historical economic improvement requires the frozen winner to beat BOTH
spot and ready-only controls on nominal and joint-stress later continuous
accounts, stay within 20% on all full paths and both seed sensitivities, and
retain positive later/full P&L in every scenario. Separately report whether
forecast skill improves over unconditional history. The entire date range was
previously studied; it is not a new unseen holdout. The 20% historical screen
is not a future loss guarantee; fixed cost assumptions are not a verified live
fee tier or liquidity quote. Passing does not authorize deployment or merging.
No extra parameter search or post-result candidate addition in this round.
