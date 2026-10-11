# Bounded paid-option purchases: fixed follow-up — 2026-10-12

User instruction: continue improving the bot. Owned-cash long-option offline
research remains optional and disabled by default; no live orders or deployment.
The previous model's profit concentration and growing purchase budgets motivate
this follow-up. All dates and previous outcomes have already been studied.
This is exploratory refinement, not independent confirmation or an unseen holdout.

Freeze this file's SHA-256 before calculating new variant payoffs. Preserve the
original LONG_OPTIONS_RESEARCH_PLAN.md and its observation/source hashes.

## Fixed variants and funding

- Independently funded accounts: 1,000 and 10,000 USD-equivalent quote units.
- Same completed-day capped spot policy, contract schedule/selection, expiry,
  trade-price proxies, costs, minimum lots and synthetic marks as the old study.
- Controls: full-capital spot; 90% spot / 10% idle cash; old inverse call-only OTM;
  old inverse momentum-direction OTM. No new option contracts or signal tuning.
- Six new variants: cross call-only OTM and momentum-direction OTM with each of
  the following maximum aggregate monthly purchase debits (premium plus all fees):
  2% of current funded account opening equity; 1% of current opening equity;
  2% of initial total account capital. All also retain the old 20% limit on current
  option cash. Thus the smaller of the two budgets applies; lots round down.
- Opening equity uses previous spot holdings/cash at the current spot open plus
  current option cash. There are no open options at monthly purchases. Do not use
  the current bar's close, later inventory, future profit or synthetic option marks
  to size entries. No transfers from spot, borrowing or loss-doubling.
- Unaffordable or missing selected contracts are skipped without redistribution.
  Unspent option cash and gains remain cash. An entry-debit cap is not a cap on
  held mark-to-market exposure or a guaranteed future total-account drawdown.

## Chronology and selection

- Development: 2022–2024. Evaluate nominal, cost stress, +4h delay and joint
  cost-plus-delay stress before freezing a choice for each capital size.
- Eligible development candidates must have positive whole-account profit and
  modeled conservative 4h drawdown <=20% in all four scenarios.
- Among eligible candidates maximize the minimum scenario net profit; tie by
  nominal net profit, then deterministic policy name. Controls can win. If none
  qualifies, freeze cash. Persist choices before any new later-period payoffs.
- Report every variant on separately funded 2025, Jan–Sep2026, continuous
  2025–Sep2026 and continuous 2022–Sep2026 accounts, under all four scenarios.
- Report post-development failures rather than choosing a replacement using
  later data. Compare prior controls' shared economic fields with archived results.

## Robustness diagnostics, not predictive entry filters

- For each full-period nominal option variant independently identify its three
  largest positive closed-option net profits. Recompute the whole path with those
  entry events unfilled, retaining the unspent cash and resizing all later buys.
  This is an explicitly ex-post missed-fill stress, not a causal signal or a
  claimed probability of execution failure. Record the removed instruments/events.
- Repeat nominal held marks with half/double entry IV. Terminal expiry cash profit
  must remain unchanged. Short rolling gains/drawdown remain model-dependent.
- Report profit concentration, number of buys/winners, quote/lot skips, all cost
  contributions, largest monthly debit/account fraction and the loss of top fills.
- Independently reconcile every option/spot path, funded account contributions and
  monthly debit ledgers. Preserve exact default-off spot behavior. All cash and
  positions must remain nonnegative. No claim of executable ask/depth verification.

## Interpretation

Historical trades plus markups still do not establish executable purchase prices.
Held prices and drawdowns remain synthetic. Reused history, uniform fee scenarios,
coin/venue basis, stablecoin parity and unmodeled venue failure remain limitations.
A positive refinement is only a research candidate, not authorization for live
trading or proof that a 20% future loss bound can be enforced.

Primary specifications and sources: [the original protocol](LONG_OPTIONS_RESEARCH_PLAN.md).
