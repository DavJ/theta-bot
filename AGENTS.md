# Theta Bot constraints

The user requires owned-funds **spot-only** execution. On 2026-10-10 the user
explicitly authorized offline profitability modeling of fully paid long options
as an optional, disabled-by-default research sleeve. This narrow exception does
not authorize live option trading or relax the remaining constraints.

- Do not introduce loans, leveraged/margin-funded positions, financial shorts,
  standalone futures, written options, leveraged tokens or martingale loss-doubling.
- The long-option research exception must use owned cash, premium/fee budgets,
  nonnegative positions, no replenishment from the spot sleeve, explicit data and
  execution-model limitations, and an off switch. Long options contain economic
  leverage; do not describe them as unleveraged or proven safe.
- Historical derivative studies are archived rejected work; do not promote them
  to active strategies. Mathematical derivatives in features are unrelated.
- Cash and asset inventory must remain nonnegative, including execution fees.
  Reject impossible fills and unverified live market/account information.
- The active historical drawdown selection budget is 20%; it is not a guaranteed
  future maximum loss. Report net results, costs and continuous-account drawdown.
- Freeze a research protocol and development choice before later comparisons.
  Previously studied data must not be called a fresh unseen holdout.
- Offline research and a passing backtest do not authorize live trading,
  deployment, changing live defaults or merging a PR.

See [the binding policy](docs/evaluation/SPOT_ONLY_POLICY.md) and the latest
[spot-flow protocol](docs/evaluation/SPOT_FLOW_RESEARCH_PLAN.md).
