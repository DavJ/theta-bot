# Theta Bot constraints

The user requires owned-funds **spot-only** trading. This constraint applies to
new research, executable strategies, configuration and recommendations.

- Do not introduce leverage, loans, margin accounts, financial shorts, futures,
  options, leveraged tokens or martingale loss-doubling.
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
