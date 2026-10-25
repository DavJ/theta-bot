# Theta Bot — spot trading research

Python research software for cryptocurrency spot trading. The project contains
Jacobi theta and Mellin features, Kalman and forecast-fusion models, and daily
momentum portfolio experiments with shared-cash execution accounting.

**A profitable production strategy is not established.** Offline replay,
chronological evaluation and execution guards are implemented. The positive
historical benchmark uses 30-day momentum and inverse 60-day volatility;
theta-derived forecasts and additional signal sources remain research modules.

## Current results

Latest entry-hysteresis/risk-utilization studies and public-depth execution calibration: **2026-10-25**; fixed-allocation benchmark audit: **2026-10-23**; Möbius/theta diagnostic: **2026-10-22**; prospective-validation tooling: **2026-10-16**; execution/risk study: **2026-10-13**; signal models: **2026-10-12**;
optional long-option models: **2026-10-10 / 2026-10-12**. These are historical simulations,
with previously studied periods, rather than newly unseen confirmation data.

The [fixed entry-hysteresis study](docs/evaluation/THETABOT_ENTRY_HYSTERESIS_2026-10-25.md)
requires completed 30-day momentum above **0.64%** to enter eligibility and keeps
eligibility until momentum is nonpositive. This is a trailing-strength heuristic,
not an estimated future net edge. With the same 15% requested volatility guard,
10,000-USDT full nominal fees/impact fall from **857.55 to 832.25 USDT** and fills
from **1,321 to 1,299**. Net improves modestly; both account sizes pass the fixed
profit and turnover screens. The other four variants and all outcomes remain
published; none is substituted after results.

The subsequent [fixed risk-utilization comparison](docs/evaluation/THETABOT_RISK_UTILIZATION_2026-10-25.md)
changes only the entry policy's constant volatility request. Results below use
10,000 initial USDT and continuous January 2022–September 2026 accounts. Net is
after modeled trading costs, before operating costs/tax; inventory is marked at
the end. Joint stress doubles fees/slippage and adds one 4h execution-signal lag.
DD is the worst full conservative drawdown across all four scenarios.

| Policy / annual-volatility request | Full net | Net under joint stress | Nominal CAGR | Worst full DD |
|---|---:|---:|---:|---:|
| Original guarded edge / 15% | +83.37% | +70.47% | 13.62% | -15.03% |
| Entry hysteresis / 15% | +85.19% | +71.42% | 13.86% | -15.03% |
| Entry hysteresis / 17.5% (diagnostic) | +96.10% | +79.07% | 15.24% | -17.09% |
| Entry hysteresis / 20% (fixed candidate) | +98.13% | +78.28% | 15.49% | -18.56% |

The fixed 20% candidate passes the descriptive screen at both capitals, but the
17.5% diagnostic earns more under combined stress with a smaller drawdown. There
is no uniformly superior risk request. At 20%, nominal realized annual daily
volatility rises from **11.63% to 14.28%** versus entry-15, mean exposure from
**17.20% to 20.95%**, and fees/impact from **832.25 to 989.07 USDT**. The larger
profit mainly reflects more risk utilization, not a new forecast. Neither the
volatility request nor the historical 20% selection budget caps future losses.

Both new protocols were published before their respective payoffs; the second
study is explicitly an adaptive follow-up on known history. Together they
reconcile **216 account paths**, reproduce **72 prior control summaries**, and
independently audit **18 full CSV accounts**. The entry archive records the
reproduction of lost transient outputs from the exact published source/runtime
and matching archives, with a fresh selection timestamp. Existing prospective
and public-depth protocols retain their source fingerprints. These results add
research candidates; production profitability remains unestablished.

The [fixed calendar/asset-omission audit](docs/evaluation/THETABOT_ROBUSTNESS_2026-10-25.md)
keeps all four policies and replays 10,000-USDT accounts without each asset in
turn, with the existing capped budget redistributed to eligible retained assets.
The fixed entry-20 candidate remains profitable in all later/full omission
accounts under nominal and combined stress, and every omission path stays within
the historical 20% drawdown budget. Full nominal/joint returns are **79.88%/61.63%**
without BTC, **80.52%/67.22%** without ETH and **78.61%/62.09%** without BNB,
versus **98.13%/78.28%** with all three. No subset replaces the fixed universe.

Actual nominal net contributions in the original entry-20 account are
**3,044.37 USDT BTC, 3,308.10 ETH and 3,460.07 BNB**. These sum to the shared
account's **9,812.54-USDT** profit and include each asset's fees/impact; they are
not stand-alone strategy returns. The full omission replays account for different
cash allocation and risk scaling. This diagnostic supports that the known-history
profit is not supplied by only one of these three surviving assets.

Calendar results still show losses: entry-20 has **6/46 negative rolling
12-month windows**, worst **-5.98%**, and a **-10.11% worst month** nominally;
combined stress has **7/46 negative windows**, worst **-4.90%**. The windows
overlap, so these fractions are not future probabilities. Entry-15 has four
negative nominal windows with worst **-4.65%**. Excluding the best three months
mathematically leaves entry-20 at **+40.04% nominal / +26.48% combined stress**;
this is ex-post concentration arithmetic, not an executable timing strategy.
The study reconciles **96 paths**, reproduces **24 preceding controls** and
independently audits **14 full accounts**. It adds no new signal, parameter
selection or unseen evidence, and does not establish dependable monthly income.
The full-history accounts begin in cash and require 60 completed daily returns
before sizing; entry-20 first buys on **2022-03-21**. The 2022 result therefore
includes an initial cash/warmup phase. Later accounts can use preceding price
history but start with their own cash and no inherited positions.

The [public-depth calibration](docs/evaluation/THETABOT_DEPTH_PAPER_2026-10-23.md)
adds conditional paper execution across visible bid/ask levels, exact quantity
steps, owned funding and quote-fee accounting. Its protocol was published before
reading the six BTC/ETH/BNB USDT/USDC books. The first capture supports **48
independently funded static round-trip probes and 96 audited legs**. With a
hypothetical 0.1% quote fee per leg, 2,500-quote nominal round-trip costs range
from **0.1998% to 0.2059%**; modeled stress is approximately **0.499–0.505%**.
Actual account commissions remain unknown. These probes measure conditional
execution cost, not strategy profit or actual exchange fills. REST depth has
no source book timestamp. One batch fails the fixed 200-book/30-day coverage
gate, and historical cost assumptions remain unchanged. The explicit collector
uses two public GET endpoints; no background collection is started. The
[fixed plan](docs/evaluation/DEPTH_PAPER_EXECUTION_PLAN.md) documents the
unverified reference-price, account and execution-rule constraints.

The [2026-10-25 continuation](docs/evaluation/THETABOT_DEPTH_PAPER_2026-10-25T020053Z_51206499ac4f.md)
retains both captures in one verified archive: **96 conditional round trips and
192 audited legs**, with two distinct books per market spanning approximately
**29.4 hours**. Latest 2,500-quote nominal costs are approximately
**0.1998–0.2011%** under the same frozen model. Coverage remains sparse and the
200-book/30-day gate is unmet. The new publication wrapper uses UTC observation
times and capture hashes in filenames, rejects conflicting existing files, and
normalizes ZIP metadata for repeatable evidence. Original source/protocol and
the first published capture remain unchanged. Explicit continuation:

```bash
python -m scripts.write_depth_paper_report \
  --protocol docs/evaluation/THETABOT_DEPTH_PROTOCOL_2026-10-23.json \
  --out artifacts/depth_paper --collect
```

Restore the latest verified evidence archive's captures, reports and protocol
before continuing an existing study; follow the report's reproduction notes.

The new [fixed-allocation audit](docs/evaluation/THETABOT_ALLOCATION_BENCHMARKS_2026-10-23.md)
compares the unchanged guarded momentum policy with cash, 25%/50% equal targets
and capped inverse-volatility targets without momentum. All share execution
and requested risk settings; references rebalance weekly and sell for risk.
They are not literal buy-and-hold or exactly matched realized-risk portfolios.
For 10,000 USDT over January 2022–September 2026, guarded momentum earns
**+83.37% nominal / +70.47% combined stress**, versus **+31.84% / +32.75%**
for equal-25, **+50.39% / +49.75%** for equal-50 and **+50.90% / +50.09%**
for inverse-volatility. Worst full drawdowns are **15.03%, 19.83%, 28.30% and
28.68%**, respectively; nominal realized annual daily volatility is
**11.65%, 12.28%, 16.40% and 16.63%**. The fixed incremental-value screen passes
at both 1,000 and 10,000 USDT. This supports the existing momentum rule on known
history, without establishing future alpha or production eligibility.

The 10,000-USDT full-history terminal surplus over the invested references
corresponds to **35.78–66.20 USDT per calendar month under combined stress**.
This is descriptive headroom for *additional* operating costs relative to each
reference, not a monthly withdrawal replay or income forecast. Both policies
can incur operating costs. Full provenance, 120 reconciled paths, 24 reproduced
old controls and ten independently audited CSV accounts are retained with the
[frozen plan](docs/evaluation/ALLOCATION_BENCHMARK_RESEARCH_PLAN.md).

The earlier signal-comparison table below starts each account with 1,000 USDT.
Full history is one continuous account from
January 2022 through September 2026; the later comparison is a separate continuous
account from January 2025 through September 2026. Net results include modeled
0.1% fees per fill, 5bps slippage and half a 2bps spread. Cost stress doubles fees
and slippage; timing stress adds one 4h signal delay. Terminal inventory is marked
to market. CAGR is the annualized compounded return over full history.

| Research policy | Full net | Full CAGR | Later net | Full DD: nominal | Full DD: doubled costs | Full DD: +4h delay |
|---|---:|---:|---:|---:|---:|---:|
| Original BTC/ETH/BNB momentum (`spot_control`) | +103.91% | +16.19% | +31.75% | -18.54% | -19.24% | -20.20% |
| Capped allocation (`capped_control`) | +110.26% | +16.95% | +35.45% | -18.62% | -19.32% | -20.37% |
| Ranked 30-day momentum (`ranked_m30`) | +115.49% | +17.55% | +40.85% | -20.81% | -21.40% | -20.15% |
| 14/30/60-day consensus (`consensus`) | +85.03% | +13.84% | +27.96% | -19.32% | -19.96% | -17.91% |
| Latest development-frozen broad policy (`broad_top5_vol20`) | +54.54% | +9.60% | -0.92% | -22.01% | -23.13% | -22.99% |

DD is the conservative 4h high/low portfolio drawdown bound, including continuous
account peaks. It assumes unfavorable intrabar/cross-asset ordering and is not
an exact tick-level measurement. The historical selection budget is **20%**;
it is not a guaranteed future maximum loss.

Capped allocation has higher nominal profit than the original benchmark, but its
delayed drawdown exceeds the budget. Ranked momentum exceeds the nominal budget.
Consensus stays within the budget in these three historical scenarios but earns
less than capped allocation and does not pass the fixed improvement screen.
None of these comparisons establishes a production-ready strategy.

The broad-universe experiment expands to **15 assets from a dated December 2021 cohort**,
with historical liquidity selection, weekly ranks, old/new Terra separation and
documented MATIC/POL continuity. Development freezes `broad_top5_vol20` before
later economic replay. All six broader policies trail both controls on full
nominal net profit and exceed 20% full drawdown. The frozen policy also fails
cost/time stress; a material improvement is not supported. Its descriptive paired
block interval is not confirmatory statistical significance on reused data.

The multiscale and broad-universe studies each reconcile **104 account paths** at
every 4h close. Both previous controls are reproduced. No policy in either study
doubles its account in a rolling 7/30/90-day window. These overlapping historical
windows do not estimate independent future doubling probabilities.

- [Broad-universe results, source audits and all rejected variants](docs/evaluation/THETABOT_BROAD_UNIVERSE_2026-10-09.md) · [JSON evidence](docs/evaluation/THETABOT_BROAD_UNIVERSE_2026-10-09.json) · [Fixed protocol](docs/evaluation/BROAD_UNIVERSE_RESEARCH_PLAN.md)
- [Multiscale, cost and timing results](docs/evaluation/THETABOT_MULTISCALE_SPOT_2026-10-09.md) · [Fixed protocol](docs/evaluation/MULTISCALE_SPOT_RESEARCH_PLAN.md)

The Bayesian/Monte Carlo follow-up adds two causal five-day log-return experts
and a mixture weighted by already completed joint forecast errors. Bayesian
regression separates predictive variance from uncertainty in its mean. Conditional
Monte Carlo samples whole completed return blocks from similar historical price,
volume and theta states; sampling does not add independent market information.
Eight fixed policies compare buy filters and optional target blends against the
existing capped spot strategy and a matching warmup-only control.

For a separately funded 10,000-USDT account over January 2022–September 2026,
spot earns **+108.82%**, warmup alone **+113.01%**, the Bayesian blend **+58.21%**,
MC blend **+56.30%** and combined blend **+60.50%**, after modeled costs.
The combined blend's worst full modeled DD is **12.89%**, versus **21.16%** spot,
but reduced exposure earns less. All three experts have worse log-return MSE and
gain-probability Brier loss than unconditional history on identical full/later
rows. The frozen choice for both account sizes is the warmup-only control; its
full joint-stress drawdown exceeds the 20% budget. No economic improvement passes.

The MC buy filter changes full net profit from **+7.08% to +19.45%** solely with
the three fixed seeds; thresholds can amplify numerical sampling differences.
The study reconciles **368 paths**, reproduces **32 archived controls**, and audits
**31,212 source decisions** plus all 16 full nominal exported ledgers. The
integration defaults off, preserves the original target when disabled, and
adds no live strategy choice or option exposure.

- [Bayesian/Monte Carlo results and rejected variants](docs/evaluation/THETABOT_BAYES_MONTE_CARLO_2026-10-12.md) · [Compressed JSON evidence](docs/evaluation/THETABOT_BAYES_MONTE_CARLO_2026-10-12.json.gz) · [Complete CSV audit archive](docs/evaluation/THETABOT_BAYES_MONTE_CARLO_2026-10-12_EVIDENCE.zip) · [Fixed protocol](docs/evaluation/BAYES_MONTE_CARLO_RESEARCH_PLAN.md)

The cost/risk follow-up changes execution rather than adding another forecast.
Ordinary rebalances can trade to the nearest edge of the same 1pp band; mandatory
inactivity/cap sales still bypass it. An optional 15% annualized volatility guard
checks actual held weights at each 4h open using only completed, lagged daily
covariance. It reduces the proposed exposure and never replenishes cash or adds
leverage. Both options are disabled in the original replay defaults.

The combined `guarded_edge` policy was fixed before development payoffs; the
separate band/risk changes are diagnostic ablations, not later replacements.
For **10,000 initial USDT**, over January 2022–September 2026, its modeled net
profit is **+83.37%**, versus **+108.82%** control; joint cost/time stress gives
**+70.47%**, versus **+84.20%**. Worst full-period conservative DD improves from
**21.16% to 15.03%**. Full nominal fees/impact fall from **1,238.57 to 857.55 USDT**,
but more frequent risk reductions increase fill count from 818 to 1,321.
For both account sizes the risk screen passes and the profit-improvement screen
fails. Lower exposure/risk must not be described as newly discovered alpha.
Minimum-notional residuals are disclosed; requested volatility and historical
drawdown limits are not future guarantees. No live selection is enabled.

All **160 paths** reconcile; **40 archived controls** reproduce. The study
rechecks **171 official archive checksums** and exports all 32 full accounts.
A separate NumPy CSV audit verifies **332,928 bars and 23,041 fills**, including
source-open prices, costs, drawdown and actual-holdings covariance mathematics.

- [Cost/risk results and tradeoffs](docs/evaluation/THETABOT_COST_RISK_2026-10-13.md) · [Compressed JSON](docs/evaluation/THETABOT_COST_RISK_2026-10-13.json.gz) · [CSV evidence manifest](docs/evaluation/THETABOT_COST_RISK_2026-10-13_EVIDENCE.zip.manifest.json) · [Fixed protocol](docs/evaluation/COST_RISK_RESEARCH_PLAN.md)

Prospective validation now separates future observations from that known history.
A frozen protocol binds the two existing policies, capital, costs, source hashes,
runtime and a future UTC start. Explicit snapshots collect only complete official
spot 4h bars. Each update replays the entire funded forward account and checks
every previously accepted data row, receipt and ledger; changing history or
rules, rewinding time, stale data and concurrent writers fail closed.

The preliminary screen needs **180 full days of closed forward bars**, positive
candidate P&L under all four cost/time scenarios, profit above the same control
in nominal/joint stress, reconciled accounts and conservative drawdown within 20%.
An early drawdown breach rejects it immediately. Passing never enables live
trading. This is a fixed model replay, not measured executable paper orders;
the public protocol commit must predate the forward boundary. No background
collection or automatic trading is started by these commands.

- [Forward-validation plan](docs/evaluation/FORWARD_VALIDATION_PLAN.md)
- [Published freeze](docs/evaluation/THETABOT_FORWARD_PROTOCOL_2026-10-16.json) · [Initial source/state evidence](docs/evaluation/THETABOT_FORWARD_INITIAL_2026-10-16.json) · [Raw snapshot and receipt](docs/evaluation/THETABOT_FORWARD_INITIAL_2026-10-16_EVIDENCE.zip)

The published freeze starts scoring at **2026-10-17 00:00 UTC**. Its initial
public-feed check contains 719 warmup bars per asset and **zero forward bars**;
the phase is `waiting_for_start`, not a successful performance result. The first
possible 180-day endpoint is **2027-04-15 00:00 UTC**, subject to complete data
and all other gates. Collection/evaluation still require explicit invocations.

The fixed Möbius/theta diagnostic tests completed-return extrapolation as an
additional buy gate on the same guarded momentum strategy. For 10,000 USDT,
full January 2022–September 2026 net falls from **+83.37%** without a gate to
**+42.15%** with the full theta gate and **+39.12%** with Möbius weighting;
the latter's worst modeled full DD is **16.35%**, versus **15.03%** control.
The separately funded 2025–September 2026 comparison gives **+30.78%, +15.09%
and +12.52%**, respectively. The candidate fails the frozen improvement screen.

Unregularized full theta and plain cosine forecasts agree to 1.32e-16;
Möbius weighting and simple squarefree-frequency selection agree to 4.86e-17,
with exactly identical buy masks on all source rows. Exact Möbius scale inversion
reduces theta3 to a single cosine. None of these transformations supplies an
independent market signal. Later predictive correlations are near zero and the
Möbius MSE exceeds the simple trailing-mean control. The test reproduces 24 old
control paths, reconciles 120 distinct paths and independently audits 10 exported
full accounts. Known history is reused; the earlier forward protocol's source
fingerprint and live settings are preserved.

- [Möbius/theta results](docs/evaluation/THETABOT_MOBIUS_THETA_2026-10-22.md) · [Complete JSON](docs/evaluation/THETABOT_MOBIUS_THETA_2026-10-22.json.gz) · [Frozen plan](docs/evaluation/MOBIUS_THETA_RESEARCH_PLAN.md) · [Published protocol](docs/evaluation/THETABOT_MOBIUS_PROTOCOL_2026-10-22.json) · [CSV evidence manifest](docs/evaluation/THETABOT_MOBIUS_THETA_2026-10-22_EVIDENCE.zip.manifest.json)

The optional paid-option model initially reserves 10% of the original account,
with monthly purchases capped at 20% of that sleeve's current cash, including
fees. It never replenishes option losses from spot. Four fixed signals on inverse
and USDC contracts are compared with full-capital spot and idle-cash controls.
For 1,000 quote units, options do not improve full-period profit; inverse minimum
lots prevent every buy. For 10,000 units, the development-frozen inverse directional
5%-OTM policy models **+314.31%** full-account net versus **+108.82%** spot, and
**+53.44%** versus **+35.31%** on the separately funded 2025–2026 account.

This model uses historical trades with premium markups, not executable asks;
held-option marks and drawdowns are synthetic. Full profit falls to **+132.15%**
with a 4h delay. Cost stress gives **+236.23%**, but modeled DD reaches **20.14%**
and fails the 20% budget. Three options contribute 80.04% of net option P&L.
Retained wins grow the sleeve to 54.62% of the final account, so its initial 10%
funding is not a permanent exposure cap. This is an exploratory result with no
live eligibility; it does not establish executable option profitability.

- [All option models, stress results and limitations](docs/evaluation/THETABOT_LONG_OPTIONS_2026-10-10.md) · [JSON evidence](docs/evaluation/THETABOT_LONG_OPTIONS_2026-10-10.json) · [Closed-option audit](docs/evaluation/THETABOT_LONG_OPTIONS_2026-10-10_CLOSED_OPTIONS.csv) · [Fixed protocol](docs/evaluation/LONG_OPTIONS_RESEARCH_PLAN.md)

The bounded-purchase follow-up adds an optional limit on monthly premium plus
purchase costs using current funded account opening equity or initial capital.
The old 20%-of-option-cash limit also applies. A 2%-of-current-account directional
variant models **+176.13%** full profit for 10,000 units. Joint cost-plus-delay
stress improves from **+85.09%** in the old directional model to **+100.74%**;
the new variant's worst modeled DD across four scenarios is **19.89%**.
Actual nominal monthly purchase debits fall from a maximum 5.98% to 1.89% of
contemporaneous funded equity. This bounds purchase spending, not held market
values or future drawdown.

Development's four-scenario choice is the more conservative `call_current1`,
with +110.80% nominal full profit and 19.18% worst modeled DD. It does not beat
spot in every profit scenario. The directional 2% model is a descriptive candidate,
not a replacement selected on an unseen period. Excluding its three largest
positive option fills reduces full profit to **+98.73%**, below **+108.82%** spot;
profit concentration remains material. No new inverse variant buys an option
for the 1,000-unit account. The study reconciles 448 paths and reproduces 96
archived control comparisons; the execution-price limitations remain.

- [Bounded purchase results and all stresses](docs/evaluation/THETABOT_OPTION_BUDGETS_2026-10-12.md) · [JSON evidence](docs/evaluation/THETABOT_OPTION_BUDGETS_2026-10-12.json) · [Monthly purchase audit](docs/evaluation/THETABOT_OPTION_BUDGETS_2026-10-12_PURCHASE_BUDGETS.csv) · [Fixed protocol](docs/evaluation/OPTION_BUDGET_RESEARCH_PLAN.md)

The causal payoff-selection follow-up compares six price filters, two warmup-only
controls and the existing funded controls. Forecasts use only completed prior
daily closes: trailing empirical holding-period returns or a 60-day-volatility
lognormal model. A purchase needs estimated net delivery proceeds above the full
premium debit by a fixed margin; a positive estimate is not calibrated market alpha.
Entry-volatility scales of 0.75/1.25 join cost/time scenarios in the development
choice. Current-account purchase caps and the default-off research switch remain.

Development freezes `call_empirical10` for 10,000 units: **+168.61%** full modeled
net, **+52.43%** on the separately funded 2025–2026 account, **+98.87%** with joint
cost/time stress and **18.44%** worst modeled DD over six scenarios. A warmup-only
call control earns **+170.88%** nominal, so the nominal improvement cannot be
attributed solely to the forecast. The filter improves joint-stress net and modeled
DD against that control. Its three largest positive fills account for **92.96%**
of option P&L; missing them reduces full net to **+101.42%**, below **+108.82%** spot.
Forecast-volatility uncertainty changes the chosen policy's net from +132.41%
to +170.88%. Dates are reused and purchase quotes remain unverified. No inverse
option policy buys for 1,000 units under these lot/budget assumptions.

- [Payoff selection, warmup controls and forecast uncertainty](docs/evaluation/THETABOT_OPTION_SELECTION_2026-10-12.md) · [Compressed JSON evidence](docs/evaluation/THETABOT_OPTION_SELECTION_2026-10-12.json.gz) · [Decision audit](docs/evaluation/THETABOT_OPTION_SELECTION_2026-10-12_ENTRY_SELECTION.csv) · [Fixed protocol](docs/evaluation/OPTION_FORECAST_RESEARCH_PLAN.md)

## Binding trading constraints

Only owned spot cash and inventory may be traded. Leverage, loans, margin,
financial shorts, futures/options, leveraged tokens and martingale loss-doubling
are excluded from live execution. A 2026-10-10 user-authorized exception permits
offline modeling of fully paid long calls/puts in an isolated cash sleeve; the
research switch defaults off and has no order adapter. Long options still contain
economic leverage. See the [fixed option protocol](docs/evaluation/LONG_OPTIONS_RESEARCH_PLAN.md).
Cash and inventory must remain nonnegative, including fees. The
planner rejects its legacy short switch; impossible fills are rejected. Exchange
execution requires verified spot-market information and sufficient free funds.

Latest portfolio studies use 50% aggregate and 25% per-asset target caps, Monday
ordinary rebalances, daily inactivity/cap reductions, a 1pt band and a 10-USDT
minimum fill. Positions can drift between decisions. An absent quote cannot
execute a trade. Source/indicator prices stay absent; explicitly flagged replay
sentinels value held unquoted inventory at zero conservatively, and any such path
fails the candidate's quality gate.

Offline research and passing tests do not authorize deployment, orders, changing
live defaults or merging a PR. The research portfolio policies above are
available through `scripts.research_*`; they are not the runner's live defaults.
See [the binding spot-only policy](docs/evaluation/SPOT_ONLY_POLICY.md) and
[repository constraints](AGENTS.md).

## Setup and offline evaluation

Run commands from the repository root. Python 3.12 is used in CI. Install the
project's dependencies from [requirements.txt](requirements.txt); PyTorch is
optional for the dual-stream research model.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
python test_eval_metrics.py

# Bundled dataset; fully offline, no exchange orders
python -m spot_bot.evaluate --out bench_out/evaluation
```

`spot_bot.evaluate` selects among its existing single-asset strategies on the
first 60% of observations, freezes the choice and tests the remaining 40% with
historical feature warmup and fresh cash. It writes `summary.json`, `report.md`
and holdout equity/trade CSVs. The bundled dataset has been used before.
`--require-paper-pass` returns a failing status when screening fails. A passing
report does not enable live execution.

For checksum-verified recent public spot data, without account credentials:

```bash
python -m scripts.download_binance_archive --start-month 2026-04 --end-month 2026-09 \
  --out data/raw/BTCUSDT_1h_2026_04_09.csv
python -m spot_bot.evaluate --csv data/raw/BTCUSDT_1h_2026_04_09.csv \
  --data-source binance_archive --out bench_out/evaluation_recent
```

This single-asset evaluation is separate from the multi-asset research accounts
in the results table. It does not reproduce their portfolio returns.

## Reproduce the portfolio experiments

The downloaders use official public spot archives and verify published SHA-256
checksums. Source manifests retain URLs, hashes, identity rules and absence audits.
The broad downloader verifies 853 monthly archives, 15 pre-period proofs and one
corroborating daily artifact. Conflicting or unverified source rows stop the study.

```bash
# BTC/ETH/BNB controls, refinements and multiscale comparison
python -m scripts.download_spot_flow
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_momentum_refinement
python -m scripts.write_momentum_refinement_report
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_multiscale_spot
python -m scripts.write_multiscale_spot_report

# Fixed historical 15-asset cohort; reuses the three-asset cache when available
python -m scripts.download_spot_universe
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_broad_universe
python -m scripts.write_broad_universe_report

# Optional paid-option MODEL: historical trade proxies, synthetic held marks
python -m scripts.download_long_options
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_long_options --options-enabled
python -m scripts.write_long_options_report

# Optional funded-account purchase limits; joint cost/time and missed-fill stresses
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_option_budgets --options-enabled
python -m scripts.write_option_budgets_report

# Causal option payoff forecasts, warmup-only controls and forecast uncertainty
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_option_selection --options-enabled
python -m scripts.write_option_selection_report

# Optional Bayesian/conditional-Monte-Carlo spot signal overlays
python -m scripts.research_bayes_monte_carlo --probabilistic-enabled --jobs 4
python -m scripts.write_bayes_monte_carlo_report

# Fixed cost/risk execution study; freeze before any new candidate payoffs
python -m scripts.research_cost_risk --freeze-only
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_cost_risk --jobs 4 --report-out docs/evaluation
python -m scripts.audit_cost_risk

# Fixed entry hysteresis, then bounded fixed risk-utilization follow-up
# Use the pinned runtimes recorded in the dated protocols; each freeze is exclusive.
python -m scripts.research_entry_hysteresis freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_entry_hysteresis run --jobs 4
python -m scripts.research_risk_utilization freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_risk_utilization run --jobs 4
python -m scripts.research_robustness freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_robustness run --jobs 4

# Prospective model replay: publish the protocol before its future start
python -m scripts.forward_spot freeze --protocol artifacts/forward_spot/protocol.json
python -m scripts.download_forward_spot --protocol artifacts/forward_spot/protocol.json --out data/raw/forward_spot
# SNAPSHOT is the immutable directory printed by the downloader
python -m scripts.forward_spot evaluate --protocol artifacts/forward_spot/protocol.json --data SNAPSHOT --out artifacts/forward_spot/accounts
```

Each fixed study persists its development selection before later economic replay.
Completed-day targets execute at the next open, with sells before buys and shared
cash. Raw CSV/cache/manifests live under `data/raw/`; replay selections, targets,
quote flags and equity/fill CSVs live under `artifacts/` (Git ignored). Durable
report/JSON evidence is under `docs/evaluation/`. Reports link their exact protocols.

For an individual existing strategy on a local CSV:

```bash
python -m spot_bot.run_live --mode backtest --strategy kalman_fusion \
  --csv-in data/raw/spot_flow/BTCUSDT_4h.csv --timeframe 4h --limit-total 20000 \
  --execution-policy market --fee-rate 0.001 --slippage-bps 5 --spread-bps 2 \
  --max-exposure 0.3 --out-summary bench_out/kalman_fusion_summary.json
```

`dryrun` emits intent without orders; `paper` uses simulated fills and requires a
SQLite `--db`. Exchange execution code also exists and has an explicit live opt-in.
Its presence does not establish a profitable bot. Limit-maker simulation lacks
order-book queue and partial-fill evidence. Use `--help` for runner parameters.

## Architecture and research components

| Area | Role |
|---|---|
| [spot_bot/core/](spot_bot/core/) | Owned-funds planning, simulated execution, fee accounting and portfolio state |
| [spot_bot/backtest/](spot_bot/backtest/) | Single/shared-cash replay, quote flags, drawdown and independent ledger reconciliation |
| [spot_bot/portfolio/](spot_bot/portfolio/) | Completed-day momentum, allocation, covariance budgets, liquidity and weekly ranks |
| [spot_bot/strategies/](spot_bot/strategies/) | Mean reversion, Kalman, trend/breakout and forecast-fusion strategies |
| [spot_bot/features/](spot_bot/features/) | Causal features and external-source availability handling |
| [spot_bot/execution/](spot_bot/execution/) | Exchange execution adapters and spot-account checks |
| [theta_bot_averaging/](theta_bot_averaging/) | Theta/Mellin models and walk-forward predictor experiments |
| [scripts/](scripts/) | Checked data downloaders, fixed economic studies and report writers |

The theta research includes orthonormal Jacobi bases, complex-time/phase
representations, biquaternion drift and PCA regimes. The dual-stream model
combines theta and Mellin features with optional PyTorch and a sklearn fallback.
These implementations and their mathematical tests do not establish market profit.

The historical [V9 predictivity evaluation](V9_EVALUATION_SUMMARY.md) reports mixed,
weak results on mock data. It does not support the former README's expected
real-market correlation/hit-rate ranges or a validated theta trading edge.
Synthetic diagnostics remain in [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md).
Signed forecasts are research outputs; executable spot targets are nonnegative.
Legacy long/short P&L experiments are outside the current trading policy.

Further technical material: [theta/CTT documentation](CTT_README.md),
[dual-stream configuration](configs/dual_stream_example.yaml),
[implementation notes](IMPLEMENTATION_SUMMARY.txt) and
[Atlas theta-function paper](theta_bot_averaging/paper/atlas_evaluation.tex).
These are research references, not production-profit certificates.

## Earlier experiments and evidence

| Study | Evidence |
|---|---|
| Momentum allocation refinements | [Results](docs/evaluation/THETABOT_MOMENTUM_REFINEMENT_2026-10-09.md) |
| Executed spot-flow signals | [Results](docs/evaluation/THETABOT_SPOT_FLOW_2026-10-07.md) |
| Forex/equities/oil/rates, sentiment and variable-lag forecasts | [Results](docs/evaluation/THETABOT_EXTERNAL_SIGNALS_2026-10-06.md) |
| 20% historical risk-budget comparison | [Results](docs/evaluation/THETABOT_RISK_BUDGET_2026-10-06.md) |
| Portfolio execution and sizing sensitivity | [Execution](docs/evaluation/THETABOT_EXECUTION_PORTFOLIO_2026-10-06.md) · [Sizing](docs/evaluation/THETABOT_EXPOSURE_2026-10-06.md) |
| Multi-model forecasts and correlation-aware Kalman fusion | [Results](docs/evaluation/THETABOT_MULTI_2026-10-06.md) |
| Bayesian/conditional-Monte-Carlo spot signal overlays | [Results](docs/evaluation/THETABOT_BAYES_MONTE_CARLO_2026-10-12.md) |
| Ordinary band-edge execution and actual-holdings volatility guard | [Results](docs/evaluation/THETABOT_COST_RISK_2026-10-13.md) |
| Fixed entry hysteresis and diagnostic confirmations | [Results](docs/evaluation/THETABOT_ENTRY_HYSTERESIS_2026-10-25.md) |
| Fixed 15%/17.5%/20% volatility requests | [Results](docs/evaluation/THETABOT_RISK_UTILIZATION_2026-10-25.md) |
| Fixed asset omissions, calendar losses and asset attribution | [Results](docs/evaluation/THETABOT_ROBUSTNESS_2026-10-25.md) |
| Single-asset chronological audit | [Results](docs/evaluation/THETABOT_2026-10-06.md) |

Higher historical exposure and legacy execution metrics in earlier reports are
archived comparisons. The latest corrected replay accounts for fees, spread,
slippage, continuous peaks and delayed completed-day signals. Repeatedly studied
periods must not be relabeled as fresh holdouts. The external-source report also
discloses unavailable first-release vintages and the composite nature of Fear &
Greed; its joint oil/macro test does not isolate miners' electricity costs.

## Verification and remaining work

The cost/risk change adds 25 focused cases for completed covariance clocks,
ordinary turnover, required cap sales, actual inventory reductions, minimum
notionals, unchanged defaults, prefix invariance, independent CSV math and
corruption rejection. The preceding Bayesian/Monte Carlo study passed 669
regression tests and eight legacy tests; its predictive/economic failures remain
documented. The final cost/risk tree passes **694 regression tests and eight
legacy tests locally**, plus the chronological evaluator and entry-point syntax.
The prospective layer adds 21 focused cases; the complete updated tree passes
**715 regression and eight legacy tests locally**. These verify the software and
the evidence controls, not world-leading profitability.
The fixed-allocation audit adds 24 focused cases; the complete tree passes
**763 regression and eight legacy tests locally**. The new cases cover completed
day publication, future-prefix invariance, independently computed capped weights,
owned funding, risk warmup and rejection of altered source/runtime freezes.
The depth-paper layer adds 34 focused cases; that tree passed
**797 regression and eight legacy tests locally**. These check depth consumption,
exact lot steps, fee-inclusive cash budgets, refusal of unavailable liquidity,
observation clocks, independent rational accounting and immutable evidence.
The publication wrapper adds three cases for deterministic replay, conflict
rejection without changing any files, and safe capture identity. All **37
depth/publication cases pass locally**; the full suite also runs in CI.
The entry/risk follow-up adds **54 focused cases**, all passing locally, plus
**107 related regression cases and eight legacy tests**. The new checks cover
entry-state memory, strict hurdle/warmup, future-prefix invariance, unchanged
controls and prior frozen sources, configured-risk raw CSV reconciliation,
corrupted risk fields, both covariance lags and altered protocol/report rejection.
The calendar/omission audit adds **38 focused cases**, all passing locally.
They check budget redistribution, absence of omitted inventory/fills with both
lags, compounded calendar/rolling returns without account resets, complete-month
boundaries, log-return concentration, asset attribution including price gaps and
fees/impact, and rejection of altered evidence even after outer rehashing.
An additional standard-library evidence test recomputes **798 monthly returns
and 1,120 overlapping 12/24-month windows** with Decimal arithmetic from the
14 published raw accounts, and verifies all **64 CSV checksums**, calendar
years, concentration arithmetic, source fingerprints and attribution totals.
The [CI workflow](https://github.com/DavJ/theta-bot/actions/workflows/eval-metrics.yml)
also checks entry-point syntax, chronological evaluation and evidence upload.
The broader suite covers future-prefix invariance, token identity,
publication clocks, sparse quotes, owned balances and independent accounting.
CI success proves these checks passed; it does not prove future profit or replace
a security review.

Further work requires a fixed protocol, source/publication audits and a candidate
that survives costs, latency and the 20% historical risk budget. Confirmation
then needs genuinely unseen or forward paper evidence. Rejected variants and all
cost/risk failures remain documented. No production rollout follows automatically.

Additional data guidance: [real-data runbook](docs/data/REAL_DATA_RUNBOOK.md) and
[data protocol](docs/data/DATA_PROTOCOL.md). Older
[production-preparation notes](PRODUCTION_PREPARATION.md) are historical tooling
references and do not supersede the current status or binding trading constraints.
