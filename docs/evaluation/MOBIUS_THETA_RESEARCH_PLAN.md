# Fixed Möbius/theta experiment — 2026-10-22

This protocol precedes the new forecasts and economic payoffs. January 2022–
September 2026 and the surviving BTC/ETH/BNB universe were already studied.
This is a diagnostic historical replay, not an unseen holdout or a continuation
of the separate forward account frozen on 2026-10-16.

## Question and mathematical controls

Does the user's proposed Möbius weighting improve theta-based forecasts and
net spot results? The fixed candidate is `mobius_gate`; no development or later
winner replaces it. All other variants diagnose its mechanism.

Use completed 4h close log returns, 48-return rolling windows, q=0.9, harmonics
n=1..8 and a six-bar (24h) forecast. These parameters adapt the legacy rolling
theta embedding; there is no horizon, q, window or threshold search. Demean each
past window, fit its cosine basis by unregularized least squares, extrapolate
the next six returns and add six times the past mean. The phase grid is exactly
the legacy linspace(0,2*pi,48), continued at the same step into the future.

Compare these predictions on identical matured rows:

- `theta_full`: q^(n*n) cos(2*n*phase), all eight harmonics.
- `mobius`: multiply those columns by mu(n), including zeros at n=4 and 8.
- `fourier_full`: the same cosine columns without q scaling.
- `fourier_squarefree`: unweighted cosines at precisely mu(n)!=0 indices.
- `fundamental`: the single cosine obtained by exact Möbius scale inversion.
- `mean`: six times the trailing mean, with the same warmup.

Nonzero column scaling/signs do not change the unregularized regression span;
theta/full-Fourier and Möbius/squarefree-Fourier predictions and buy masks must
agree. Report discrepancies and stop if masks differ. For 0<q<1 and real z,
sum_k mu(k)[theta3(k*z,q^(k*k))-1] = 2*q*cos(2*z), because the coefficient
of harmonic m contains sum_{k|m}mu(k). Thus exact inversion is a fundamental
cosine control, not independent market information. M(N)=sum_{n<=N}mu(n);
summation by parts is an equivalent representation, not another fitted expert.

References: https://dlmf.nist.gov/20.2.E3 and https://dlmf.nist.gov/27.5.

## Predictive measurement

The target at the close of bar t is log(close[t+6]/close[t]). Score only when
the full six-bar target is inside the period. Use a fixed six-bar subsample
anchored at 2022-01-01 to avoid overlapping scored targets within each asset;
assets remain dependent. Report sample counts, MSE, MAE, sign accuracy and
Pearson correlation for development, 2025, 2026, later and full periods.
Same-window reconstruction correlation is not predictive evidence. Forecast
errors and historical ranks are descriptive, without significance claims.

## Spot replay

Recover the identical 171 checksum-verified official monthly spot archives and
use the existing reader, shared-cash engine and risk/accounting code. No new
market or account connection, execution adapter, price proxy or synthetic data.
The control is the previous `guarded_edge`: capped 30-day momentum, Monday
ordinary rebalancing, 50% total/25% asset targets, 1 percentage-point band edge,
10-USDT minimum, actual-holdings 15% annualized volatility guard using 60
completed daily returns and 25% covariance shrinkage.

Each forecast only permits new buys/increases when its predicted 24h log return
exceeds the fixed 0.0032 nominal round-trip hurdle. This is a heuristic forecast
gate, not an exact break-even quote or a 24h holding-period instruction. It does
not prevent sells, inactivity exits, cap reductions or emergency risk sales.
The engine shifts both targets and buy masks to the next open, plus one more
4h bar under timing stress. Forecast targets use closes; trade returns use
modeled opens and terminal marks. Fixed gates are not retuned for stress costs.

Evaluate 1,000 and 10,000 USDT independently under nominal, doubled fee/slippage,
one extra 4h signal/covariance lag, and combined stress, with the old cost model.
Development is 2022–2024; later is a separately funded continuous 2025–Sep2026
account; full is one continuous 2022–Sep2026 account. Never add their returns.
Freeze the fixed development result before computing later/full economics.

Full Fourier duplicates theta and squarefree Fourier duplicates Möbius only
after prediction AND full buy-mask equivalence are verified. Reuse and label
their identical economic paths instead of pretending these are independent
trials. Reproduce all 24 matching old guarded-edge control paths.

Historical candidate improvement requires reconciled accounts, strictly lower
later forecast MSE than both theta and mean, positive later/full net in every
scenario, later/full nominal and joint-stress net above both guarded control
and theta gate for both capital sizes, and every candidate modeled drawdown
within 20%. Also disclose structural equivalence to squarefree Fourier: a pass
cannot establish special theta/Möbius alpha. No candidate replacement or tuning
after inspecting outcomes, even if this screen fails.

## Evidence and scope

Freeze plan, implementation, used engine/loader source fingerprints and runtime
before payoffs. Save an exclusive, checksummed development selection. Preserve
source provenance, all summaries, full forecast CSVs, and nominal/joint full
10,000-USDT cash/inventory/trade CSVs for every distinct policy. Independently
reconcile exported CSV NAV, fills and nonnegative balances, verify source-open
prices/costs, and verify prior control summaries. Export checksummed evidence
and explicit limitations. Files added outside spot_bot preserve the prior
forward protocol's source fingerprint. Old source/defaults remain unmodified.

USDT archives support research; they do not establish available EEA execution.
Any USDC/EUR implementation needs separate market/cost validation. No order-book
depth, executable paper fills, taxes, custody, operating costs or quote-FX risk
is modeled. The 20% historical screen does not cap future loss. Passing tests
or this historical screen never permits live trading or changes live eligibility.
