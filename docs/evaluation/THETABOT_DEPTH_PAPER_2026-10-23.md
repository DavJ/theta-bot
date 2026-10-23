# Public spot-depth paper calibration — 2026-10-23

**Conditional static execution-cost probes; observed exchange fills: 0.**

Six markets, four all-in buy budgets and two frozen fee/reserve scenarios.
Each independently funded probe buys visible asks then sells the acquired inventory into the same book.
USDT and USDC accounts are separate; actual commissions and account eligibility are unknown.

| Market | Full spread (bps) | 100 nominal round-trip (bps) | 2,500 nominal (bps) | 2,500 stress (bps) | Request ms |
|---|---:|---:|---:|---:|---:|
| BTCUSDT | 0.0012 | 19.9812 | 19.9812 | 49.8765 | 2672.0 |
| ETHUSDT | 0.0417 | 20.0216 | 20.0216 | 49.9167 | 1940.0 |
| BNBUSDT | 0.1347 | 20.1145 | 20.1145 | 50.0093 | 1989.0 |
| BTCUSDC | 0.0012 | 19.9812 | 19.9812 | 49.8765 | 2002.0 |
| ETHUSDC | 0.0417 | 20.0217 | 20.0217 | 49.9168 | 1992.0 |
| BNBUSDC | 0.1349 | 20.1146 | 20.5885 | 50.4817 | 2911.0 |

Accepted conditional probes: 48; rejected probes on valid books: 0.
Both legs of every accepted probe independently reconcile using rational arithmetic.

Protocol: `0c187f023ed71190de31d782c5d5ace6cc6c353f8b63077f5eff3288741ecaf3`. Capture: `c5cc5075014dd64b7014b7d71e5605b7a5be8ddca73f4f7d2e85e6e2424c125e`.
Observed at: 2026-10-23T20:37:17.076Z.

## Interpretation

- Nominal modeled fees are 0.1% per leg in quote; stress uses 0.2% plus a hypothetical 5bps adverse reserve per leg.
- Depth REST replies contain no exchange book timestamp. Source book age remains unknown; the receipt clock is only local observation time.
- These static round trips measure spread/depth/cost, not a momentum strategy return or realized exchange execution.
- Market quantity filters are checked. VWAP notional checks do not establish reference-price, execution-rule or account-specific eligibility.
- One batch does not estimate tail costs or justify reducing historical slippage assumptions.
- The coverage gate requires 200 distinct successful books per market spanning 30 days.
- No background collector or order submission is started. The earlier candle-based prospective protocol stays separate.

[Frozen plan](DEPTH_PAPER_EXECUTION_PLAN.md).

## Coverage

| Market | Distinct successful books | Span days | Coverage gate |
|---|---:|---:|:---:|
| BTCUSDT | 1 | 0.0000 | False |
| ETHUSDT | 1 | 0.0000 | False |
| BNBUSDT | 1 | 0.0000 | False |
| BTCUSDC | 1 | 0.0000 | False |
| ETHUSDC | 1 | 0.0000 | False |
| BNBUSDC | 1 | 0.0000 | False |

## Reproduce / explicitly collect another batch

```bash
python -m scripts.research_paper_execution freeze --protocol artifacts/depth_paper/protocol.json
python -m scripts.research_paper_execution collect --protocol artifacts/depth_paper/protocol.json --out artifacts/depth_paper
# Or replay a saved connector capture:
python -m scripts.research_paper_execution replay --protocol artifacts/depth_paper/protocol.json --capture CAPTURE.json --out artifacts/depth_paper
```
