# Prospectively frozen spot validation — 2026-10-16

The known-history research improved drawdown, not net profit. This next step
tests that fixed claim on observations occurring after a published freeze.
It adds no new return model, live order adapter or default strategy.

## Fixed design

- Publish the protocol before its forward start, at the next UTC midnight.
- Start each policy/scenario with 10,000 USDT and no inventory. Replay the
  entire forward account on every update; never reset capital at a new year,
  evaluation invocation or favorable window.
- Compare only the existing capped-allocation `spot_control` and pre-fixed
  `guarded_edge`. No later winner selection, parameter grid or option sleeve.
- Keep the existing 50% total / 25% asset targets, 1pp ordinary band, Monday
  rebalancing, inactivity/cap exits and 10-USDT minimum. The candidate retains
  its 15% annualized volatility target and 60-day / 25%-shrunken covariance.
- Test nominal, doubled fee/slippage, extra 4h signal/covariance lag, and their
  combination. Nominal per-fill costs are 0.1% fee, 5bps impact and half of a
  2bps spread; stress leaves spread unchanged.
- Use BTCUSDT, ETHUSDT and BNBUSDT official **spot** `/api/v3/klines`, UTC 4h
  bars. The 120 days before start are feature warmup, not scored forward data.
  Reject missing, duplicate, misaligned, invalid or unfinished bars and changed
  historical prefixes. Retain each raw snapshot and its SHA-256 provenance.

## Evidence and preliminary screen

Bind the settings, evaluator/strategy/core source hashes and runtime versions
in an exclusively created protocol. It must be frozen before the start; publish
its Git commit before that boundary as an external timestamp anchor. Local hashes
detect accidental changes; they are not an unforgeable publication timestamp.

Require **180 full days of closed forward bars**, not 180 days merely elapsed
on a clock. A short or stale dataset cannot pass. Preliminary success requires
positive candidate net P&L in all four scenarios, nominal and joint-stress net
P&L above the fixed control, all candidate conservative drawdowns within 20%,
and independently reconciled owned cash/inventory/NAV for every account.
An early drawdown breach rejects the screen immediately. Passing this small
screen never sets `live_eligible` true and does not establish world-leading
performance or future profits.

The evaluator checks every previously accepted data row and receipt/ledger hash
before updating. Outputs are immutable run directories plus an atomically
replaced state pointer under an exclusive lock. A crash cannot publish a partly
written new state. Extending the history must preserve previous fills/NAV;
code/config/runtime changes require a new, explicitly separate freeze.

This is a prospective **model replay**, not a record of live or executable paper
orders. Market-open fills, spread and slippage remain hypotheses; the timing
stress does not delay emergency order execution. Terminal holdings are marked
to market. Custody, taxes, USDT FX/depeg risk and historical order-book depth are
not modeled. The 20% screen and volatility request are not future loss caps.

## Commands

```bash
python -m scripts.forward_spot freeze --protocol artifacts/forward_spot/protocol.json
python -m scripts.download_forward_spot --protocol artifacts/forward_spot/protocol.json --out data/raw/forward_spot
# Use the immutable snapshot directory printed by the downloader:
python -m scripts.forward_spot evaluate --protocol artifacts/forward_spot/protocol.json --data SNAPSHOT --out artifacts/forward_spot/accounts
```

Each command is invoked explicitly. No recurring task, background daemon, live
exchange connection, trading API key, default change or deployment is enabled.
