"""Write all fixed spot-only results, costs and rejected variants for review."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def pct(value):
    return f"{100 * value:+.2f}%"


def write_report(source, out):
    result = json.loads(source.read_text())
    out.mkdir(parents=True, exist_ok=True)
    stem = "THETABOT_SPOT_FLOW_2026-10-07"
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    json_path = out / (stem + ".json")
    json_path.write_text(payload)
    nominal, doubled = result["later"]["nominal"], result["later"]["doubled"]
    frozen = result["frozen_candidate"]
    variants = list(result["development"])
    manifest = result["source_manifest"]
    archive_count = sum(len(d["archives"]) for d in manifest["datasets"].values())
    row_count = sum(d["rows"] for d in manifest["datasets"].values())
    lines = ["# Theta Bot: spot-only 4h flow study — 2026-10-07", "",
        "**No profitability improvement. All six new variants lose after costs.**",
        f"The development-frozen choice is `{frozen}`; the historical improvement screen is **{result['historical_improvement_screen_passed']}**.",
        "No live trading, defaults, deployment or exchange orders are enabled.", "",
        "The binding constraint excludes leverage, loans, margin, shorts, derivatives, leveraged tokens and",
        "martingale loss-doubling. The live executor verifies a spot market and free spot funds before",
        "submission; unavailable information rejects an order. The planner forbids `allow_short=True`.",
        "Recorded fills reject cash borrowing and sales exceeding owned inventory, including fees.",
        "This fixes an old impossible-fill accounting path that could credit an oversized sale and then",
        "clamp inventory to zero. Cash/inventory clamping now handles numerical residue only.", "",
        "## Data and fixed design", "",
        f"{archive_count} official monthly spot archives provide {row_count:,} complete 4h bars",
        "for BTCUSDT, ETHUSDT and BNBUSDT, 2022-01-01 through 2026-09-30. Each published SHA-256",
        "was checked; timestamp-unit transitions, candle close times, OHLC and taker-volume fields were",
        "validated. No missing bars were filled. These are aggregated executed taker trades, not historical",
        "order-book depth, bid/ask imbalance or tick-level microstructure data.", "",
        "Read [the protocol](SPOT_FLOW_RESEARCH_PLAN.md) and [the spot-only policy](SPOT_ONLY_POLICY.md).",
        f"Protocol SHA-256: `{result['protocol_sha256']}`. Written before new economic results.",
        "Each signal uses completed bars and trades at the next open. A shared 1,000-USDT account sells",
        "before buying. All seven paths retain 50% aggregate/25% per-asset target caps, a 1% rebalance",
        "band and a 10-USDT minimum fill. None sizes from a loss-recovery multiplier.", "",
        "Nominal fills pay 0.1% fee +5bps slippage +half a 2bps spread on each side. Stress fills pay",
        "0.2% fee +10bps slippage with the same spread. Impact below includes slippage and half-spread;",
        "it is already in the fill price and is not debited twice. No account fee tier or measured fills",
        "are assumed. Open inventory stays marked to market at period ends.", "",
        "## Development choice, frozen before later replay", "",
        "Choose the largest positive 2022–2024 net profit with conservative 4h drawdown no worse than −20%.",
        "The selection file is written before any later path. All variants are disclosed below.", "",
        "| Variant | Development net | Development DD bound | Eligible |",
        "|---|---:|---:|:---:|"]
    for name, s in result["development"].items():
        eligible = s["net_pnl"] > 0 and s["max_intraday_drawdown_bound"] >= -.2
        lines.append(f"| `{name}` | {pct(s['total_return'])} | {pct(s['max_intraday_drawdown_bound'])} | {'yes' if eligible else 'no'} |")
    lines += ["", "## Net account results", "",
        "2025 and 2026 columns start with independent 1,000-USDT accounts. Continuous paths preserve",
        "all costs, losses and compounding. All dates have already been studied; they are not a fresh holdout.", "",
        "| Variant | 2025 net | Jan–Sep 2026 net | Continuous 2025–Sep 2026 | Full 2022–Sep 2026 | Full CAGR | Full DD bound | Full doubled-cost net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in variants:
        full = nominal["full_continuous"][name]
        lines.append(f"| `{name}` | {pct(nominal['2025'][name]['total_return'])} | {pct(nominal['2026'][name]['total_return'])} | "
            f"{pct(nominal['later_continuous'][name]['total_return'])} | {pct(full['total_return'])} | {pct(full['cagr'])} | "
            f"{pct(full['max_intraday_drawdown_bound'])} | {pct(doubled['full_continuous'][name]['total_return'])} |")
    control = nominal["full_continuous"]["spot_control"]
    lines += ["", f"Control equity reaches {control['final_equity']:.2f} USDT from 1,000 USDT over 1,734 days.",
        f"Its sampled 4h-close drawdown is {pct(control['maxDD'])}; conservative 4h high/low bound is",
        f"{pct(control['max_intraday_drawdown_bound'])}. These marks differ from daily-close drawdown.",
        "The high/low bound combines unfavorable cross-asset extrema and allows the high to precede",
        "the low within a candle; it is conservative rather than an exact tick-level drawdown.",
        "The 20% budget is a historical rejection threshold, not a future loss guarantee.", "",
        "## Costs and cash accounting", "",
        "| Variant, full nominal account | Fills | Turnover / initial capital | Fees USDT | Impact USDT | Net P&L USDT |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in variants:
        s = nominal["full_continuous"][name]
        lines.append(f"| `{name}` | {s['trades_count']} | {s['turnover']:.2f} | {s['fees_paid_total']:.2f} | {s['slippage_paid_total']:.2f} | {s['net_pnl']:+.2f} |")
    all_summaries = [*result["development"].values(),
                    *(s for c in result["later"].values() for p in c.values() for s in p.values()),
                    *result["extra_one_bar_delay_frozen"].values()]
    max_error = max(s["accounting"]["max_equity_error"] for s in all_summaries)
    lines += ["", f"All {len(all_summaries)} paths reconcile cash, inventory and equity at **every 4h close**",
        f"against an independent signed cash-flow ledger. Largest equity residual: {max_error:.3g} USDT.",
        "Every account retains nonnegative cash and inventory. The eight nominal/stress control",
        "comparisons reproduce the daily engine's daily-close equity and fill quantities/prices/fees/impact",
        "within 1e−8 absolute tolerance. This reproduces the earlier spot net profits on the same dates.", "",
        "Cost addback is a diagnostic, not a simulated no-cost account: removing costs would change later",
        "position sizes and compounding. Trading more frequently has not established an economic edge.", "",
        "## Weekly and monthly doubling", "",
        "Full nominal continuous account; windows use calendar-day closing NAV, including initial capital.",
        "Counts refer to rolling endpoint-to-endpoint doubling, not an intraday spike. Windows overlap and",
        "are historical observations, not independent estimates of future probability.", "",
        "| Variant | 7d doublings / windows | Best 7d | 30d doublings / windows | Best 30d | 90d doublings / windows | Best 90d |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in variants:
        r = nominal["full_continuous"][name]["rolling_gains"]
        cells = [f"`{name}`"]
        for days in ("7", "30", "90"):
            cells += [f"{r[days]['doublings']} / {r[days]['windows']}", pct(r[days]["maximum_return"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Timing stress and screen", "",
        "One extra 4h bar delays both the frozen signal and its scheduled decision. The control's daily",
        "decision therefore moves from midnight to 04:00 UTC; this does not accidentally become a",
        "whole-day delay. This is a hypothetical timing stress, not measured network or exchange latency.", "",
        "| Frozen candidate period | Extra 4h delay net | Extra 4h delay DD bound |",
        "|---|---:|---:|"]
    for label, s in result["extra_one_bar_delay_frozen"].items():
        lines.append(f"| {label} | {pct(s['total_return'])} | {pct(s['max_intraday_drawdown_bound'])} |")
    lines += ["", "| Historical improvement gate | Passed |", "|---|:---:|"]
    lines += [f"| {name} | {str(value).lower()} |" for name, value in result["gates"].items()]
    lines += ["", "The control cannot count as its own improvement. No new strategy passes the profit criteria.",
        "The flow-breakout filter reduces losses relative to the price breakout but remains unprofitable;",
        "the low-turnover reversion model also loses. This rejects these six fixed implementations, not",
        "all possible uses of executed flow or finer market data.", "",
        "A subsequent research direction is a cost-aware entry gate with a fixed holding interval and",
        "training-only calibration: trade only when predicted gain clears round-trip costs and uncertainty.",
        "It must compare matched price/flow models, preserve spot caps and be frozen before subsequent",
        "unseen/paper data. That gate and richer order-book data are **not tested or implemented here**.", "",
        "## Reproduction and limitations", "",
        "```bash", "python -m scripts.download_spot_flow",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_spot_flow",
        "python -m scripts.write_spot_flow_report",
        "python -m pytest -q tests", "```", "",
        "Raw ZIPs, checked source manifest, frozen selection and all equity/trade CSVs are written under",
        "`data/raw/spot_flow/` and `artifacts/spot_flow_20261007/` (ignored by Git). The adjacent committed",
        f"[{stem}.json]({stem}.json) includes every summary, source URL/hash and gate.",
        f"Evidence JSON SHA-256: `{hashlib.sha256(json_path.read_bytes()).hexdigest()}`.", ""]
    lines += [f"- {limit}." for limit in result["limitations"]]
    lines += ["", "Primary source documentation: [Binance archive schema and checksums](https://github.com/binance/binance-public-data)",
        "and [spot market-data endpoints](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints).", ""]
    path = out / (stem + ".md")
    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/spot_flow_20261007/summary.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/evaluation"))
    args = parser.parse_args()
    print(write_report(args.source, args.out))


if __name__ == "__main__":
    main()
