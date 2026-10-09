"""Write all multiscale results and disclose timing/cost failures without reselection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def pct(value):
    return f"{100 * value:+.2f}%"


def write_report(source, out):
    r = json.loads(source.read_text())
    out.mkdir(parents=True, exist_ok=True)
    stem = "THETABOT_MULTISCALE_SPOT_2026-10-09"
    json_path = out / (stem + ".json")
    json_path.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    nom, stress, delay = r["later"]["nominal"], r["later"]["doubled"], r["extra_one_bar_delay"]
    frozen, names, passed = r["frozen_candidate"], list(r["development"]), r["historical_improvement_screen_passed"]
    conclusion = ("The frozen new policy passes the fixed historical profit/cost/timing screen."
                  if passed else "No frozen new policy passes the fixed historical profit/cost/timing screen.")
    lines = ["# Theta Bot: multiscale spot momentum — 2026-10-09", "", f"**{conclusion}**",
        f"Development-frozen choice: `{frozen}`. Live eligibility: **false**.",
        "Dates and the surviving three-asset universe were studied previously; results remain exploratory.",
        "No live orders, default changes, deployment or PR merge are enabled.", "",
        "## Fixed hypothesis and execution", "",
        "Two controls plus six fixed changes: 14/30/60-day momentum agreement, dual 30/90-day",
        "agreement, weekly top-two ranking by 30-day or median multihorizon standardized strength,",
        "and an actual-buy restriction after a sharp seven-day rise on the capped/consensus policies.",
        "Read [the protocol](MULTISCALE_SPOT_RESEARCH_PLAN.md) and [spot-only policy](SPOT_ONLY_POLICY.md).",
        f"Protocol SHA-256: `{r['protocol_sha256']}`; recorded before new economic results.", "",
        "The same 171 checked official BTC/ETH/BNB spot archives supply 31,212 complete 4h bars",
        "(January 2022–September 2026). Daily targets and permissions are published only with the",
        "completed day's final 4h candle. The engine shifts both to the following open. Weekly ranks",
        "are selected at completed Sunday closes; lost momentum eligibility can exit daily, but a",
        "new ranked member waits until the next weekly selection. Ties follow stable column order.", "",
        "One funded 1,000-USDT account, 50% aggregate/25% asset target caps, Monday ordinary",
        "rebalance, 1pt band and 10-USDT minimum fill. Daily inactive exits and cap sales bypass",
        "the ordinary band and buy restriction. Forbidden increases remain cash. The permission",
        "compares targets with actual owned inventory and also applies to additions after entry.",
        "The optional buy mask defaults to unrestricted buys and preserves previous control results.",
        "No leverage, borrowing, margin, financial shorts, derivatives or martingale is introduced.", "",
        "The six hypotheses are fixed heuristics, not calibrated future net-profit forecasts or p-values.",
        "No new grid, later-period selection or post-result variant is used. Continuous paths preserve",
        "peaks, losses, costs and compounding. Inventory is marked to market at period ends.", "",
        "Nominal per-fill costs: 0.1% fee +5bps slippage +half a 2bps spread. Stress doubles fees and",
        "slippage, leaving spread unchanged. Impact is already in the fill price, not debited twice.",
        "Every variant also receives the same extra 4h target/permission delay, moving decisions to",
        "04:00 UTC. This hypothetical timing stress is not an estimate of measured latency.", "",
        "## Development selection", "",
        "Highest positive continuous 2022–2024 profit under a -20% conservative 4h drawdown bound,",
        "including both controls. The frozen choice is persisted before later-period economic replay.", "",
        "| Variant | Development net | Development DD bound | Eligible |", "|---|---:|---:|:---:|"]
    for name, s in r["development"].items():
        eligible = s["net_pnl"] > 0 and s["max_intraday_drawdown_bound"] >= -.2
        lines.append(f"| `{name}` | {pct(s['total_return'])} | {pct(s['max_intraday_drawdown_bound'])} | {'yes' if eligible else 'no'} |")
    lines += ["", "## Net account results", "",
        "2025 and Jan–Sep 2026 begin independent accounts. Continuous columns carry all results",
        "forward. These are fresh accounts on known dates, not new unseen holdout periods.", "",
        "| Variant | 2025 net | Jan–Sep 2026 net | Continuous 2025–Sep 2026 | Full 2022–Sep 2026 | Full CAGR | Full DD bound | Full doubled-cost net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        s = nom["full_continuous"][name]
        lines.append(f"| `{name}` | {pct(nom['2025'][name]['total_return'])} | {pct(nom['2026'][name]['total_return'])} | "
            f"{pct(nom['later_continuous'][name]['total_return'])} | {pct(s['total_return'])} | {pct(s['cagr'])} | "
            f"{pct(s['max_intraday_drawdown_bound'])} | {pct(stress['full_continuous'][name]['total_return'])} |")
    best = max(names, key=lambda n: nom["full_continuous"][n]["total_return"])
    s = nom["full_continuous"][best]
    lines += ["", f"The descriptive full-nominal leader is `{best}`: terminal NAV {s['final_equity']:.2f} USDT",
        f"from 1,000 USDT, close-only DD {pct(s['maxDD'])}, conservative high/low bound {pct(s['max_intraday_drawdown_bound'])}.",
        "This description does not replace the development-frozen choice or bypass any stress gate.",
        "The high/low bound assumes unfavorable intracandle/cross-asset ordering, not exact tick",
        "chronology. The 20% historical threshold does not guarantee a maximum future loss.", "",
        "## Costs and trading frequency", "",
        "One fill is one executed buy/sell leg, not a completed round trip. Full nominal accounts.", "",
        "| Variant | Fills | Active days | Fills/month | Turnover / initial capital | Fees USDT | Impact USDT | Net P&L USDT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        s = nom["full_continuous"][name]
        f = s["fill_frequency"]
        lines.append(f"| `{name}` | {s['trades_count']} | {f['active_trading_days']} | {f['fills_per_month']:.2f} | "
            f"{s['turnover']:.2f} | {s['fees_paid_total']:.2f} | {s['slippage_paid_total']:.2f} | {s['net_pnl']:+.2f} |")
    summaries = [*r["development"].values(), *(s for c in r["later"].values() for p in c.values() for s in p.values()),
                 *(s for p in delay.values() for s in p.values())]
    error = max(s["accounting"]["max_equity_error"] for s in summaries)
    lines += ["", f"All {r['reconciled_paths']} paths independently reconcile cash, inventory and NAV at every 4h close.",
        f"Largest NAV residual: {error:.3g} USDT. Sixteen nominal/stress checks reproduce both controls'",
        "previous profits, drawdown, costs and fill counts within 1e-8 absolute tolerance. Minimum cash",
        "and inventory are nonnegative. Cost addback is diagnostic, not a simulated no-cost account.", "",
        "## Every-policy timing and cost-risk stress", "",
        "No descriptive leader is exempt from the one-bar delay. A full delayed drawdown above 20%",
        "fails this round's historical screen even when nominal profits are higher.", "",
        "| Variant | Delayed 2025 net | Delayed 2026 net | Delayed later net | Delayed full net | Delayed full DD | Doubled-cost full DD |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        lines.append(f"| `{name}` | " + " | ".join(pct(delay[p][name]["total_return"])
            for p in ("2025", "2026", "later_continuous", "full_continuous")) +
            f" | {pct(delay['full_continuous'][name]['max_intraday_drawdown_bound'])} | "
            f"{pct(stress['full_continuous'][name]['max_intraday_drawdown_bound'])} |")
    lines += ["", "## Rolling short-horizon gains", "",
        "Full nominal continuous account, calendar-day closing endpoints including initial capital.",
        "Windows overlap and are historical observations, not independent future probabilities.", "",
        "| Variant | 7d doublings/windows | Best 7d | 30d doublings/windows | Best 30d | 90d doublings/windows | Best 90d |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        rolling = nom["full_continuous"][name]["rolling_gains"]
        cells = [f"`{name}`"]
        for days in ("7", "30", "90"):
            cells += [f"{rolling[days]['doublings']}/{rolling[days]['windows']}", pct(rolling[days]["maximum_return"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Frozen improvement gates", "", "| Gate | Passed |", "|---|:---:|"]
    lines += [f"| {name} | {str(value).lower()} |" for name, value in r["gates"].items()]
    lines += ["", f"Historical screen: **{passed}**. {conclusion}",
        "Selecting a control again cannot count as a new improvement. Passing code tests does not",
        "prove profit, and even a passed historical screen on reused data would need unseen/paper confirmation.", "",
        "## Reproduction and limitations", "", "```bash", "python -m scripts.download_spot_flow",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_multiscale_spot",
        "python -m scripts.write_multiscale_spot_report", "python -m pytest -q tests", "```", "",
        "Checked archives/manifest, frozen selection, completed targets/buy permissions and every",
        "equity/fill CSV are reproducible under `data/raw/spot_flow/` and `artifacts/multiscale_spot_20261009/`",
        f"(ignored by Git). [{stem}.json]({stem}.json) retains all summaries, source URLs/hashes, signal",
        "hashes, control checks and gates.", f"Evidence SHA-256: `{hashlib.sha256(json_path.read_bytes()).hexdigest()}`.", ""]
    lines += [f"- {limitation}." for limitation in r["limitations"]]
    lines += ["", "Primary motivation: [time-series cryptocurrency momentum](https://www.nber.org/papers/w24877)",
        "and [cross-sectional cryptocurrency factors](https://www.nber.org/papers/w25882). Their portfolios",
        "and results are not replicated here; their long-short strategies are excluded by the spot-only policy.",
        "Data schema/checksums: https://github.com/binance/binance-public-data .", ""]
    path = out / (stem + ".md")
    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/multiscale_spot_20261009/summary.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/evaluation"))
    args = parser.parse_args()
    print(write_report(args.source, args.out))


if __name__ == "__main__":
    main()
