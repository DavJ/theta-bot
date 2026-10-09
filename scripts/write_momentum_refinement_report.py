"""Publish every fixed momentum result, including rejected variants and costs."""
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
    stem = "THETABOT_MOMENTUM_REFINEMENT_2026-10-09"
    json_path = out / (stem + ".json")
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    nom, stress = result["later"]["nominal"], result["later"]["doubled"]
    frozen = result["frozen_candidate"]
    names = list(result["development"])
    passed = result["historical_improvement_screen_passed"]
    conclusion = ("The development-frozen new variant passes the fixed historical improvement screen."
                  if passed else "No improvement passes the fixed development-frozen screen.")
    lines = ["# Theta Bot: momentum refinement — 2026-10-09", "", f"**{conclusion}**",
        f"Frozen choice: `{frozen}`. Live eligibility: **false**.",
        "The previously studied dates are reused; this is exploratory evidence, not a new unseen holdout.",
        "No live order, default strategy, deployment or PR merge is enabled.", "",
        "## Fixed design and execution", "",
        "Six fixed changes to the existing 30-day momentum control: wider ordinary trading band,",
        "three-day entry confirmation, stronger entry threshold, strength plus wider band, BTC regime",
        "filter, and capped weight redistribution. Read [the frozen protocol](MOMENTUM_REFINEMENT_RESEARCH_PLAN.md)",
        "and [the binding spot-only policy](SPOT_ONLY_POLICY.md). The implementation depends on PR #121.",
        f"Protocol SHA-256: `{result['protocol_sha256']}`; written before new economic results.", "",
        "The same 171 checked official spot archives supply 31,212 complete 4h bars for BTC, ETH and BNB",
        "from January 2022 through September 2026. No external predictor or missing-bar interpolation",
        "is added. Completed daily signals are published with the day's last 4h close and used at the",
        "next daily open. Ordinary rebalancing is Monday; inactive exits and cap sales bypass the",
        "ordinary band. The wider band is 3 percentage points, versus the control's 1 percentage point.", "",
        "All targets retain 50% aggregate/25% per-asset caps in one shared 1,000-USDT account. Holdings",
        "can drift between daily decisions. Sells precede buys, cash/inventory remain nonnegative",
        "including fees, and no loss-doubling, leverage, borrowing, short or derivative is used.",
        "Redistribution can deploy more of the same permitted budget; it does not raise the caps.",
        "The entry state is signal eligibility, not a record of filled holdings. The strength threshold",
        "is a fixed past-return heuristic, not a predicted future net gain or significance test.", "",
        "Nominal costs per fill: 0.1% fee +5bps slippage +half a 2bps spread. Stress doubles the fee and",
        "slippage, with spread unchanged. Impact is already in the fill price and is not charged twice.",
        "Period-end inventory remains marked to market. Peaks and losses persist on continuous paths.", "",
        "## Development choice", "",
        "Choose the highest positive continuous 2022–2024 net profit with conservative 4h drawdown",
        "no worse than -20%, including the control. Persist the choice before later economic replay.", "",
        "| Variant | Development net | Development DD bound | Eligible |", "|---|---:|---:|:---:|"]
    for name, summary in result["development"].items():
        eligible = summary["net_pnl"] > 0 and summary["max_intraday_drawdown_bound"] >= -.2
        lines.append(f"| `{name}` | {pct(summary['total_return'])} | {pct(summary['max_intraday_drawdown_bound'])} | {'yes' if eligible else 'no'} |")
    lines += ["", "## Net account results", "",
        "2025 and Jan–Sep 2026 start independent accounts. The two continuous columns carry all",
        "losses, fills and compounding forward. Later results cannot select a replacement winner.", "",
        "| Variant | 2025 net | Jan–Sep 2026 net | Continuous 2025–Sep 2026 | Full 2022–Sep 2026 | Full CAGR | Full DD bound | Full stress net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        full = nom["full_continuous"][name]
        lines.append(f"| `{name}` | {pct(nom['2025'][name]['total_return'])} | {pct(nom['2026'][name]['total_return'])} | "
            f"{pct(nom['later_continuous'][name]['total_return'])} | {pct(full['total_return'])} | {pct(full['cagr'])} | "
            f"{pct(full['max_intraday_drawdown_bound'])} | {pct(stress['full_continuous'][name]['total_return'])} |")
    if frozen != "cash":
        full, control = nom["full_continuous"][frozen], nom["full_continuous"]["spot_control"]
        later_gain = nom["later_continuous"][frozen]["total_return"] - nom["later_continuous"]["spot_control"]["total_return"]
        lines += ["", f"Frozen `{frozen}` ends at {full['final_equity']:.2f} USDT; control at {control['final_equity']:.2f} USDT.",
            f"Its continuous later return differs from control by {100 * later_gain:+.2f} percentage points.",
            f"Full 4h-close DD is {pct(full['maxDD'])}, conservative high/low bound {pct(full['max_intraday_drawdown_bound'])}."]
    lines += ["", "The high/low bound assumes unfavorable cross-asset/intracandle ordering; it is not exact",
        "tick drawdown. The 20% historical threshold does not guarantee a maximum future loss.", "",
        "The descriptive `capped_allocation` comparison improves all four nominal period returns",
        "relative to control. It uses the same signal with more complete deployment within the same",
        "caps, rather than an additional predictive source. It was not the development-frozen choice",
        "and cannot replace the failed BTC-regime selection after inspecting later data. Its higher",
        "historical profit is an exploratory finding that needs subsequent unseen/paper confirmation.",
        "The prescribed timing stress covers the frozen choice and control, not this later leader.", "",
        "## Costs and trading frequency", "",
        "Full nominal continuous account. A fill is one executed buy/sell leg, not a round trip.", "",
        "| Variant | Fills | Active trading days | Fills/month | Turnover / initial capital | Fees USDT | Impact USDT | Net P&L USDT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        summary = nom["full_continuous"][name]
        frequency = summary["fill_frequency"]
        lines.append(f"| `{name}` | {summary['trades_count']} | {frequency['active_trading_days']} | {frequency['fills_per_month']:.2f} | "
            f"{summary['turnover']:.2f} | {summary['fees_paid_total']:.2f} | {summary['slippage_paid_total']:.2f} | {summary['net_pnl']:+.2f} |")
    summaries = [*result["development"].values(), *(s for c in result["later"].values() for p in c.values() for s in p.values()),
                 *(s for c in result["extra_one_bar_delay"].values() for s in c.values())]
    max_error = max(s["accounting"]["max_equity_error"] for s in summaries)
    lines += ["", f"All {result['reconciled_paths']} paths independently reconcile cash, owned inventory and NAV at every 4h close.",
        f"Largest NAV residual: {max_error:.3g} USDT. The eight control checks reproduce the preceding",
        "report's profits/costs/DD and the original daily engine's equity and fill quantities/prices/fees/impact",
        "within 1e-8 absolute tolerance. No borrowing or financial short is hidden by inventory clamping.",
        "Cost addback is diagnostic; it is not the result of a hypothetical no-cost replay.", "",
        "## Short-horizon gains", "",
        "Continuous full nominal account, rolling calendar-day closing endpoints including initial NAV.",
        "Windows overlap; counts are historical observations, not independent future probabilities.", "",
        "| Variant | 7d doublings / windows | Best 7d | 30d doublings / windows | Best 30d | 90d doublings / windows | Best 90d |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        rolling = nom["full_continuous"][name]["rolling_gains"]
        cells = [f"`{name}`"]
        for days in ("7", "30", "90"):
            cells += [f"{rolling[days]['doublings']} / {rolling[days]['windows']}", pct(rolling[days]["maximum_return"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## One-bar timing stress and improvement gates", "",
        "An extra 4h signal delay moves both target availability and decisions to 04:00 UTC.",
        "The frozen choice and control are delayed identically; this is a hypothetical timing stress,",
        "not a measured network delay. Timing results do not choose another winner.", "",
        "| Variant | Period | Delayed net | Delayed DD bound |", "|---|---|---:|---:|"]
    for name, periods in result["extra_one_bar_delay"].items():
        for label, summary in periods.items():
            lines.append(f"| `{name}` | {label} | {pct(summary['total_return'])} | {pct(summary['max_intraday_drawdown_bound'])} |")
    lines += ["", "| Development-frozen improvement gate | Passed |", "|---|:---:|"]
    lines += [f"| {name} | {str(value).lower()} |" for name, value in result["gates"].items()]
    lines += ["", f"Historical improvement screen: **{passed}**. {conclusion}",
        "A control selected again cannot count as an improvement. Any later-period leader is descriptive",
        "unless it was also development-frozen and passes every fixed gate. No live eligibility is inferred.", "",
        "## Reproduction and limitations", "", "```bash", "python -m scripts.download_spot_flow",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_momentum_refinement",
        "python -m scripts.write_momentum_refinement_report", "python -m pytest -q tests", "```", "",
        "Source archives, manifest, frozen selection and every equity/trade CSV are reproducible under",
        "`data/raw/spot_flow/` and `artifacts/momentum_refinement_20261009/` (ignored by Git). This committed",
        f"[{stem}.json]({stem}.json) retains all summaries, source URLs/hashes, gates and control checks.",
        f"Evidence JSON SHA-256: `{hashlib.sha256(json_path.read_bytes()).hexdigest()}`.", ""]
    lines += [f"- {limitation}." for limitation in result["limitations"]]
    lines += ["", "Primary documentation: [Binance public spot archive schema/checksums](https://github.com/binance/binance-public-data).", ""]
    path = out / (stem + ".md")
    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/momentum_refinement_20261009/summary.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/evaluation"))
    args = parser.parse_args()
    print(write_report(args.source, args.out))


if __name__ == "__main__":
    main()
