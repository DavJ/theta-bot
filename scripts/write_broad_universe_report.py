"""Publish every fixed broad-cohort result, including failures and data/source audits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def pct(value):
    return "n/a" if value is None else f"{value * 100:+.2f}%"


def write_report(source, out):
    r = json.loads(source.read_text())
    out.mkdir(parents=True, exist_ok=True)
    stem = "THETABOT_BROAD_UNIVERSE_2026-10-09"
    evidence = out / (stem + ".json")
    evidence.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    nom, stress, delay = r["later"]["nominal"], r["later"]["doubled"], r["extra_one_bar_delay"]
    names = list(r["development"])
    frozen = r["frozen_candidate"]
    passed = r["historical_improvement_screen_passed"]
    material = r["material_improvement_screen_passed"]
    manifest = r["source_manifest"]
    archives = [a for d in manifest["datasets"].values() for a in d["archives"]]
    repairs = [x for a in archives for x in a["duplicate_corroboration"]]
    conclusion = ("The fixed material historical screen passes; unseen/paper confirmation is still required."
                  if material else "A material improvement is NOT supported by this experiment.")
    lines = ["# Theta Bot — fixed historical broad spot universe", "",
        f"**{conclusion}** Development froze `{frozen}` before later account replay.",
        f"Historical improvement screen: **{str(passed).lower()}**; material screen: **{str(material).lower()}**.",
        "Live eligibility remains false. Owned spot only; no leverage, borrowing, margin, shorts,",
        "derivatives, leveraged tokens or martingale. This study does not enable orders or change defaults.", "",
        "## What changed and what was fixed", "",
        "Expand from BTC/ETH/BNB to 15 underlying assets selected from the dated 2021-12-26 top-20",
        "market-cap snapshot, with a verified December 2021 spot-USDT source requirement. Retain old",
        "Terra's collapse and trading absence, distinguish it from new Terra 2.0, and follow the",
        "documented 1:1 MATIC/POL identity migration. CRO's source is HTTP 404 and is not replaced.",
        "Stablecoins and wrapped BTC are excluded. This fixed historical cohort is not the whole market.", "",
        "Six new policies combine completed-day liquidity, momentum ranks, inverse volatility and",
        "a reduction-only volatility budget. New assets need 90 contiguous complete days, mean quote",
        "volume >=10m USDT/day, and historical top-ten cohort liquidity. Sunday ranks execute through",
        "Monday rebalances; daily inactivity/cap reductions are allowed. Controls retain the old rules.", "",
        "Shared 1,000-USDT accounts, 50% aggregate/25% asset target caps, 1pt rebalance band, 10-USDT",
        "minimum fills and sell-before-buy execution. Targets published at the completed 20h candle",
        "execute at the next 00h open; delay stress executes at 04h. Exposure can drift between decisions.",
        "Nominal fills cost 0.1% fee +5bps slippage +half a 2bps spread; cost stress doubles fee/slippage.",
        "Open inventory remains marked at period ends. Later/full accounts preserve compounding and peaks.", "",
        f"[Fixed protocol](BROAD_UNIVERSE_RESEARCH_PLAN.md), SHA-256 `{r['protocol_sha256']}`.",
        f"Frozen UTC timestamp: `{r['frozen_at']}`; source-manifest SHA-256 `{r['source_manifest_sha256']}`.",
        "Its documented source-validation amendment was written before economic replay and changed no",
        "strategy or economic threshold. Periods were already used in previous rounds; this is not an unseen holdout.", "",
        "## Development selection: 2022–2024", "",
        "Positive net, conservative 4h DD within 20% and zero held-unquoted bars; then highest net.", "",
        "| Policy | Net | Conservative DD | Held-unquoted asset/bars | Eligible | Frozen |",
        "|---|---:|---:|---:|:---:|:---:|"]
    for name in names:
        s = r["development"][name]
        eligible = s["net_pnl"] > 0 and s["max_intraday_drawdown_bound"] >= -.2 and not s["held_unquoted_asset_bars"]
        lines.append(f"| `{name}` | {pct(s['total_return'])} | {pct(s['max_intraday_drawdown_bound'])} | "
            f"{s['held_unquoted_asset_bars']} | {str(eligible).lower()} | {str(name == frozen).lower()} |")
    lines += ["", "## Net results after modeled costs", "",
        "2025 and Jan–Sep 2026 each start a fresh account. Later is one continuous Jan 2025–Sep 2026",
        "account; full is one continuous Jan 2022–Sep 2026 account, not a sum of reset periods.", "",
        "| Policy | 2025 net | Jan–Sep 2026 net | Later net | Full net | Full CAGR | Full close DD | Full conservative DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        s = nom["full_continuous"][name]
        cells = [f"`{name}`", pct(nom["2025"][name]["total_return"]), pct(nom["2026"][name]["total_return"]),
            pct(nom["later_continuous"][name]["total_return"]), pct(s["total_return"]), pct(s["cagr"]),
            pct(s["maxDD"]), pct(s["max_intraday_drawdown_bound"])]
        lines.append("| " + " | ".join(cells) + " |")
    if frozen != "cash":
        s = nom["full_continuous"][frozen]
        baseline = nom["full_continuous"]["capped_control"]
        lines += ["", f"The frozen `{frozen}` ends at {s['final_equity']:.2f} USDT versus capped control's",
            f"{baseline['final_equity']:.2f} USDT. Its later-continuous net is "
            f"{pct(nom['later_continuous'][frozen]['total_return'])} versus "
            f"{pct(nom['later_continuous']['capped_control']['total_return'])}.",
            "A small development advantage does not establish a later profit advantage. No later",
            "leader is substituted for the frozen choice; all rejected policies stay visible."]
    lines += ["", "## Every-policy cost and timing stress", "",
        "| Policy | Doubled-cost later net | Doubled-cost full net | Doubled-cost full DD | Delayed later net | Delayed full net | Delayed full DD |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        cells = [f"`{name}`", pct(stress["later_continuous"][name]["total_return"]),
            pct(stress["full_continuous"][name]["total_return"]), pct(stress["full_continuous"][name]["max_intraday_drawdown_bound"]),
            pct(delay["later_continuous"][name]["total_return"]), pct(delay["full_continuous"][name]["total_return"]),
            pct(delay["full_continuous"][name]["max_intraday_drawdown_bound"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "The high/low bound assumes unfavorable within-bar/cross-asset ordering; it is not",
        "exact tick chronology. The 20% historical selection budget is not a future maximum-loss guarantee.", "",
        "## Costs, turnover and fill frequency", "",
        "A fill is one buy/sell leg, not a completed round trip. Full nominal accounts.", "",
        "| Policy | Fills | Active days | Fills/month | Turnover / initial | Fees USDT | Impact USDT | Net P&L USDT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        s, f = nom["full_continuous"][name], nom["full_continuous"][name]["fill_frequency"]
        lines.append(f"| `{name}` | {s['trades_count']} | {f['active_trading_days']} | {f['fills_per_month']:.2f} | "
            f"{s['turnover']:.2f} | {s['fees_paid_total']:.2f} | {s['slippage_paid_total']:.2f} | {s['net_pnl']:+.2f} |")
    summaries = [*r["development"].values(), *(s for c in r["later"].values() for p in c.values() for s in p.values()),
                 *(s for p in delay.values() for s in p.values())]
    error = max(s["accounting"]["max_equity_error"] for s in summaries)
    lines += ["", f"All {r['reconciled_paths']} paths independently reconcile signed cash flows, owned inventory",
        f"and NAV at every 4h close; largest residual {error:.3g} USDT. Sixteen comparisons reproduce",
        "both preceding nominal/cost controls' net, CAGR, DD, fees/impact and fill counts within 1e-8.",
        "Cash/inventory stay nonnegative. Cost addback is diagnostic, not a no-cost account replay.", "",
        "## Source, identity and missing-market audit", "",
        f"{len(archives)} monthly archives plus 15 pre-period proofs are SHA-256 checked. "
        f"{sum(d['observed_rows'] for d in manifest['datasets'].values()):,} genuine bars and "
        f"{sum(d['missing_rows'] for d in manifest['datasets'].values())} absent asset/bars on the common grid.",
        "Two documented terminal partial candles retain their actual halt closes. The AVAX duplicate",
        "is identical in every field and corroborated by the entire day in a separate checksum-verified",
        f"daily archive ({len(repairs)} corroboration artifact). Both artifacts repeat the same row; this",
        "is not an independent underlying market measurement. Raw hashes and row counts remain in JSON.", "",
        "| Underlying research ID | Genuine bars | Absent bars | Absent ranges |",
        "|---|---:|---:|---|"]
    for asset, d in manifest["datasets"].items():
        missing = "; ".join(f"{x['first']} through {x['last']} ({x['bars']})" for x in d["missing_ranges"]) or "none"
        lines.append(f"| `{asset}` | {d['observed_rows']} | {d['missing_rows']} | {missing} |")
    held_paths = sum(s["held_unquoted_asset_bars"] > 0 for s in summaries)
    lines += ["", "Source/indicator gaps stay NaN; only a flagged replay view uses zero sentinel marks.",
        "No missing market can execute a fill. Owned quantity/cash persist; a held missing market is",
        "conservatively valued at zero until actual quotes resume. This pessimistic assumption is not",
        f"a real observed price. {held_paths}/{len(summaries)} paths hold any unquoted inventory; any such",
        "frozen-policy path would fail quality, irrespective of final profit. The known Polygon notice",
        "takes effect only after the completed announcement day; missing days reset 90-day readiness.", "",
        "## Short-horizon account gains", "",
        "Full nominal continuous account with initial capital included. Overlapping calendar windows",
        "are historical observations, not independent probabilities or future return promises.", "",
        "| Policy | 7d doublings/windows | Best 7d | 30d doublings/windows | Best 30d | 90d doublings/windows | Best 90d |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        cells = [f"`{name}`"]
        for days in ("7", "30", "90"):
            g = nom["full_continuous"][name]["rolling_gains"][days]
            cells.extend([f"{g['doublings']}/{g['windows']}", pct(g["maximum_return"])])
        lines.append("| " + " | ".join(cells) + " |")
    b = r["descriptive_paired_bootstrap"]
    lines += ["", "## Descriptive uncertainty and frozen gates", ""]
    if b:
        lines += [f"Frozen versus capped control, continuous later nominal daily log-return differences:",
            f"annualized mean log excess {pct(b['annualized_mean_log_excess'])}; 99% circular 14-day-block",
            f"percentile interval [{pct(b['lower'])}, {pct(b['upper'])}], {b['replicates']} replicates, seed {b['seed']}.",
            "A positive lower bound is an additional historical gate. This interval is descriptive:",
            "periods were already inspected, multiple rounds and universe selection are not accounted",
            "for, and it is not confirmatory statistical significance. Unseen/paper evidence is required.", ""]
    lines += ["| Gate | Passed |", "|---|:---:|"]
    lines.extend(f"| {name} | {str(value).lower()} |" for name, value in r["gates"].items())
    lines.extend(f"| Material: {name} | {str(value).lower()} |" for name, value in r["material_gates"].items())
    baseline = nom["full_continuous"]["capped_control"]
    lines += ["", "The material threshold additionally requires full CAGR at least "
        f"{pct(1.5 * baseline['cagr'])} and later net at least "
        f"{pct(1.5 * nom['later_continuous']['capped_control']['total_return'])}.",
        conclusion, "",
        "## Reproduction and evidence", "", "```bash", "python -m scripts.download_spot_universe",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_broad_universe",
        "python -m scripts.write_broad_universe_report", "python -m pytest -q tests", "```", "",
        "Checked raw source/cache/manifest: `data/raw/spot_universe/`. Frozen selection, targets, quote",
        "mask and every equity/fill CSV: `artifacts/broad_spot_20261009/` (Git ignored). The original",
        "three-asset source cache is reused without changing its checksums.",
        f"[{stem}.json]({stem}.json) retains every account summary, source URL/hash, alias rule, audit,",
        f"signal hash and gate; evidence SHA-256 `{hashlib.sha256(evidence.read_bytes()).hexdigest()}`.", ""]
    lines.extend(f"- {limitation}." for limitation in r["limitations"])
    lines += ["", f"Historical cohort: {manifest['snapshot']}",
        "Primary identity references are linked in the protocol and stored in evidence; archive schema:",
        "https://github.com/binance/binance-public-data .", ""]
    report = out / (stem + ".md")
    report.write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/broad_spot_20261009/summary.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/evaluation"))
    args = parser.parse_args()
    print(write_report(args.source, args.out))


if __name__ == "__main__":
    main()
