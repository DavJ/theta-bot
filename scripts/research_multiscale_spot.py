"""Run the fixed eight-policy spot comparison, with all-policy timing stress."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.download_spot_flow import BEGIN, END
from scripts.research_spot_flow import load_spot_flow, freeze_development, NOMINAL, STRESS, VALIDATION, TRANSFER
from scripts.research_momentum_refinement import fill_frequency
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.multiscale import multiscale_targets

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "docs/evaluation/MULTISCALE_SPOT_RESEARCH_PLAN.md"
PREVIOUS = REPO / "docs/evaluation/THETABOT_MOMENTUM_REFINEMENT_2026-10-09.json"
CONTROLS = {"spot_control": "spot_control", "capped_control": "capped_allocation"}


def run_research(markets, manifest, out):
    out.mkdir(parents=True, exist_ok=True)
    targets, masks, _ = multiscale_targets(markets)
    plan_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    previous = json.loads(PREVIOUS.read_text())
    signal_hashes = {}
    for name, target in targets.items():
        payload = target.to_csv()
        (out / f"{name}_completed_targets.csv").write_text(payload)
        signal_hashes[name] = {"target_sha256": hashlib.sha256(payload.encode()).hexdigest(), "buy_mask_sha256": None}
        if masks[name] is not None:
            payload = masks[name].to_csv()
            (out / f"{name}_completed_buy_mask.csv").write_text(payload)
            signal_hashes[name]["buy_mask_sha256"] = hashlib.sha256(payload.encode()).hexdigest()

    def replay(name, start, end, costs, label, delay=0):
        eq, fills, summary = run_intraday_spot(markets, targets[name], start=start, end=end,
            daily_control=True, signal_delay_bars=delay, completed_buy_mask=masks[name], **costs)
        summary["accounting"] = reconcile_spot_path(markets, eq, fills)
        summary["fill_frequency"] = fill_frequency(eq, fills)
        eq.to_csv(out / f"{name}_{label}_equity.csv", index=False)
        fills.to_csv(out / f"{name}_{label}_trades.csv", index=False)
        return summary

    development = {}
    for name in targets:
        development[name] = replay(name, BEGIN, VALIDATION, NOMINAL, "development")
        print(f"Development {name}: {development[name]['total_return']:.3%}; "
              f"DD {development[name]['max_intraday_drawdown_bound']:.3%}", flush=True)
    frozen = freeze_development(development)
    selection = {"frozen_candidate": frozen, "development": development,
        "criterion": "largest positive 2022–2024 net profit with conservative 4h DD >= -20%",
        "protocol_sha256": plan_hash}
    (out / "frozen_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print(f"Frozen before later economic replay: {frozen}", flush=True)
    periods = {"2025": (VALIDATION, TRANSFER), "2026": (TRANSFER, END),
               "later_continuous": (VALIDATION, END), "full_continuous": (BEGIN, END)}
    later = {cost: {p: {} for p in periods} for cost in ("nominal", "doubled")}
    control_checks = {}
    for cost_name, costs in (("nominal", NOMINAL), ("doubled", STRESS)):
        for label, (start, end) in periods.items():
            for name in targets:
                summary = replay(name, start, end, costs, f"{label}_{cost_name}")
                later[cost_name][label][name] = summary
                print(f"{label} {cost_name} {name}: {summary['total_return']:.3%}; "
                      f"DD {summary['max_intraday_drawdown_bound']:.3%}", flush=True)
                if name in CONTROLS:
                    old = previous["later"][cost_name][label][CONTROLS[name]]
                    for field in ("total_return", "max_intraday_drawdown_bound", "fees_paid_total", "slippage_paid_total", "trades_count"):
                        if not np.isclose(summary[field], old[field], rtol=1e-10, atol=1e-8):
                            raise ValueError(f"Previous {name} {field} changed")
                    control_checks[f"{name}_{label}_{cost_name}"] = True
    timing = {p: {} for p in periods}
    for label, (start, end) in periods.items():
        for name in targets:
            timing[label][name] = replay(name, start, end, NOMINAL, f"{label}_extra_delay", delay=1)
            print(f"{label} delayed {name}: {timing[label][name]['total_return']:.3%}; "
                  f"DD {timing[label][name]['max_intraday_drawdown_bound']:.3%}", flush=True)
    all_summaries = [*development.values(), *(s for c in later.values() for p in c.values() for s in p.values()),
                     *(s for p in timing.values() for s in p.values())]
    gates = {"development_eligible": frozen != "cash", "new_candidate": frozen not in (*CONTROLS, "cash"),
             "accounting": all(s["accounting"]["reconciled"] for s in all_summaries),
             "controls_unchanged": all(control_checks.values())}
    if frozen != "cash":
        nominal, doubled = later["nominal"], later["doubled"]
        gates.update({f"beats_both_controls_{p}": nominal[p][frozen]["total_return"] >
                      max(nominal[p][c]["total_return"] for c in CONTROLS) for p in periods})
        gates.update(full_nominal_drawdown=nominal["full_continuous"][frozen]["max_intraday_drawdown_bound"] >= -.2,
                     full_doubled_drawdown=doubled["full_continuous"][frozen]["max_intraday_drawdown_bound"] >= -.2,
                     full_delayed_drawdown=timing["full_continuous"][frozen]["max_intraday_drawdown_bound"] >= -.2,
                     later_doubled_positive=doubled["later_continuous"][frozen]["net_pnl"] > 0,
                     later_delayed_positive=timing["later_continuous"][frozen]["net_pnl"] > 0)
    result = {"protocol": str(PLAN.relative_to(REPO)), **selection, "source_manifest": manifest,
        "previous_evidence_sha256": hashlib.sha256(PREVIOUS.read_bytes()).hexdigest(),
        "signal_hashes": signal_hashes, "later": later, "extra_one_bar_delay": timing,
        "control_checks": control_checks, "gates": gates,
        "historical_improvement_screen_passed": bool(all(gates.values())), "live_eligible": False,
        "reconciled_paths": len(all_summaries), "costs": {"nominal": NOMINAL, "doubled": STRESS},
        "initial_usdt": 1000., "constraints": {"spot_only": True, "leverage": False, "borrowing": False,
            "shorts": False, "derivatives": False, "martingale": False, "target_cap": .5, "asset_cap": .25},
        "limitations": ["Previously studied periods; not a new unseen holdout",
            "Three surviving assets; universe survivorship not controlled",
            "Momentum, ranks and no-chase threshold are fixed heuristics, not calibrated forecasts or p-values",
            "Modeled fees/impact, not actual fills or account-specific fee tiers",
            "Overlapping rolling windows do not estimate independent future doubling probabilities",
            "Conservative within-bar high/low bound, not exact tick drawdown",
            "20% historical budget does not guarantee future maximum loss"]}
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != plan_hash:
        raise ValueError("Protocol changed during research")
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/spot_flow"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/multiscale_spot_20261009"))
    args = parser.parse_args()
    markets, manifest = load_spot_flow(args.data)
    run_research(markets, manifest, args.out)


if __name__ == "__main__":
    main()
