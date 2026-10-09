"""Replay the fixed momentum refinements, freeze development choice, reconcile cash."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.download_spot_flow import BEGIN, END
from scripts.research_spot_flow import load_spot_flow, freeze_development, NOMINAL, STRESS, VALIDATION, TRANSFER
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.backtest.portfolio import run_portfolio_backtest
from spot_bot.portfolio.momentum_refinement import MomentumRefinement, momentum_refinement_targets
from spot_bot.portfolio.trend import TrendPortfolio

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "docs/evaluation/MOMENTUM_REFINEMENT_RESEARCH_PLAN.md"
PREVIOUS = REPO / "docs/evaluation/THETABOT_SPOT_FLOW_2026-10-07.json"


def fill_frequency(equity, fills):
    """Individual fills, not completed round trips, across the whole portfolio."""
    daily_dates = pd.DatetimeIndex(equity.timestamp).normalize().unique()
    filled_dates = pd.DatetimeIndex(fills.timestamp).normalize().unique()
    month_keys = daily_dates.strftime("%Y-%m")
    all_months = pd.Index(month_keys).unique()
    per_month = (pd.Series(pd.DatetimeIndex(fills.timestamp).strftime("%Y-%m"))
                 .value_counts().reindex(all_months, fill_value=0))
    return {"calendar_days": len(daily_dates), "active_trading_days": len(filled_dates),
            "fills_per_week": len(fills) / (len(daily_dates) / 7),
            "fills_per_month": len(fills) / len(all_months),
            "median_monthly_fills": float(per_month.median()),
            "minimum_monthly_fills": int(per_month.min()),
            "maximum_monthly_fills": int(per_month.max())}


def run_research(markets, manifest, out):
    out.mkdir(parents=True, exist_ok=True)
    targets, daily = momentum_refinement_targets(markets)
    plan_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    previous = json.loads(PREVIOUS.read_text())

    def replay(name, start, end, costs, label, delay=0):
        policy = MomentumRefinement(name)
        eq, fills, summary = run_intraday_spot(markets, targets[name], start=start, end=end,
            daily_control=True, signal_delay_bars=delay, rebalance_band=policy.rebalance_band, **costs)
        summary["accounting"] = reconcile_spot_path(markets, eq, fills)
        summary["fill_frequency"] = fill_frequency(eq, fills)
        eq.to_csv(out / f"{name}_{label}_equity.csv", index=False)
        fills.to_csv(out / f"{name}_{label}_trades.csv", index=False)
        return summary, eq, fills

    development = {}
    for name in targets:
        development[name], _, _ = replay(name, BEGIN, VALIDATION, NOMINAL, "development")
        print(f"Development {name}: {development[name]['total_return']:.3%}; "
              f"DD {development[name]['max_intraday_drawdown_bound']:.3%}", flush=True)
    frozen = freeze_development(development)
    selection = {"frozen_candidate": frozen, "development": development,
        "criterion": "largest positive 2022–2024 net profit with conservative 4h DD >= -20%",
        "protocol_sha256": plan_hash}
    # Later economic replay must follow this persisted choice, never precede it.
    (out / "frozen_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print(f"Frozen before later economic replay: {frozen}", flush=True)
    periods = {"2025": (VALIDATION, TRANSFER), "2026": (TRANSFER, END),
               "later_continuous": (VALIDATION, END), "full_continuous": (BEGIN, END)}
    later = {cost: {period: {} for period in periods} for cost in ("nominal", "doubled")}
    control_comparison = {}
    for cost_name, costs in (("nominal", NOMINAL), ("doubled", STRESS)):
        for label, (start, end) in periods.items():
            for name in targets:
                summary, eq, fills = replay(name, start, end, costs, f"{label}_{cost_name}")
                later[cost_name][label][name] = summary
                print(f"{label} {cost_name} {name}: {summary['total_return']:.3%}; "
                      f"DD {summary['max_intraday_drawdown_bound']:.3%}", flush=True)
                if name == "spot_control":
                    old = previous["later"][cost_name][label][name]
                    for field in ("total_return", "max_intraday_drawdown_bound", "fees_paid_total",
                                  "slippage_paid_total", "trades_count"):
                        if not np.isclose(summary[field], old[field], rtol=1e-10, atol=1e-8):
                            raise ValueError(f"Preceding report's control {field} changed")
                    eq_d, fills_d, daily_summary = run_portfolio_backtest(daily,
                        TrendPortfolio("momentum_vol", max_exposure=.5, asset_cap=.25),
                        start=start, end=end, **costs)
                    sampled = eq.set_index("timestamp").equity.resample("1D").last()
                    if not np.allclose(sampled, eq_d.equity, rtol=1e-10, atol=1e-8):
                        raise ValueError("Control daily equity changed")
                    if len(fills) != len(fills_d) or any(not np.allclose(fills[c], fills_d[c], rtol=1e-10, atol=1e-8)
                                                       for c in ("qty", "price", "fee", "slippage")):
                        raise ValueError("Control daily fills changed")
                    control_comparison[f"{label}_{cost_name}"] = {
                        "preceding_report_unchanged": True, "same_daily_equity_and_fills": True,
                        "daily_net_return": daily_summary["total_return"]}
    timing = {name: {} for name in dict.fromkeys(("spot_control", frozen)) if name != "cash"}
    for name in timing:
        for label, (start, end) in periods.items():
            timing[name][label], _, _ = replay(name, start, end, NOMINAL, f"{label}_extra_delay", delay=1)
    all_summaries = [*development.values(), *(s for c in later.values() for p in c.values() for s in p.values()),
                     *(s for c in timing.values() for s in c.values())]
    gates = {"development_eligible": frozen != "cash", "new_candidate": frozen not in ("cash", "spot_control"),
             "accounting": all(s["accounting"]["reconciled"] for s in all_summaries)}
    if frozen != "cash":
        nominal, doubled = later["nominal"], later["doubled"]
        gates.update({f"beats_control_{p}": nominal[p][frozen]["total_return"] > nominal[p]["spot_control"]["total_return"]
                      for p in periods})
        gates.update(full_nominal_drawdown=nominal["full_continuous"][frozen]["max_intraday_drawdown_bound"] >= -.2,
                     full_doubled_drawdown=doubled["full_continuous"][frozen]["max_intraday_drawdown_bound"] >= -.2,
                     later_doubled_positive=doubled["later_continuous"][frozen]["net_pnl"] > 0)
    result = {"protocol": str(PLAN.relative_to(REPO)), **selection, "source_manifest": manifest,
        "previous_control_evidence_sha256": hashlib.sha256(PREVIOUS.read_bytes()).hexdigest(),
        "later": later, "extra_one_bar_delay": timing, "control_equivalence": control_comparison,
        "gates": gates, "historical_improvement_screen_passed": bool(all(gates.values())), "live_eligible": False,
        "reconciled_paths": len(all_summaries), "costs": {"nominal": NOMINAL, "doubled": STRESS},
        "initial_usdt": 1000., "constraints": {"spot_only": True, "leverage": False, "borrowing": False,
            "shorts": False, "derivatives": False, "martingale": False, "target_cap": .5, "asset_cap": .25},
        "limitations": ["Previously studied periods; not a fresh holdout",
            "Three surviving assets; universe survivorship not controlled",
            "Fixed entry-strength heuristic, not a forecast of future net returns or a significance test",
            "Modeled fees and impact, not measured live fills or account-specific tiers",
            "Signal eligibility state is distinct from actual filled holdings",
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
    parser.add_argument("--out", type=Path, default=Path("artifacts/momentum_refinement_20261009"))
    args = parser.parse_args()
    markets, manifest = load_spot_flow(args.data)
    run_research(markets, manifest, args.out)


if __name__ == "__main__":
    main()
