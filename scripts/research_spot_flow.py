"""Run the fixed 4h spot-only comparison offline, with no order connection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.download_spot_flow import BEGIN, END, SYMBOLS, validate_bars
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.backtest.portfolio import run_portfolio_backtest
from spot_bot.portfolio.flow import spot_flow_targets, daily_control_targets
from spot_bot.portfolio.trend import TrendPortfolio

VALIDATION = pd.Timestamp("2025-01-01", tz="UTC")
TRANSFER = pd.Timestamp("2026-01-01", tz="UTC")
NOMINAL = dict(fee_rate=.001, slippage_bps=5., spread_bps=2.)
STRESS = dict(fee_rate=.002, slippage_bps=10., spread_bps=2.)
REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "docs/evaluation/SPOT_FLOW_RESEARCH_PLAN.md"


def load_spot_flow(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if (manifest.get("source") != "binance_spot_archive" or manifest.get("timeframe") != "4h"
            or manifest.get("gap_policy") != "reject" or set(manifest.get("datasets", {})) != set(SYMBOLS)):
        raise ValueError("Required spot-only archive provenance missing")
    expected = pd.date_range(BEGIN, END, freq="4h", inclusive="left").as_unit("ns")
    months = pd.date_range(BEGIN, END, freq="MS", inclusive="left").strftime("%Y-%m").tolist()
    result = {}
    for symbol in SYMBOLS:
        entry = manifest["datasets"][symbol]
        path = folder / f"{symbol}_4h.csv"
        if entry["file"] != path.name or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Spot dataset checksum mismatch")
        archives = entry.get("archives", [])
        if (len(archives) != len(months) or [a["month"] for a in archives] != months
                or any(a["url"] != f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/4h/{symbol}-4h-{a['month']}.zip"
                       or len(a.get("sha256", "")) != 64 for a in archives)):
            raise ValueError("Non-spot or incomplete archive manifest")
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).astype("datetime64[ns, UTC]")
        frame = frame.set_index("timestamp")
        if not frame.index.equals(expected) or len(frame) != entry["rows"] or sum(a["rows"] for a in archives) != len(frame):
            raise ValueError("Complete identical spot 4h dates required")
        validate_bars(frame)
        result[symbol] = frame
    return result, manifest


def freeze_development(development):
    eligible = [n for n, s in development.items() if s["net_pnl"] > 0 and s["max_intraday_drawdown_bound"] >= -.20]
    return max(eligible, key=lambda n: development[n]["net_pnl"]) if eligible else "cash"


def run_research(markets, manifest, out):
    out.mkdir(parents=True, exist_ok=True)
    targets = spot_flow_targets(markets)
    control, daily = daily_control_targets(markets)
    targets = {"spot_control": control, **targets}
    plan_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    development = {}

    def replay(name, start, end, costs, label, delay=0):
        eq, fills, summary = run_intraday_spot(markets, targets[name], start=start, end=end,
            daily_control=name == "spot_control", signal_delay_bars=delay, **costs)
        summary["accounting"] = reconcile_spot_path(markets, eq, fills)
        eq.to_csv(out / f"{name}_{label}_equity.csv", index=False)
        fills.to_csv(out / f"{name}_{label}_trades.csv", index=False)
        return summary, eq, fills

    for name in targets:
        development[name], _, _ = replay(name, BEGIN, VALIDATION, NOMINAL, "development")
        print(f"Development {name}: {development[name]['total_return']:.3%}", flush=True)
    frozen = freeze_development(development)
    selection = {"frozen_candidate": frozen, "development": development,
        "criterion": "largest positive 2022–2024 net return with conservative 4h DD >= -20%",
        "protocol_sha256": plan_hash}
    # Write the development choice BEFORE any later-period economic replay.
    (out / "frozen_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print(f"Frozen before later replays: {frozen}", flush=True)
    periods = {"2025": (VALIDATION, TRANSFER), "2026": (TRANSFER, END),
               "later_continuous": (VALIDATION, END), "full_continuous": (BEGIN, END)}
    later = {cost: {label: {} for label in periods} for cost in ("nominal", "doubled")}
    control_comparison = {}
    for cost_name, costs in (("nominal", NOMINAL), ("doubled", STRESS)):
        for label, (start, end) in periods.items():
            for name in targets:
                summary, eq, fills = replay(name, start, end, costs, f"{label}_{cost_name}")
                later[cost_name][label][name] = summary
                print(f"{label} {cost_name} {name}: {summary['total_return']:.3%}; DD {summary['max_intraday_drawdown_bound']:.3%}", flush=True)
                if name == "spot_control":
                    original_eq, original_trades, original = run_portfolio_backtest(daily,
                        TrendPortfolio("momentum_vol", max_exposure=.5, asset_cap=.25), start=start, end=end, **costs)
                    sampled = eq.set_index("timestamp").resample("1D").last()
                    pd.testing.assert_series_equal(sampled.equity.reset_index(drop=True),
                        original_eq.equity.reset_index(drop=True), check_names=False, rtol=1e-10, atol=1e-8)
                    if len(fills) != len(original_trades):
                        raise ValueError("Control trade count changed between daily and 4h replay")
                    for column in ("qty", "price", "fee", "slippage"):
                        if not np.allclose(fills[column], original_trades[column], rtol=1e-10, atol=1e-8):
                            raise ValueError(f"Control {column} changed between daily and 4h replay")
                    control_comparison[f"{label}_{cost_name}"] = {"daily_net_return": original["total_return"],
                        "four_hour_net_return": summary["total_return"], "daily_drawdown_bound": original["max_intraday_drawdown_bound"],
                        "four_hour_drawdown_bound": summary["max_intraday_drawdown_bound"],
                        "same_close_equity_and_fills": True}
    timing = {}
    if frozen != "cash":
        for label, (start, end) in periods.items():
            timing[label], _, _ = replay(frozen, start, end, NOMINAL, f"{label}_extra_delay", delay=1)
    gates = {"development_eligible": frozen != "cash"}
    if frozen != "cash":
        nom, doubled = later["nominal"], later["doubled"]
        gates.update({f"beats_control_{label}": nom[label][frozen]["total_return"] > nom[label]["spot_control"]["total_return"]
                      for label in ("2025", "2026", "later_continuous")})
        gates.update(full_nominal_drawdown=nom["full_continuous"][frozen]["max_intraday_drawdown_bound"] >= -.2,
                     full_doubled_drawdown=doubled["full_continuous"][frozen]["max_intraday_drawdown_bound"] >= -.2,
                     later_doubled_positive=doubled["later_continuous"][frozen]["net_pnl"] > 0,
                     accounting=all(s[frozen]["accounting"]["reconciled"] for c in later.values() for s in c.values()))
    result = {"protocol": str(PLAN.relative_to(REPO)), "protocol_sha256": plan_hash,
        "source_manifest": manifest, **selection, "later": later, "extra_one_bar_delay_frozen": timing,
        "control_daily_equivalence": control_comparison, "gates": gates,
        "historical_improvement_screen_passed": bool(all(gates.values())), "live_eligible": False,
        "costs": {"nominal": NOMINAL, "doubled": STRESS}, "initial_usdt": 1000.,
        "constraints": {"spot_only": True, "leverage": False, "borrowing": False, "shorts": False,
                        "derivatives": False, "martingale": False, "target_cap": .5, "asset_cap": .25},
        "limitations": ["Previously studied historical periods; not a fresh holdout",
                        "Three surviving assets; universe survivorship not controlled",
                        "Aggregated taker-flow bars, not historical order-book depth",
                        "Modeled costs, not measured live fills or account-specific fee tier",
                        "Overlapping doubling windows are not independent future probabilities",
                        "Conservative within-bar high/low bound, not exact tick drawdown",
                        "20% historical budget is not a future loss guarantee"]}
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/spot_flow"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/spot_flow_20261007"))
    args = parser.parse_args()
    markets, manifest = load_spot_flow(args.data)
    run_research(markets, manifest, args.out)


if __name__ == "__main__":
    main()
