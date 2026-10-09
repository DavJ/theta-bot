"""Fixed eight-policy historical spot-cohort study; no live account or order access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.download_spot_flow import BEGIN, END
from scripts.download_spot_universe import load_universe
from scripts.research_spot_flow import NOMINAL, STRESS, VALIDATION, TRANSFER
from scripts.research_momentum_refinement import fill_frequency
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.broad_universe import broad_targets

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "docs/evaluation/BROAD_UNIVERSE_RESEARCH_PLAN.md"
PREVIOUS = REPO / "docs/evaluation/THETABOT_MULTISCALE_SPOT_2026-10-09.json"
CONTROLS = ("spot_control", "capped_control")


def freeze_broad(development):
    eligible = [name for name, s in development.items() if s["net_pnl"] > 0
                and s["max_intraday_drawdown_bound"] >= -.20 and s["held_unquoted_asset_bars"] == 0]
    return max(eligible, key=lambda name: development[name]["net_pnl"]) if eligible else "cash"


def paired_block_interval(candidate_daily, control_daily):
    """Descriptive 99% circular-block interval on already studied later-period log returns."""
    if (not candidate_daily.index.equals(control_daily.index) or candidate_daily.empty
            or not candidate_daily.index.to_series().diff().dropna().eq(pd.Timedelta("1D")).all()
            or not np.isfinite(candidate_daily).all() or not np.isfinite(control_daily).all()
            or candidate_daily.le(0).any() or control_daily.le(0).any()):
        raise ValueError("Aligned positive finite calendar-day account values required")
    candidate = np.diff(np.log(np.r_[1000., candidate_daily.to_numpy()]))
    control = np.diff(np.log(np.r_[1000., control_daily.to_numpy()]))
    excess = candidate - control
    n = len(excess)
    rng = np.random.default_rng(23812)
    starts = rng.integers(0, n, size=(2000, int(np.ceil(n / 14))))
    indices = (starts[:, :, None] + np.arange(14)) % n
    samples = excess[indices.reshape(2000, -1)[:, :n]].mean(axis=1) * 365
    lower, upper = np.quantile(samples, [.005, .995])
    return {"days": n, "block_days": 14, "replicates": 2000, "seed": 23812, "confidence": .99,
        "annualized_mean_log_excess": float(excess.mean() * 365),
        "lower": float(lower), "upper": float(upper), "positive_lower_bound": bool(lower > 0),
        "interpretation": "descriptive on reused data; not confirmatory significance; selection uncertainty omitted"}


def run_research(markets, manifest, out):
    out.mkdir(parents=True, exist_ok=True)
    targets, executable, quoted, _ = broad_targets(markets)
    plan_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    previous = json.loads(PREVIOUS.read_text())
    source_hash = hashlib.sha256((json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
    hashes, daily_accounts = {}, {}
    for name, target in targets.items():
        payload = target.to_csv()
        (out / f"{name}_completed_targets.csv").write_text(payload)
        hashes[name] = hashlib.sha256(payload.encode()).hexdigest()
    quote_payload = quoted.to_csv()
    (out / "quoted_mask.csv").write_text(quote_payload)

    def replay(name, start, end, costs, label, delay=0):
        eq, fills, summary = run_intraday_spot(executable, targets[name], start=start, end=end,
            quoted_mask=quoted, daily_control=True, signal_delay_bars=delay, **costs)
        summary["accounting"] = reconcile_spot_path(executable, eq, fills)
        summary["fill_frequency"] = fill_frequency(eq, fills)
        eq.to_csv(out / f"{name}_{label}_equity.csv", index=False)
        fills.to_csv(out / f"{name}_{label}_trades.csv", index=False)
        if label == "later_continuous_nominal":
            daily_accounts[name] = eq.set_index("timestamp").equity.resample("1D").last()
        return summary

    development = {}
    for name in targets:
        development[name] = replay(name, BEGIN, VALIDATION, NOMINAL, "development")
        s = development[name]
        print(f"Development {name}: {s['total_return']:.3%}; DD {s['max_intraday_drawdown_bound']:.3%}; "
              f"held-unquoted {s['held_unquoted_asset_bars']}", flush=True)
    frozen = freeze_broad(development)
    selection = {"frozen_candidate": frozen, "development": development,
        "criterion": "largest positive 2022–2024 net with conservative DD >= -20% and zero held-unquoted bars",
        "protocol_sha256": plan_hash, "source_manifest_sha256": source_hash,
        "frozen_at": str(pd.Timestamp.now(tz="UTC"))}
    (out / "frozen_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print(f"Frozen before later economic replay: {frozen}", flush=True)
    periods = {"2025": (VALIDATION, TRANSFER), "2026": (TRANSFER, END),
               "later_continuous": (VALIDATION, END), "full_continuous": (BEGIN, END)}
    later = {cost: {p: {} for p in periods} for cost in ("nominal", "doubled")}
    checks = {}
    for cost_name, costs in (("nominal", NOMINAL), ("doubled", STRESS)):
        for label, (start, end) in periods.items():
            for name in targets:
                s = replay(name, start, end, costs, f"{label}_{cost_name}")
                later[cost_name][label][name] = s
                print(f"{label} {cost_name} {name}: {s['total_return']:.3%}; "
                      f"DD {s['max_intraday_drawdown_bound']:.3%}; unquoted {s['held_unquoted_asset_bars']}", flush=True)
                if name in CONTROLS:
                    old = previous["later"][cost_name][label][name]
                    for field in ("total_return", "cagr", "max_intraday_drawdown_bound", "fees_paid_total",
                                  "slippage_paid_total", "trades_count"):
                        if not np.isclose(s[field], old[field], rtol=1e-10, atol=1e-8):
                            raise ValueError(f"Previous {name} {field} changed")
                    checks[f"{name}_{label}_{cost_name}"] = True
    timing = {p: {} for p in periods}
    for label, (start, end) in periods.items():
        for name in targets:
            s = replay(name, start, end, NOMINAL, f"{label}_extra_delay", delay=1)
            timing[label][name] = s
            print(f"{label} delayed {name}: {s['total_return']:.3%}; "
                  f"DD {s['max_intraday_drawdown_bound']:.3%}; unquoted {s['held_unquoted_asset_bars']}", flush=True)
    summaries = [*development.values(), *(s for c in later.values() for p in c.values() for s in p.values()),
                 *(s for p in timing.values() for s in p.values())]
    gates = {"development_eligible": frozen != "cash", "new_candidate": frozen not in (*CONTROLS, "cash"),
             "accounting": all(s["accounting"]["reconciled"] for s in summaries), "controls_unchanged": all(checks.values())}
    interval, material = None, {"full_cagr_1_5x": False, "later_profit_1_5x": False}
    if frozen != "cash":
        nom, doubled = later["nominal"], later["doubled"]
        selected_paths = [development[frozen], *(p[frozen] for c in later.values() for p in c.values()),
                          *(p[frozen] for p in timing.values())]
        interval = paired_block_interval(daily_accounts[frozen], daily_accounts["capped_control"])
        gates.update({f"beats_both_controls_{p}": nom[p][frozen]["total_return"] >
                      max(nom[p][c]["total_return"] for c in CONTROLS) for p in periods})
        gates.update(no_held_unquoted_inventory=all(s["held_unquoted_asset_bars"] == 0 for s in selected_paths),
            full_nominal_drawdown=nom["full_continuous"][frozen]["max_intraday_drawdown_bound"] >= -.2,
            full_doubled_drawdown=doubled["full_continuous"][frozen]["max_intraday_drawdown_bound"] >= -.2,
            full_delayed_drawdown=timing["full_continuous"][frozen]["max_intraday_drawdown_bound"] >= -.2,
            later_doubled_positive=doubled["later_continuous"][frozen]["net_pnl"] > 0,
            later_delayed_positive=timing["later_continuous"][frozen]["net_pnl"] > 0,
            descriptive_99pct_lower_positive=interval["positive_lower_bound"])
        material = {"full_cagr_1_5x": nom["full_continuous"][frozen]["cagr"] >=
                    1.5 * nom["full_continuous"]["capped_control"]["cagr"],
                    "later_profit_1_5x": nom["later_continuous"][frozen]["total_return"] >=
                    1.5 * nom["later_continuous"]["capped_control"]["total_return"]}
    passed = bool(all(gates.values()))
    result = {"protocol": str(PLAN.relative_to(REPO)), **selection, "source_manifest": manifest,
        "previous_evidence_sha256": hashlib.sha256(PREVIOUS.read_bytes()).hexdigest(),
        "target_hashes": hashes, "quoted_mask_sha256": hashlib.sha256(quote_payload.encode()).hexdigest(),
        "later": later, "extra_one_bar_delay": timing, "control_checks": checks, "gates": gates,
        "descriptive_paired_bootstrap": interval, "material_gates": material,
        "historical_improvement_screen_passed": passed,
        "material_improvement_screen_passed": bool(passed and all(material.values())), "live_eligible": False,
        "reconciled_paths": len(summaries), "costs": {"nominal": NOMINAL, "doubled": STRESS}, "initial_usdt": 1000.,
        "constraints": {"spot_only": True, "leverage": False, "borrowing": False, "shorts": False,
                        "derivatives": False, "martingale": False, "target_cap": .5, "asset_cap": .25},
        "limitations": ["Previously studied periods; not a new unseen holdout",
            "Fixed dated 15-asset cohort reduces current-survivor bias but omits post-2021 entrants and other venues",
            "Zero sentinel valuation for missing held markets is pessimistic, not an observed price; such paths fail quality gate",
            "Old Terra 2.0 airdrops excluded; Polygon/old Terra quantity continuity uses documented identity rules",
            "Model/universe selection and repeated-research uncertainty are not captured by the descriptive bootstrap",
            "Modeled fees/impact, not actual fills, account-specific fee tiers or intrabar liquidity guarantees",
            "Overlapping rolling windows do not estimate independent future doubling probabilities",
            "Conservative within-bar high/low bound, not exact tick chronology",
            "20% historical budget does not guarantee future maximum loss"]}
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != plan_hash:
        raise ValueError("Protocol changed during economic replay")
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/spot_universe"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/broad_spot_20261009"))
    args = parser.parse_args()
    markets, manifest = load_universe(args.data)
    run_research(markets, manifest, args.out)


if __name__ == "__main__":
    main()
