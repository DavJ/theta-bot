"""Execute the predeclared dual-filter experiment; never place exchange orders.

Protocol: docs/evaluation/DUAL_SCALE_RESEARCH_PLAN.md. Development choices are
frozen before validation. 2026 is diagnostic because parts were already seen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spot_bot.backtest.fast_backtest import run_backtest
from spot_bot.evaluate import EvaluationConfig, buy_and_hold, load_market_data


@dataclass(frozen=True)
class Candidate:
    name: str
    price_space: str
    conf_power: float


CANDIDATES = (
    Candidate("legacy_dual", "dollars", 1.0),
    Candidate("log_vol_dual", "log_vol", 1.0),
    Candidate("log_vol_no_conf", "log_vol", 0.0),
)
DEVELOPMENT_END = pd.Timestamp("2025-01-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2026-01-01", tz="UTC")
DIAGNOSTIC_END = pd.Timestamp("2026-10-01", tz="UTC")


def load_archives(paths, config, as_of):
    frames, manifests = [], []
    for path in map(Path, paths):
        frame = load_market_data(path, config, as_of, allow_gaps=True)
        manifest = json.loads(path.with_suffix(".source.json").read_text())
        if (manifest.get("source") != "binance_archive" or manifest.get("symbol") != "BTCUSDT"
                or manifest.get("timeframe") != config.timeframe or manifest.get("rows") != len(frame)
                or manifest.get("dataset_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or not manifest.get("archives")):
            raise ValueError("Archive provenance does not match BTCUSDT dataset")
        if not all(item["url"].startswith("https://data.binance.vision/data/spot/monthly/klines/")
                   for item in manifest["archives"]):
            raise ValueError("Expected official Binance archive URLs")
        expected = pd.date_range(frame.timestamp.iloc[0], frame.timestamp.iloc[-1], freq="1h")
        missing = expected.difference(pd.DatetimeIndex(frame.timestamp))
        declared_missing = pd.to_datetime([ts for item in manifest["archives"]
                                          for ts in item.get("missing_timestamps", [])], utc=True)
        if not missing.equals(pd.DatetimeIndex(declared_missing).sort_values()):
            raise ValueError("Every archive gap must be explicitly listed in the source manifest")
        frames.append(frame)
        manifests.append(manifest)
    frame = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    expected = pd.date_range(pd.Timestamp("2022-01-01", tz="UTC"), DIAGNOSTIC_END, freq="1h", inclusive="left")
    actual = pd.DatetimeIndex(frame.timestamp)
    declared_missing = pd.to_datetime([ts for m in manifests for item in m["archives"]
                                      for ts in item.get("missing_timestamps", [])], utc=True)
    if (not actual.is_unique or not actual.isin(expected).all() or len(declared_missing) > 24
            or not expected.difference(actual).equals(pd.DatetimeIndex(declared_missing).sort_values())
            or actual[0] != expected[0] or actual[-1] != expected[-1]):
        raise ValueError("Protocol requires complete, nonoverlapping hourly data for Jan 2022–Sep 2026")
    return frame, manifests


def run_candidate(frame, candidate, config, start=None):
    return run_backtest(
        frame, config.timeframe, "kalman_mr_dual", config.psi_mode,
        config.psi_window, config.rv_window, config.conc_window, config.base,
        config.fee_rate, config.slippage_bps, config.max_exposure,
        initial_usdt=config.initial_usdt, spread_bps=config.spread_bps,
        dual_price_space=candidate.price_space, conf_power=candidate.conf_power,
        evaluation_start=start, log=False,
    )


def benchmarks(frame, config, summary):
    return {
        "cash_return": 0.0,
        "initial_30pct_hold": buy_and_hold(frame, config, config.max_exposure),
        "hindsight_mean_exposure_hold": buy_and_hold(frame, config, summary["mean_exposure"]),
        "note": "Mean-exposure allocation uses hindsight; descriptive, not investable or fully risk matched.",
    }


def research(frame, config=EvaluationConfig()):
    config.validate()
    development_frame = frame.loc[frame.timestamp < DEVELOPMENT_END]
    development = {}
    for candidate in CANDIDATES:
        print(f"Development: {candidate.name}", flush=True)
        _, _, summary = run_candidate(development_frame, candidate, config)
        development[candidate.name] = summary

    frozen = max(CANDIDATES, key=lambda c: development[c.name]["net_pnl"])
    dev = development[frozen.name]
    development_eligible = dev["net_pnl"] > 0 and dev["maxDD"] >= -config.max_drawdown
    # Lock the decision before observing any validation metric. Cash is a real
    # alternative to deploying the least bad loss-making candidate.
    paper_candidate = frozen.name if development_eligible else "cash"
    print(f"Frozen candidate: {frozen.name}; development action: {paper_candidate}", flush=True)

    validation_frame = frame.loc[frame.timestamp < VALIDATION_END]
    validation_period = validation_frame.loc[validation_frame.timestamp >= DEVELOPMENT_END]
    stress = replace(config, fee_rate=config.fee_rate * 2, slippage_bps=config.slippage_bps * 2)
    validation, stress_validation, reference = {}, {}, {}
    selected_equity = selected_trades = None
    for candidate in CANDIDATES:
        print(f"Validation: {candidate.name}", flush=True)
        equity, trades, summary = run_candidate(validation_frame, candidate, config, DEVELOPMENT_END)
        validation[candidate.name] = summary
        reference[candidate.name] = benchmarks(validation_period, config, summary)
        print(f"Cost stress: {candidate.name}", flush=True)
        _, _, stressed = run_candidate(validation_frame, candidate, stress, DEVELOPMENT_END)
        stress_validation[candidate.name] = stressed
        if candidate == frozen:
            selected_equity, selected_trades = equity, trades

    folds = []
    for indices in np.array_split(validation_period.index.to_numpy(), 3):
        start = frame.loc[int(indices[0]), "timestamp"]
        end = frame.loc[int(indices[-1]), "timestamp"] + pd.Timedelta(hours=1)
        print(f"Validation subperiod: {start}", flush=True)
        _, _, summary = run_candidate(frame.loc[frame.timestamp < end], frozen, config, start)
        folds.append({"start": str(start), "end_exclusive": str(end), **summary})

    print("2026 diagnostic replay (partly already seen)", flush=True)
    _, _, diagnostic = run_candidate(frame, frozen, config, VALIDATION_END)
    held = validation[frozen.name]
    gates = {
        "development_eligible": development_eligible,
        "validation_net_positive": held["net_pnl"] > 0,
        "validation_sharpe": held["sharpe"] >= config.min_sharpe,
        "validation_drawdown": held["maxDD"] >= -config.max_drawdown,
        "cost_stress_positive": stress_validation[frozen.name]["net_pnl"] > 0,
        "all_subperiods_positive": all(f["net_pnl"] > 0 for f in folds),
        "capital_return_beats_30pct_hold": held["total_return"] > reference[frozen.name]["initial_30pct_hold"]["total_return"],
        "diagnostic_2026_positive": diagnostic["net_pnl"] > 0,
    }
    return {
        "protocol": "docs/evaluation/DUAL_SCALE_RESEARCH_PLAN.md",
        "config": asdict(config), "candidates": [asdict(c) for c in CANDIDATES],
        "development_period": ["2022-01-01", "2025-01-01 (exclusive)"],
        "validation_period": ["2025-01-01", "2026-01-01 (exclusive)"],
        "diagnostic_period": ["2026-01-01", "2026-10-01 (exclusive); partly previously inspected"],
        "frozen_candidate": frozen.name, "development_paper_candidate": paper_candidate,
        "development": development, "validation": validation, "stress_validation": stress_validation,
        "validation_benchmarks": reference, "validation_folds": folds,
        "diagnostic_2026": diagnostic, "gates": gates,
        "research_screen_passed": all(gates.values()), "live_eligible": False,
        "recommended_action": "paper review" if all(gates.values()) else "cash; continue research",
        "limitations": [
            "Only three predefined candidates on one asset. No proof of statistical significance or future returns.",
            "2026 diagnostics are not an untouched holdout. Candidate comparison never changes the frozen selection.",
            "Fees and market slippage/spread are charged; OHLC fills cannot measure queues, partial fills or latency.",
            "The existing sell guard can retain losing inventory. Terminal equity is marked to market without liquidation.",
            "gross_pnl is an arithmetic cost add-back on the same trades, not a zero-cost strategy rerun.",
            "Mean exposure is measured at candle closes, not an intrabar execution-weighted exposure.",
            "Explicitly declared source gaps are preserved, not filled. Rolling windows count observed bars; orders during missing bars are not simulated.",
        ],
    }, selected_equity, selected_trades


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/dual_scale"))
    parser.add_argument("--as-of", default=None)
    args = parser.parse_args(argv)
    as_of = pd.to_datetime(args.as_of, utc=True) if args.as_of else pd.Timestamp.now(tz="UTC")
    frame, manifests = load_archives(args.csv, EvaluationConfig(), as_of)
    report, equity, trades = research(frame)
    report.update(as_of=str(as_of), provenance=manifests,
                  missing_timestamps=[ts for m in manifests for item in m["archives"]
                                      for ts in item.get("missing_timestamps", [])])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    equity.to_csv(args.out / "validation_equity.csv", index=False)
    trades.to_csv(args.out / "validation_trades.csv", index=False)
    print(f"Screen: {report['research_screen_passed']}; action: {report['recommended_action']}; {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
