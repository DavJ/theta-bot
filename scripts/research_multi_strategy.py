"""Compare fixed source strategies and Kalman fusion; select on development.

Protocol: docs/evaluation/MULTI_STRATEGY_RESEARCH_PLAN.md. This is offline
historical research, never an exchange-order or live-enablement command.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research_dual_scale import DEVELOPMENT_END, VALIDATION_END, load_archives
from spot_bot.backtest.fast_backtest import run_backtest
from spot_bot.evaluate import EvaluationConfig, buy_and_hold
from spot_bot.strategies.multi_strategy import MULTI_APPROACHES, MultiStrategy


@dataclass(frozen=True)
class Candidate:
    name: str
    strategy: str
    price_space: str = "dollars"
    conf_power: float = 1.0


CANDIDATES = (
    Candidate("legacy_theta", "kalman_mr_dual"),
    Candidate("log_vol_theta", "kalman_mr_dual", "log_vol"),
    Candidate("log_vol_no_conf", "kalman_mr_dual", "log_vol", 0.0),
    *(Candidate(name, name) for name in MULTI_APPROACHES),
)


def run_candidate(data, candidate, config, start=None):
    return run_backtest(data, config.timeframe, candidate.strategy, config.psi_mode,
                        config.psi_window, config.rv_window, config.conc_window, config.base,
                        config.fee_rate, config.slippage_bps, config.max_exposure,
                        initial_usdt=config.initial_usdt, spread_bps=config.spread_bps,
                        dual_price_space=candidate.price_space, conf_power=candidate.conf_power,
                        allow_loss_exits=True, enforce_exposure_cap=True, evaluation_start=start, log=False)


def _worker(arguments):
    data, candidate, config, start = arguments
    equity, _, summary = run_candidate(data, candidate, config, start)
    yearly = {}
    previous = config.initial_usdt
    for year, rows in equity.groupby(equity.timestamp.dt.year):
        end = float(rows.equity.iloc[-1])
        yearly[str(year)] = end / previous - 1
        previous = end
    return candidate.name, {**summary, "calendar_returns_continuous_portfolio": yearly}


def evaluate_many(data, config, start=None, jobs=3):
    tasks = [(data, candidate, config, start) for candidate in CANDIDATES]
    if jobs == 1:
        return dict(map(_worker, tasks))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return dict(pool.map(_worker, tasks))


def select_on_development(development, config):
    eligible = [candidate for candidate in CANDIDATES
                if development[candidate.name]["net_pnl"] > 0
                and development[candidate.name]["maxDD"] >= -config.max_drawdown]
    return max(eligible, key=lambda c: development[c.name]["net_pnl"]) if eligible else None


def forecast_skill(data, config):
    timestamps = pd.to_datetime(data.timestamp, utc=True)
    # Build exactly the same causal theta features used by the backtest.
    from spot_bot.features import FeatureConfig, compute_features
    f = compute_features(data, FeatureConfig(base=config.base, rv_window=config.rv_window,
                                             conc_window=config.conc_window, psi_mode=config.psi_mode,
                                             psi_window=config.psi_window))
    for column in ["timestamp", "open", "high", "low", "close", "volume"]:
        f[column] = data[column].to_numpy()
    f = f.dropna(subset=["C", "S", "rv", "close"])
    diagnostics = MultiStrategy("kalman_fusion", config.max_exposure, config.fee_rate,
                                config.slippage_bps, config.spread_bps).generate_frame(f)
    diagnostics.index = pd.DatetimeIndex(pd.to_datetime(f.timestamp, utc=True))
    forecasts = diagnostics.resample("1D").first()
    close = pd.Series(data.close.to_numpy(), index=timestamps).resample("1D", label="right", closed="left").last()
    next_return = close.pct_change(fill_method=None).shift(-1)
    period = forecasts.loc[(forecasts.index >= DEVELOPMENT_END) & (forecasts.index < VALIDATION_END)].copy()
    period["realized_next_day_return"] = next_return.reindex(period.index)
    valid = period.dropna(subset=["fused_return", "mean_return", "realized_next_day_return"])
    skill = {}
    for column in ["fused_return", "mean_return", *(f"forecast_{name}" for name in ["theta", "momentum", "ema_trend", "breakout", "range_reversion"])]:
        error = valid[column] - valid.realized_next_day_return
        skill[column] = {"observations": len(valid), "mean_squared_error": float((error ** 2).mean()),
                         "directional_accuracy": float((np.sign(valid[column]) == np.sign(valid.realized_next_day_return)).mean())}
    skill["zero_return_forecast_mse"] = float((valid.realized_next_day_return ** 2).mean())
    skill["mean_error_correlation"] = float(valid.forecast_error_mean_correlation.mean())
    return skill, period


def research(data, config=EvaluationConfig(), *, jobs=3, out=None):
    config.validate()
    print("Development: all 11 predefined candidates, 2022–2024", flush=True)
    development = evaluate_many(data.loc[data.timestamp < DEVELOPMENT_END], config, jobs=jobs)
    selected = select_on_development(development, config)
    frozen = selected.name if selected else "cash"
    selection = {"frozen_candidate": frozen, "criterion": "maximum positive net P&L with development maxDD >= -15%",
                 "development": development, "config": asdict(config)}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "frozen_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print(f"Frozen development choice: {frozen}", flush=True)

    print("2025 historical validation (shared research session, not a new lockbox)", flush=True)
    prefix = data.loc[data.timestamp < VALIDATION_END]
    validation = evaluate_many(prefix, config, DEVELOPMENT_END, jobs)
    print("2025 doubled fee/slippage stress", flush=True)
    stress_config = replace(config, fee_rate=config.fee_rate * 2, slippage_bps=config.slippage_bps * 2)
    stress = evaluate_many(prefix, stress_config, DEVELOPMENT_END, jobs)
    print("2026 diagnostic replay: all predefined candidates", flush=True)
    diagnostic = evaluate_many(data, config, VALIDATION_END, jobs)

    quarters = {}
    review_candidates = {"kalman_fusion": next(c for c in CANDIDATES if c.name == "kalman_fusion")}
    if selected:
        review_candidates[selected.name] = selected
    boundaries = pd.date_range(DEVELOPMENT_END, VALIDATION_END, freq="QS")
    for name, candidate in review_candidates.items():
        quarters[name] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            print(f"Independently reset validation quarter: {name}, {start.date()}", flush=True)
            _, _, summary = run_candidate(data.loc[data.timestamp < end], candidate, config, start)
            quarters[name].append({"start": str(start), "end_exclusive": str(end), **summary})

    periods = {"development": data.loc[data.timestamp < DEVELOPMENT_END],
               "validation_2025": data.loc[(data.timestamp >= DEVELOPMENT_END) & (data.timestamp < VALIDATION_END)],
               "diagnostic_2026": data.loc[data.timestamp >= VALIDATION_END]}
    benchmarks = {name: {"cash_return": 0.0, "initial_30pct_hold": buy_and_hold(frame, config, config.max_exposure),
                         "full_hold_capital_reference": buy_and_hold(frame, config, 1.0)}
                  for name, frame in periods.items()}
    gates = {"development_candidate": selected is not None}
    if selected:
        held = validation[selected.name]
        gates.update(validation_positive=held["net_pnl"] > 0,
                     validation_drawdown=held["maxDD"] >= -config.max_drawdown,
                     doubled_costs_positive=stress[selected.name]["net_pnl"] > 0,
                     all_quarters_positive=all(q["net_pnl"] > 0 for q in quarters[selected.name]),
                     diagnostic_2026_positive=diagnostic[selected.name]["net_pnl"] > 0,
                     capital_return_beats_30pct_hold=held["total_return"] > benchmarks["validation_2025"]["initial_30pct_hold"]["total_return"])
    skill, predictions = forecast_skill(prefix, config)
    return {
        "protocol": "docs/evaluation/MULTI_STRATEGY_RESEARCH_PLAN.md", "config": asdict(config),
        "candidates": [asdict(c) for c in CANDIDATES], "frozen_candidate": frozen,
        "development": development, "validation_2025": validation,
        "stress_validation_2025": stress, "diagnostic_2026": diagnostic,
        "independently_reset_validation_quarters": quarters, "benchmarks": benchmarks,
        "forecast_skill_2025": skill, "gates": gates, "research_screen_passed": all(gates.values()),
        "live_eligible": False, "confirmatory_lockbox_passed": False,
        "recommended_action": "paper review only" if all(gates.values()) else "cash; continue research",
        "limitations": [
            "2025 is shared historical validation; 2026 is previously inspected diagnostic replay. No fresh confirmatory lockbox.",
            "Maximum means highest net P&L among this fixed BTC spot candidate set, within the stated exposure/drawdown constraints.",
            "All candidates permit loss exits in this comparison. This differs from the original 0.82% run's sell guard.",
            "One declared missing source hour is not filled; rolling windows count observed candles.",
            "OHLC fills/close timeout omit queues, partial fills and live latency; terminal equity is not liquidated.",
            "Fusion covariance and daily-return calibration are modeling assumptions, not proof of Gaussian independent forecast errors.",
            "The 30% cap is enforced at decision prices, overriding hysteresis; market drift during execution or minimum notional can leave a residual deviation. Capital-return benchmarks are not risk matched.",
        ],
    }, predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/multi_strategy"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--jobs", type=int, choices=[1, 2, 3, 4], default=3)
    args = parser.parse_args(argv)
    as_of = pd.to_datetime(args.as_of, utc=True) if args.as_of else pd.Timestamp.now(tz="UTC")
    data, manifests = load_archives(args.csv, EvaluationConfig(), as_of)
    report, predictions = research(data, jobs=args.jobs, out=args.out)
    report.update(as_of=str(as_of), provenance=manifests, rows=len(data),
                  missing_timestamps=[ts for m in manifests for a in m["archives"] for ts in a.get("missing_timestamps", [])])
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    predictions.to_csv(args.out / "daily_forecasts_2025.csv", index=True)
    print(f"Screen: {report['research_screen_passed']}; frozen: {report['frozen_candidate']}; {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
