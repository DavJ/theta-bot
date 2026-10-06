"""Fixed source ablations and purged five-day forecast portfolio research."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research_execution_portfolio import load_daily, VALIDATION, TRANSFER, END
from spot_bot.backtest.portfolio import run_portfolio_backtest
from spot_bot.features.external import external_features, validate_signals
from spot_bot.portfolio.forecast import (ForecastConfig, TargetPortfolio, forecast_targets,
                                         price_features, walk_forward_forecast)
from spot_bot.portfolio.risk import PortfolioRisk
from spot_bot.portfolio.trend import TrendPortfolio

BEGIN = pd.Timestamp("2022-01-01", tz="UTC")
MODELS = {"ridge_price": ("ridge", ()), "ridge_macro": ("ridge", ("macro",)),
          "ridge_sentiment": ("ridge", ("sentiment",)), "ridge_crypto": ("ridge", ("crypto",)),
          "ridge_all": ("ridge", ("macro", "sentiment", "crypto")),
          "boost_price": ("boost", ()), "boost_all": ("boost", ("macro", "sentiment", "crypto"))}
CANDIDATES = ("control_momentum", *MODELS, "blend_all")
RISK = PortfolioRisk(volatility_target=.20)
COSTS = dict(fee_rate=.001, slippage_bps=5., spread_bps=2.)
STRESS = dict(fee_rate=.002, slippage_bps=10., spread_bps=2.)


def load_sources(path, markets):
    manifest = json.loads((path / "manifest.json").read_text())
    signal_path = path / "signals.csv"
    if hashlib.sha256(signal_path.read_bytes()).hexdigest() != manifest["signals_sha256"]:
        raise ValueError("External signal snapshot hash mismatch")
    records = pd.read_csv(signal_path)
    records["event_time"] = pd.to_datetime(records.event_time, utc=True, format="mixed")
    records["available_at"] = pd.to_datetime(records.available_at, utc=True, format="mixed")
    records = validate_signals(records)
    required = {"EURUSD", "USDJPY", "NASDAQCOM", "VIXCLS", "DGS10", "DCOILWTICO", "FEAR_GREED",
                *(f"FLOW_{s}" for s in markets), *(f"FUNDING_{s}" for s in markets)}
    if set(records.source) != required:
        raise ValueError("Fixed protocol requires all 13 sources")
    for symbol, market in markets.items():
        meta = manifest["spot_files"][symbol]
        source_path = path / meta["file"]
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError("Spot flow file hash mismatch")
        source = pd.read_csv(source_path)
        source.index = pd.to_datetime(source.timestamp, utc=True)
        if not source.index.equals(market.index):
            raise ValueError("Spot flow and original market dates differ")
        np.testing.assert_allclose(source[["open", "high", "low", "close", "volume"]].to_numpy(),
                                   market[["open", "high", "low", "close", "volume"]].to_numpy(), rtol=1e-12)
    return records, manifest


def _forecast_worker(task):
    name, symbol, features, close, algorithm = task
    result = walk_forward_forecast(features, close, ForecastConfig(algorithm=algorithm))
    return name, symbol, result


def predictions(markets, sources, out, jobs, *, names=tuple(MODELS), extra_delay=pd.Timedelta(0)):
    base = price_features(markets)
    index = next(iter(markets.values())).index
    external, coverage = external_features(sources, index, extra_delay=extra_delay)
    tasks = []
    for name in names:
        algorithm, families = MODELS[name]
        for symbol, frame in markets.items():
            features = pd.concat([base[symbol], *(external[f] for f in families)], axis=1)
            tasks.append((name, symbol, features, frame.close, algorithm))
    result = {name: {} for name in names}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for name, symbol, frame in pool.map(_forecast_worker, tasks):
            result[name][symbol] = frame
            frame.to_csv(out / f"{name}_{symbol}_forecast.csv")
            print(f"Forecast {name} {symbol}: {frame.forecast.notna().sum()} available decisions", flush=True)
    return result, coverage


def _targets(markets, forecasts):
    close = pd.DataFrame({s: f.close for s, f in markets.items()})
    targets = {"control_momentum": TrendPortfolio("momentum_vol", max_exposure=.5, asset_cap=.25).weights(close)}
    for name, items in forecasts.items():
        frame = pd.DataFrame({s: items[s].forecast for s in markets})
        targets[name] = forecast_targets(frame, close)
    if "ridge_all" in targets and "boost_all" in targets:
        targets["blend_all"] = (targets["control_momentum"] + targets["ridge_all"] + targets["boost_all"]) / 3
    return targets


def _replay(markets, name, targets, out, label, start, end, costs=COSTS):
    policy = (TrendPortfolio("momentum_vol", max_exposure=.5, asset_cap=.25) if name == "control_momentum"
              else TargetPortfolio("momentum_vol", max_exposure=1, asset_cap=.5, targets=targets[name]))
    equity, trades, summary = run_portfolio_backtest(markets, policy, start=start, end=end,
                                                    risk=None if name == "control_momentum" else RISK, **costs)
    if (equity.usdt.min() < -1e-8 or equity.filter(like="base_").min().min() < -1e-8):
        raise AssertionError("Negative cash or spot inventory")
    nav = equity.usdt.to_numpy().copy()
    for symbol, frame in markets.items():
        nav += equity[f"base_{symbol}"].to_numpy() * frame.loc[pd.DatetimeIndex(equity.timestamp), "close"].to_numpy()
    np.testing.assert_allclose(nav, equity.equity.to_numpy(), rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose([trades.fee.sum(), trades.slippage.sum()],
                               [summary["fees_paid_total"], summary["slippage_paid_total"]], atol=1e-8)
    summary["cagr"] = (summary["final_equity"] / 1000) ** (365 / len(equity)) - 1
    summary["drawdown_within_20pct_bound"] = summary["max_intraday_drawdown_bound"] >= -.2
    monthly = equity.set_index("timestamp").equity.resample("MS").last()
    previous = monthly.shift(1).fillna(1000)
    summary["monthly_returns"] = {str(t.date()): float(v) for t, v in (monthly / previous - 1).items()}
    equity.to_csv(out / f"{name}_{label}_equity.csv", index=False)
    trades.to_csv(out / f"{name}_{label}_trades.csv", index=False)
    print(f"Replay {name} {label}: net={summary['total_return']:+.3%}; "
          f"boundDD={summary['max_intraday_drawdown_bound']:.3%}", flush=True)
    return summary


def forecast_diagnostics(markets, forecasts):
    close = pd.DataFrame({s: f.close for s, f in markets.items()})
    actual = close.shift(-5) / close - 1
    losses, metrics = {}, {}
    for name, items in forecasts.items():
        pred = pd.DataFrame({s: f.forecast for s, f in items.items()})
        mask = (close.index >= VALIDATION) & (close.index < END - pd.Timedelta("5D"))
        p, a = pred.loc[mask], actual.loc[mask]
        good = p.notna() & a.notna()
        err = (p - a).pow(2).where(good)
        losses[name] = err.mean(axis=1).where(good.all(axis=1))
        pairs = pd.DataFrame({"pred": p.to_numpy().ravel(), "actual": a.to_numpy().ravel()}).dropna()
        coefficient = pairs.pred.corr(pairs.actual, method="spearman") if len(pairs) > 30 else np.nan
        metrics[name] = {"paired_asset_dates": len(pairs), "mse": float(err.stack().mean()),
                         "zero_forecast_mse": float(a.pow(2).where(good).stack().mean()),
                         "spearman_ic": float(coefficient) if np.isfinite(coefficient) else None}
    additions = {}
    for name, reference in [("ridge_macro", "ridge_price"), ("ridge_sentiment", "ridge_price"),
                             ("ridge_crypto", "ridge_price"), ("ridge_all", "ridge_price"),
                             ("boost_all", "boost_price")]:
        diff = (losses[reference] - losses[name]).dropna().to_numpy()
        if len(diff) < 40:
            additions[name] = {"reference": reference, "days": len(diff), "significant_99pct": False}
            continue
        rng = np.random.default_rng(20261006)
        n, block, reps = len(diff), 20, 1000
        starts = rng.integers(0, n - block + 1, size=(reps, int(np.ceil(n / block))))
        indices = (starts[..., None] + np.arange(block)).reshape(reps, -1)[:, :n]
        boot = diff[indices].mean(axis=1)
        low, high = np.quantile(boot, [.005, .995])
        additions[name] = {"reference": reference, "days": n, "mean_mse_improvement": float(diff.mean()),
                           "bonferroni_99pct_ci": [float(low), float(high)], "block_days": block,
                           "bootstrap_replications": reps, "significant_99pct": bool(low > 0)}
    return {"period": "2025-01-01 through 2026-09-25; completed five-day outcomes", "models": metrics,
            "incremental_source_value": additions,
            "interpretation": "20-day block bootstrap across asset-averaged paired loss; approximate uncertainty, not economic proof"}


def research(markets, sources, out, jobs):
    out.mkdir(parents=True, exist_ok=True)
    forecasts, coverage = predictions(markets, sources, out, jobs)
    targets = _targets(markets, forecasts)
    development = {n: _replay(markets, n, targets, out, "development", BEGIN, VALIDATION) for n in CANDIDATES}
    eligible = [n for n, s in development.items() if s["net_pnl"] > 0 and s["drawdown_within_20pct_bound"]]
    frozen = max(eligible, key=lambda n: development[n]["net_pnl"]) if eligible else "cash"
    selection = {"frozen_candidate": frozen, "development": development,
                 "criterion": "maximum positive continuous 2022-2024 net return with conservative drawdown <=20%"}
    (out / "frozen_external_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print(f"Frozen before later-period economic metrics: {frozen}", flush=True)
    scenarios = {}
    for name in CANDIDATES:
        scenario = {"development": development[name], "periods": {}, "doubled_costs": {},
                    "risk": None if name == "control_momentum" else asdict(RISK)}
        for label, start, end in [("2025", VALIDATION, TRANSFER), ("2026_01_09", TRANSFER, END),
                                   ("continuous", BEGIN, END)]:
            scenario["periods"][label] = _replay(markets, name, targets, out, label, start, end)
            scenario["doubled_costs"][label] = _replay(markets, name, targets, out, f"{label}_double_cost", start, end, STRESS)
        scenario["all_historical_drawdowns_within_20pct_bound"] = all(
            s["drawdown_within_20pct_bound"] for s in [scenario["development"], *scenario["periods"].values(),
                                                        *scenario["doubled_costs"].values()])
        scenarios[name] = scenario
    sensitivity = {}
    external_choice = (frozen in MODELS and bool(MODELS[frozen][1])) or frozen == "blend_all"
    if external_choice:
        delayed_dir = out / "publication_delay_3d"
        delayed_dir.mkdir(exist_ok=True)
        names = ("ridge_all", "boost_all") if frozen == "blend_all" else (frozen,)
        delayed, _ = predictions(markets, sources, delayed_dir, jobs, names=names, extra_delay=pd.Timedelta("3D"))
        delayed_targets = _targets(markets, delayed)
        for label, start, end in [("2025", VALIDATION, TRANSFER), ("2026_01_09", TRANSFER, END)]:
            sensitivity[label] = _replay(markets, frozen, delayed_targets, delayed_dir, label, start, end)
    gates = {"development_eligible": frozen != "cash"}
    if frozen != "cash":
        chosen = scenarios[frozen]
        gates.update(historical_risk_budget=chosen["all_historical_drawdowns_within_20pct_bound"],
                     later_net_positive=all(chosen["periods"][p]["net_pnl"] > 0 for p in ("2025", "2026_01_09")),
                     doubled_costs_positive=all(s["net_pnl"] > 0 for s in chosen["doubled_costs"].values()))
        if external_choice:
            gates["extra_publication_delay_positive"] = all(s["net_pnl"] > 0 and s["drawdown_within_20pct_bound"]
                                                            for s in sensitivity.values())
    return {"protocol": "docs/evaluation/EXTERNAL_SIGNAL_RESEARCH_PLAN.md", **selection,
            "scenarios": scenarios, "gates": gates, "historical_screen_passed": all(gates.values()),
            "live_eligible": False, "model": asdict(ForecastConfig()), "model_groups": MODELS,
            "costs": COSTS, "stress_costs": STRESS, "drawdown_limit": .2,
            "source_coverage": coverage, "forecast_diagnostics": forecast_diagnostics(markets, forecasts),
            "publication_delay_sensitivity": sensitivity,
            "publication_delay_sensitivity_applicable": external_choice,
            "historical_first_release_vintages_verified": False,
            "reconciliation": "all paths: nonnegative cash/inventory, NAV and executed fee/impact totals checked"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, nargs="+", required=True)
    parser.add_argument("--sources", type=Path, default=Path("data/raw/external_signals"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/external_signals"))
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()
    markets, daily_provenance = load_daily(args.daily, pd.Timestamp.now(tz="UTC"))
    sources, source_manifest = load_sources(args.sources, markets)
    report = {**research(markets, sources, args.out, args.jobs), "daily_provenance": daily_provenance,
              "external_provenance": source_manifest, "as_of": str(pd.Timestamp.now(tz="UTC"))}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.report_out is not None:
        from scripts.write_external_signal_report import write_report
        write_report(report, args.report_out, args.out)
    print(f"Completed: frozen={report['frozen_candidate']}; historical screen={report['historical_screen_passed']}")


if __name__ == "__main__":
    main()
