"""Fixed signed futures study; no exchange orders or live configuration changes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research_execution_portfolio import load_daily, SYMBOLS, VALIDATION, TRANSFER, END
from spot_bot.backtest.perpetual import _frame, run_perpetual_backtest
from spot_bot.backtest.portfolio import run_portfolio_backtest
from spot_bot.portfolio.signed import SIGNED_APPROACHES, SignedPortfolio
from spot_bot.portfolio.trend import TrendPortfolio

BEGIN = pd.Timestamp("2022-01-01", tz="UTC")
CANDIDATES = ("spot_control", "perp_long_control", *SIGNED_APPROACHES)
COSTS = dict(fee_rate=.001, slippage_bps=5., spread_bps=2.)
STRESS = dict(fee_rate=.002, slippage_bps=10., spread_bps=2.)
PERIODS = {"2025": (VALIDATION, TRANSFER), "2026_01_09": (TRANSFER, END),
           "later_continuous": (VALIDATION, END), "continuous": (BEGIN, END)}


def load_futures(path):
    manifest = json.loads((path / "manifest.json").read_text())
    months = pd.date_range(BEGIN, END, freq="MS", inclusive="left").strftime("%Y-%m")
    required = {(s, k, m) for s in SYMBOLS for k in ("trade", "mark", "funding") for m in months}
    archives = manifest["archives"]
    actual = [(a["symbol"], a["kind"], a["month"]) for a in archives]
    if (manifest.get("source") != "binance_usdm_archive" or set(manifest["files"]) != set(SYMBOLS)
            or len(actual) != len(required) or set(actual) != required):
        raise ValueError("Complete fixed official futures provenance required")
    for a in archives:
        s, k, m = a["symbol"], a["kind"], a["month"]
        suffix = (f"fundingRate/{s}/{s}-fundingRate-{m}.zip" if k == "funding" else
                  f"{'klines' if k == 'trade' else 'markPriceKlines'}/{s}/{'1d' if k == 'trade' else '1h'}/{s}-{'1d' if k == 'trade' else '1h'}-{m}.zip")
        if (a["url"] != "https://data.binance.vision/data/futures/um/monthly/" + suffix
                or len(a["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in a["sha256"])):
            raise ValueError("Unexpected futures archive source/hash")
        for repair in a.get("official_daily_repairs", []):
            prefix = f"https://data.binance.vision/data/futures/um/daily/markPriceKlines/{s}/1h/{s}-1h-{m}-"
            if (k != "mark" or not repair["url"].startswith(prefix) or not repair["url"].endswith(".zip")
                    or len(repair["sha256"]) != 64 or repair["rows"] != 24):
                raise ValueError("Unexpected official daily repair")
    result = {"trade": {}, "mark": {}, "funding": {}}
    for s in SYMBOLS:
        if set(manifest["files"][s]) != set(result):
            raise ValueError("Missing futures series")
        for k in result:
            meta = manifest["files"][s][k]
            if meta["file"] != f"{s}_{k}.csv":
                raise ValueError("Unexpected assembled futures filename")
            file = path / meta["file"]
            if hashlib.sha256(file.read_bytes()).hexdigest() != meta["sha256"]:
                raise ValueError("Assembled futures file hash mismatch")
            frame = pd.read_csv(file)
            if len(frame) != meta["rows"]:
                raise ValueError("Futures provenance row count mismatch")
            if k == "funding":
                frame.event_time = pd.to_datetime(frame.event_time, utc=True, format="mixed").astype("datetime64[ns, UTC]")
                if (not frame.event_time.is_unique or not frame.event_time.is_monotonic_increasing
                        or frame.event_time.min() < BEGIN or frame.event_time.max() >= END
                        or not np.isfinite(frame[["last_funding_rate", "funding_interval_hours"]].to_numpy()).all()
                        or (frame.last_funding_rate.abs() > .5).any()
                        or not frame.funding_interval_hours.between(0, 24, inclusive="right").all()):
                    raise ValueError("Invalid actual settled funding records")
                coverage = frame.groupby(frame.event_time.dt.floor("D")).funding_interval_hours.sum()
                expected = pd.date_range(BEGIN, END, freq="D", inclusive="left").astype("datetime64[ns, UTC]")
                if not coverage.index.equals(expected) or not coverage.between(23.5, 24.5).all():
                    raise ValueError("Funding day coverage is incomplete")
            else:
                frame.index = pd.to_datetime(frame.pop("timestamp"), utc=True, format="mixed").astype("datetime64[ns, UTC]")
                frequency = "1D" if k == "trade" else "1h"
                frame = _frame(frame, frequency)
                expected = pd.date_range(BEGIN, END, freq=frequency, inclusive="left").astype("datetime64[ns, UTC]")
                if (not frame.index.equals(expected) or "volume" not in frame
                        or not np.isfinite(frame.volume).all() or (frame.volume < 0).any()):
                    raise ValueError("Futures full-period grid/value mismatch")
            result[k][s] = frame
    return result, manifest


def _daily_totals(frame, column, dates):
    if frame.empty:
        return np.zeros(len(dates))
    data = frame.copy()
    data["day"] = pd.to_datetime(data.timestamp, utc=True).astype("datetime64[ns, UTC]").dt.floor("D")
    return data.groupby("day")[column].sum().reindex(dates, fill_value=0.).cumsum().to_numpy()


def _reconcile_futures(equity, trades, settled, futures, summary):
    dates = pd.DatetimeIndex(equity.timestamp).astype("datetime64[ns, UTC]")
    fees = _daily_totals(trades, "fee", dates)
    realized = _daily_totals(trades, "realized_pnl", dates)
    fund = _daily_totals(settled, "payment", dates)
    cash = 1000. + realized - fees - fund
    nav = cash.copy()
    # Independent economic identity: sum signed trade cashflows plus inventory
    # mark value equals realized plus unrealized futures P&L.
    economic_nav = 1000. - fees - fund
    for s in SYMBOLS:
        t = trades.loc[trades.symbol == s].copy()
        t["signed_cashflow"] = t.signed_delta * t.price
        qty = _daily_totals(t, "signed_delta", dates)
        entry = equity[f"entry_{s}"].to_numpy()
        mark = futures["mark"][s].loc[dates + pd.Timedelta("23h"), "close"].to_numpy()
        np.testing.assert_allclose(qty, equity[f"qty_{s}"], rtol=1e-12, atol=1e-8)
        nav += qty * (mark - entry)
        economic_nav += qty * mark - _daily_totals(t, "signed_cashflow", dates)
    np.testing.assert_allclose(cash, equity.collateral_cash, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(nav, equity.equity, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(economic_nav, equity.equity, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(equity.unrealized_pnl, equity.equity - equity.collateral_cash, atol=1e-8)
    np.testing.assert_allclose([summary["fees_paid_total"], summary["slippage_paid_total"], summary["funding_net_paid_total"]],
                               [trades.fee.sum(), trades.slippage.sum(), settled.payment.sum()], atol=1e-8)


def _replay(markets, futures, close, targets, name, out, label, start, end, costs=COSTS):
    if name == "spot_control":
        equity, trades, summary = run_portfolio_backtest(markets,
            TrendPortfolio("momentum_vol", max_exposure=.5, asset_cap=.25), start=start, end=end, **costs)
        nav = equity.usdt.to_numpy().copy()
        for s, f in markets.items():
            nav += equity[f"base_{s}"].to_numpy() * f.loc[pd.DatetimeIndex(equity.timestamp), "close"].to_numpy()
        if (equity.usdt.min() < -1e-8 or equity.filter(like="base_").min().min() < -1e-8):
            raise AssertionError("Invalid spot inventory/cash")
        np.testing.assert_allclose(nav, equity.equity, rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose([summary["fees_paid_total"], summary["slippage_paid_total"]],
                                   [trades.fee.sum(), trades.slippage.sum()], atol=1e-8)
        summary.update(funding_net_paid_total=0., funding_paid_total=0., funding_received_total=0.,
                       margin_breach_hours=0, negative_collateral_hours=0, hourly_marks_checked=0)
        summary["accounting"] = "shared spot cash plus positive inventory"
    else:
        control = name == "perp_long_control"
        equity, trades, settled, summary = run_perpetual_backtest(futures["trade"], futures["mark"], futures["funding"],
            targets[name], close, start=start, end=end, gross_cap=.5 if control else 1.,
            asset_cap=.25 if control else .5, daily_rebalance=not control,
            volatility_target=None if control else .2, uniform_cap_scaling=not control, **costs)
        _reconcile_futures(equity, trades, settled, futures, summary)
        settled.to_csv(out / f"{name}_{label}_funding.csv", index=False)
        summary["accounting"] = "USDT collateral plus signed unrealized mark P&L"
    summary["cagr"] = (summary["final_equity"] / 1000) ** (365 / len(equity)) - 1
    summary["drawdown_within_20pct_bound"] = summary["max_intraday_drawdown_bound"] >= -.2
    summary["collateral_screen_passed"] = bool(summary["margin_breach_hours"] == 0
        and summary["negative_collateral_hours"] == 0 and equity.equity.min() > 0)
    monthly = equity.set_index("timestamp").equity.resample("MS").last()
    previous = monthly.shift(1).fillna(1000.)
    summary["monthly_returns"] = {str(t.date()): float(v) for t, v in (monthly / previous - 1).items()}
    summary["days"] = len(equity)
    summary["reconciled"] = True
    equity.to_csv(out / f"{name}_{label}_equity.csv", index=False)
    trades.to_csv(out / f"{name}_{label}_trades.csv", index=False)
    print(f"Replay {name} {label}: net={summary['total_return']:+.3%}; "
          f"boundDD={summary['max_intraday_drawdown_bound']:.3%}; margin={summary['margin_breach_hours']}", flush=True)
    return summary


def _later_worker(task):
    markets, futures, close, targets, name, out = task
    scenario = {"periods": {}, "doubled_costs": {}}
    for label, (start, end) in PERIODS.items():
        scenario["periods"][label] = _replay(markets, futures, close, targets, name, out, label, start, end)
        scenario["doubled_costs"][label] = _replay(markets, futures, close, targets, name, out,
                                                 label + "_double_cost", start, end, STRESS)
    return name, scenario


def research(markets, futures, out, jobs=3):
    out.mkdir(parents=True, exist_ok=True)
    close = pd.DataFrame({s: f.close for s, f in markets.items()})
    targets = {n: SignedPortfolio(n).weights(close) for n in CANDIDATES if n != "spot_control"}
    development = {n: _replay(markets, futures, close, targets, n, out, "development", BEGIN, VALIDATION)
                   for n in CANDIDATES}
    eligible = [n for n, s in development.items()
                if s["net_pnl"] > 0 and s["drawdown_within_20pct_bound"] and s["collateral_screen_passed"]]
    frozen = max(eligible, key=lambda n: development[n]["net_pnl"]) if eligible else "cash"
    selection = {"frozen_candidate": frozen, "development": development,
                 "criterion": "highest positive continuous 2022-2024 net return, <=20% bound, no collateral breach"}
    (out / "frozen_long_short_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print(f"Frozen before later-period economic metrics: {frozen}", flush=True)
    tasks = [(markets, futures, close, targets, n, out) for n in CANDIDATES]
    if jobs == 1:
        later = dict(map(_later_worker, tasks))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            later = dict(pool.map(_later_worker, tasks))
    scenarios = {}
    for n in CANDIDATES:
        scenario = {"development": development[n], **later[n]}
        summaries = [scenario["development"], *scenario["periods"].values(), *scenario["doubled_costs"].values()]
        scenario["all_historical_drawdowns_within_20pct_bound"] = all(s["drawdown_within_20pct_bound"] for s in summaries)
        scenario["all_collateral_screens_passed"] = all(s["collateral_screen_passed"] for s in summaries)
        scenarios[n] = scenario
    gates = {"development_eligible": frozen != "cash"}
    if frozen != "cash":
        chosen, control = scenarios[frozen], scenarios["spot_control"]
        gates.update(historical_risk_budget=chosen["all_historical_drawdowns_within_20pct_bound"],
            collateral_screen=chosen["all_collateral_screens_passed"],
            separate_newer_years_positive=all(chosen["periods"][p]["net_pnl"] > 0 for p in ("2025", "2026_01_09")),
            all_cost_stresses_positive=all(s["net_pnl"] > 0 for s in chosen["doubled_costs"].values()),
            improves_continuous_later=chosen["periods"]["later_continuous"]["net_pnl"] > control["periods"]["later_continuous"]["net_pnl"],
            improves_continuous_full=chosen["periods"]["continuous"]["net_pnl"] > control["periods"]["continuous"]["net_pnl"])
    return {"protocol": "docs/evaluation/LONG_SHORT_RESEARCH_PLAN.md", **selection, "scenarios": scenarios,
            "gates": gates, "historical_screen_passed": all(gates.values()), "live_eligible": False,
            "costs": COSTS, "stress_costs": STRESS, "initial_usdt_per_account": 1000., "drawdown_limit": .2,
            "new_strategy_limits": {"target_gross": 1., "target_absolute_asset": .5, "annual_volatility_target": .2,
                                    "prior_covariance_days": 60, "diagonal_shrinkage": .25},
            "margin_assumption": "exclude any hourly conservative NAV <5% gross or negative collateral cash; not exchange liquidation simulation",
            "funding_assumption": "actual rates/timestamps; hourly marks conservatively bound payment and execution-hour before/after inventory",
            "reconciliation": "all 81 paths: signed inventory, cash/realized/fees/funding, mark NAV, independent signed trade-flow identity and cost totals",
            "fresh_holdout": False,
            "limitations": ["Known periods and surviving BTC/ETH/BNB universe; no new independent economic confirmation.",
                "Conservative mark OHLC/funding bounds, not exact tick funding settlement or exact simultaneous drawdown.",
                "Spot uses daily OHLC; futures use hourly marks. The long-only futures control isolates instrument change.",
                "Gross targets <=100% do not prevent exposure drift or guarantee future loss limits.",
                "The 5% maintenance assumption does not model actual exchange margin tiers, liquidation, ADL or collateral conversion.",
                "Terminal holdings marked to market without forced close; fees for that close not included.",
                "Taxes, collateral interest, execution latency, depth and symbol precision are not modeled."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, nargs="+", required=True)
    parser.add_argument("--futures", type=Path, default=Path("data/raw/futures_research"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/long_short"))
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()
    markets, provenance = load_daily(args.daily, pd.Timestamp.now(tz="UTC"))
    futures, source = load_futures(args.futures)
    report = {**research(markets, futures, args.out, args.jobs), "spot_provenance": provenance,
              "futures_provenance": source, "as_of": str(pd.Timestamp.now(tz="UTC"))}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.report_out is not None:
        from scripts.write_long_short_report import write_report
        write_report(report, args.report_out, args.out)
    print(f"Completed: frozen={report['frozen_candidate']}; historical screen={report['historical_screen_passed']}")


if __name__ == "__main__":
    main()
