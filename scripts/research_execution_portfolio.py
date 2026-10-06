"""Run the fixed execution/portfolio protocol without placing exchange orders."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.research_dual_scale import load_archives
from spot_bot.backtest.fast_backtest import run_backtest
from spot_bot.backtest.portfolio import run_portfolio_backtest
from spot_bot.evaluate import EvaluationConfig, buy_and_hold, load_market_data
from spot_bot.portfolio.trend import PORTFOLIO_APPROACHES, TrendPortfolio


SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
START = pd.Timestamp("2023-01-01", tz="UTC")
VALIDATION = pd.Timestamp("2025-01-01", tz="UTC")
TRANSFER = pd.Timestamp("2026-01-01", tz="UTC")
END = pd.Timestamp("2026-10-01", tz="UTC")


def load_daily(paths, as_of):
    config = EvaluationConfig(timeframe="1d")
    parts = {s: [] for s in SYMBOLS}
    manifests = []
    for path in paths:
        manifest = json.loads(path.with_suffix(".source.json").read_text())
        symbol = manifest.get("symbol")
        frame = load_market_data(path, config, as_of)
        if (manifest.get("source") != "binance_archive" or symbol not in SYMBOLS
                or manifest.get("timeframe") != "1d" or manifest.get("rows") != len(frame)
                or manifest.get("dataset_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or not manifest.get("archives")
                or not all(a.get("url", "").startswith(
                    f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/")
                           for a in manifest["archives"])):
            raise ValueError("Daily file and official archive provenance do not match")
        parts[symbol].append(frame)
        manifests.append(manifest)
    expected = pd.date_range(pd.Timestamp("2022-01-01", tz="UTC"), END, freq="1D", inclusive="left")
    result = {}
    for symbol, frames in parts.items():
        if not frames:
            raise ValueError(f"Missing market {symbol}")
        frame = pd.concat(frames).sort_values("timestamp").set_index("timestamp")
        if not frame.index.equals(expected):
            raise ValueError("Protocol requires complete nonoverlapping Jan 2022–Sep 2026 daily bars")
        result[symbol] = frame
    return result, manifests


def portfolio_research(markets, out):
    costs = dict(fee_rate=0.001, slippage_bps=5.0, spread_bps=2.0)
    development = {}
    for name in PORTFOLIO_APPROACHES:
        print(f"Portfolio development 2023–2024: {name}", flush=True)
        _, _, development[name] = run_portfolio_backtest(markets, TrendPortfolio(name),
                                                        start=START, end=VALIDATION, **costs)
    eligible = [n for n, s in development.items() if s["net_pnl"] > 0 and s["maxDD"] >= -0.15]
    frozen = max(eligible, key=lambda n: development[n]["net_pnl"]) if eligible else "cash"
    selection = {"frozen_candidate": frozen, "development": development,
                 "criterion": "maximum positive net P&L with development max drawdown <=15%"}
    (out / "frozen_portfolio_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print(f"Frozen portfolio choice before validation: {frozen}", flush=True)
    validation, transfer, stress, quarters = {}, {}, {}, []
    for name in PORTFOLIO_APPROACHES:
        print(f"Portfolio historical validation/transfer: {name}", flush=True)
        policy = TrendPortfolio(name)
        for target, start, end, label in [(validation, VALIDATION, TRANSFER, "2025"),
                                         (transfer, TRANSFER, END, "2026")]:
            equity, trades, target[name] = run_portfolio_backtest(markets, policy, start=start, end=end, **costs)
            equity.to_csv(out / f"{name}_{label}_equity.csv", index=False)
            trades.to_csv(out / f"{name}_{label}_trades.csv", index=False)
    if frozen != "cash":
        policy = TrendPortfolio(frozen)
        for start, end, label in [(VALIDATION, TRANSFER, "2025"), (TRANSFER, END, "2026")]:
            _, _, stress[label] = run_portfolio_backtest(markets, policy, start=start, end=end,
                                                        fee_rate=0.002, slippage_bps=10.0, spread_bps=2.0)
        dates = pd.date_range(VALIDATION, TRANSFER, freq="QS")
        for start, end in zip(dates[:-1], dates[1:]):
            _, _, summary = run_portfolio_backtest(markets, policy, start=start, end=end, **costs)
            quarters.append({"start": str(start), "end_exclusive": str(end), **summary})
    benchmarks = {}
    config = EvaluationConfig(timeframe="1d")
    for label, start, end in [("development", START, VALIDATION), ("2025", VALIDATION, TRANSFER),
                               ("2026", TRANSFER, END)]:
        # Three separate initial 1000-USDT sleeves, each 10% invested, averaged:
        # equivalent to one shared initial 1000-USDT account with 10% per asset.
        refs = [buy_and_hold(f.loc[(f.index >= start) & (f.index < end)], config, 0.3)
                for f in markets.values()]
        benchmarks[label] = {"cash_return": 0.0,
                             "initial_30pct_equal_hold_return": sum(r["total_return"] for r in refs) / len(refs)}
    gates = {"development_eligible": frozen != "cash"}
    if frozen != "cash":
        gates.update(validation_positive=validation[frozen]["net_pnl"] > 0,
                     transfer_positive=transfer[frozen]["net_pnl"] > 0,
                     validation_drawdown=validation[frozen]["maxDD"] >= -0.15,
                     transfer_drawdown=transfer[frozen]["maxDD"] >= -0.15,
                     cost_stress_positive=all(s["net_pnl"] > 0 for s in stress.values()),
                     all_reset_quarters_positive=all(s["net_pnl"] > 0 for s in quarters))
    return {**selection, "validation_2025": validation, "transfer_2026": transfer,
            "doubled_costs_frozen": stress, "independent_2025_quarters": quarters,
            "benchmarks": benchmarks, "gates": gates, "research_screen_passed": all(gates.values()),
            "live_eligible": False, "costs": costs,
            "limits": asdict(TrendPortfolio(frozen if frozen != "cash" else "horizons_equal"))}


def _execution_worker(task):
    data, name, strategy, space, policy, start = task
    prefix = data.loc[data.timestamp < (TRANSFER if start == VALIDATION else END)]
    c = EvaluationConfig()
    _, _, summary = run_backtest(prefix, c.timeframe, strategy, c.psi_mode, c.psi_window,
                                 c.rv_window, c.conc_window, c.base, c.fee_rate, c.slippage_bps,
                                 c.max_exposure, spread_bps=c.spread_bps,
                                 allow_loss_exits=True, enforce_exposure_cap=True,
                                 dual_price_space=space, execution_policy=policy,
                                 evaluation_start=start, log=False)
    return str(start.year), name, policy, summary


def execution_research(data, jobs):
    approaches = [("legacy_theta", "kalman_mr_dual", "dollars"),
                  ("log_vol_theta", "kalman_mr_dual", "log_vol"),
                  ("breakout", "breakout", "dollars"),
                  ("kalman_fusion", "kalman_fusion", "dollars")]
    tasks = [(data, name, strategy, space, policy, start) for start in [VALIDATION, TRANSFER]
             for name, strategy, space in approaches for policy in ["limit_then_market", "market"]]
    result = {"2025": {}, "2026": {}}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for year, name, policy, summary in pool.map(_execution_worker, tasks):
            result[year].setdefault(name, {})[policy] = summary
            print(f"BTC execution {year} {name} {policy}: {summary['total_return']:.4%}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, nargs="+", required=True)
    parser.add_argument("--hourly-btc", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/execution_portfolio"))
    parser.add_argument("--jobs", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--as-of", default=None)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    as_of = pd.to_datetime(args.as_of, utc=True) if args.as_of else pd.Timestamp.now(tz="UTC")
    markets, daily_provenance = load_daily(args.daily, as_of)
    portfolio = portfolio_research(markets, args.out)
    # Save the complete portfolio result before starting the independent BTC
    # execution diagnosis; interruption cannot erase its frozen evidence.
    (args.out / "portfolio_summary.json").write_text(json.dumps(portfolio, indent=2, allow_nan=False) + "\n")
    hourly, hourly_provenance = load_archives(args.hourly_btc, EvaluationConfig(), as_of)
    execution = execution_research(hourly, args.jobs)
    report = {"protocol": "docs/evaluation/EXECUTION_PORTFOLIO_RESEARCH_PLAN.md", "as_of": str(as_of),
              "portfolio": portfolio, "execution": execution,
              "daily_provenance": daily_provenance, "hourly_provenance": hourly_provenance,
              "live_eligible": False,
              "limitations": ["Previously inspected BTC periods are reused diagnostic data, not fresh confirmation.",
                              "ETH and BNB are new asset data; their survival and current universe choice are known.",
                              "Daily market fills omit intraday paths, order latency and liquidity limits.",
                              "Decision-price caps can drift during a candle or leave minimum-notional residuals.",
                              "Terminal holdings are marked to market; no forced final liquidation.",
                              "Cost add-back is arithmetic on actual trades, not a zero-cost strategy rerun."]}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Completed; frozen={portfolio['frozen_candidate']}; screen={portfolio['research_screen_passed']}")


if __name__ == "__main__":
    main()
