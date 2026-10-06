"""Offline, chronological screening of the existing spot strategies.

Select on development data once; evaluate that frozen choice on the holdout.
This command writes evidence and never connects to an exchange or sends orders.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spot_bot.backtest.fast_backtest import _normalize_df, _timeframe_to_timedelta, run_backtest


BUNDLED_DATA = Path(__file__).resolve().parents[1] / "data/BTCUSDT_1H_real.csv.gz"
BUNDLED_SHA256 = "f2badec1af0eba20acabcfa68cfb3bb7cea9075c52510a2153ff68d96f1ff7fa"
STRATEGIES = ("meanrev", "kalman", "kalman_mr_dual", "lstm_kalman")


@dataclass(frozen=True)
class EvaluationConfig:
    timeframe: str = "1h"
    psi_mode: str = "scale_phase"
    psi_window: int = 200
    rv_window: int = 120
    conc_window: int = 200
    base: float = 2.0
    fee_rate: float = 0.001
    slippage_bps: float = 5.0
    spread_bps: float = 2.0
    max_exposure: float = 0.3
    initial_usdt: float = 1000.0
    holdout_fraction: float = 0.4
    max_data_age_days: float = 7.0
    min_trades: int = 20
    min_sharpe: float = 1.0
    max_drawdown: float = 0.15

    def validate(self):
        if not all(np.isfinite(v) for v in asdict(self).values() if isinstance(v, (float, int))):
            raise ValueError("Configuration must contain finite numbers.")
        if not 0 < self.holdout_fraction < 1 or not 0 < self.max_exposure <= 1:
            raise ValueError("holdout_fraction and max_exposure must be positive fractions.")
        if not 0 <= self.fee_rate < 0.5 or not 0 <= self.slippage_bps < 5000 or not 0 <= self.spread_bps < 5000:
            raise ValueError("Fees/slippage/spread must be nonnegative and within supported bounds.")
        if self.initial_usdt <= 0 or self.max_data_age_days < 0 or self.min_trades < 1:
            raise ValueError("Capital/trade count must be positive and maximum data age nonnegative.")
        if min(self.rv_window, self.conc_window, self.psi_window) < 2:
            raise ValueError("Feature windows must be at least two bars.")
        if self.base <= 1 or not 0 < self.max_drawdown < 1:
            raise ValueError("base must exceed 1 and max_drawdown must be a positive fraction.")
        if _timeframe_to_timedelta(self.timeframe).total_seconds() <= 0:
            raise ValueError("timeframe must be positive.")


def load_market_data(path: Path, config: EvaluationConfig, as_of: pd.Timestamp, *, allow_gaps=False) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "timestamp" not in raw:
        raise ValueError("CSV must explicitly contain timestamp and OHLCV columns.")
    df = _normalize_df(raw)
    columns = ["open", "high", "low", "close", "volume"]
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(df[columns].to_numpy()).all():
        raise ValueError("OHLCV contains missing or nonfinite values.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative.")
    if ((df.low > df[["open", "close"]].min(axis=1)) |
            (df.high < df[["open", "close"]].max(axis=1))).any():
        raise ValueError("Invalid OHLC ranges.")
    delta = _timeframe_to_timedelta(config.timeframe)
    intervals = df.timestamp.diff().dropna()
    valid_cadence = intervals.ge(delta).all() and intervals.mod(delta).eq(pd.Timedelta(0)).all()
    if not intervals.eq(delta).all() and not (allow_gaps and valid_cadence):
        raise ValueError("Missing candles or timestamps incompatible with timeframe.")
    if df.empty or df.timestamp.iloc[-1] + delta > as_of:
        raise ValueError("Dataset is empty or contains an unclosed/future candle.")
    return df


def _backtest(df, strategy, config, start=None):
    return run_backtest(
        df=df, timeframe=config.timeframe, strategy_name=strategy,
        psi_mode=config.psi_mode, psi_window=config.psi_window,
        rv_window=config.rv_window, conc_window=config.conc_window, base=config.base,
        fee_rate=config.fee_rate, slippage_bps=config.slippage_bps,
        spread_bps=config.spread_bps, max_exposure=config.max_exposure,
        initial_usdt=config.initial_usdt, log=False, evaluation_start=start,
    )


def buy_and_hold(df, config, exposure):
    """Same-period benchmark: market entry costs, then mark to final close."""
    fill = float(df.open.iloc[0]) * (1 + (config.slippage_bps + config.spread_bps / 2) / 10000)
    allocation = config.initial_usdt * exposure
    qty = allocation / (fill * (1 + config.fee_rate))
    end = config.initial_usdt - allocation + qty * float(df.close.iloc[-1])
    return {"final_equity": end, "net_pnl": end - config.initial_usdt,
            "total_return": end / config.initial_usdt - 1}


def evaluate(path: Path, config=EvaluationConfig(), *, as_of=None, data_source="unverified"):
    config.validate()
    if data_source not in {"unverified", "binance_archive", "synthetic"}:
        raise ValueError("Unsupported data source label.")
    as_of = pd.to_datetime(as_of, utc=True) if as_of is not None else pd.Timestamp.now(tz="UTC")
    df = load_market_data(path, config, as_of)
    dataset_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    source = "binance_bundled" if dataset_hash == BUNDLED_SHA256 and data_source == "unverified" else data_source
    provenance = None
    manifest_path = path.with_suffix(".source.json")
    if data_source == "binance_archive" or (data_source == "unverified" and manifest_path.exists()):
        if not manifest_path.exists() or manifest_path.stat().st_size > 1_000_000:
            raise ValueError("Archive provenance manifest is missing or oversized.")
        provenance = json.loads(manifest_path.read_text())
        if (provenance.get("source") != "binance_archive"
                or provenance.get("dataset_sha256") != dataset_hash
                or provenance.get("rows") != len(df)
                or provenance.get("timeframe") != config.timeframe
                or not provenance.get("archives")):
            raise ValueError("Archive provenance does not match this dataset.")
        if not all(item.get("url", "").startswith("https://data.binance.vision/data/spot/monthly/klines/")
                   for item in provenance["archives"]):
            raise ValueError("Archive provenance must identify official Binance archive URLs.")
        source = "binance_archive"
    split = int(len(df) * (1 - config.holdout_fraction))
    warmup = config.rv_window + config.conc_window + config.psi_window
    if split <= warmup + 100 or len(df) - split < 300:
        raise ValueError("Need more data for development, feature warmup and a 300-bar holdout.")

    development = {}
    for name in STRATEGIES:
        _, _, summary = _backtest(df.iloc[:split], name, config)
        development[name] = summary
    # Selection is locked BEFORE any holdout is evaluated. No tuning on holdout.
    selected = max(STRATEGIES, key=lambda name: development[name]["net_pnl"])
    cutoff = df.timestamp.iloc[split]
    equity, trades, holdout = _backtest(df, selected, config, cutoff)
    stress_config = replace(config, fee_rate=config.fee_rate * 2, slippage_bps=config.slippage_bps * 2)
    _, _, stressed = _backtest(df, selected, stress_config, cutoff)

    folds = []
    for indices in np.array_split(np.arange(split, len(df)), 3):
        start, end = int(indices[0]), int(indices[-1]) + 1
        _, _, summary = _backtest(df.iloc[:end], selected, config, df.timestamp.iloc[start])
        folds.append({"start": str(df.timestamp.iloc[start]), "end": str(df.timestamp.iloc[end - 1]), **summary})

    benchmark = buy_and_hold(df.iloc[split:], config, config.max_exposure)
    delta = _timeframe_to_timedelta(config.timeframe)
    age_days = (as_of - (df.timestamp.iloc[-1] + delta)).total_seconds() / 86400
    gates = {
        "market_data_provenance": source in {"binance_bundled", "binance_archive"},
        "data_current": age_days <= config.max_data_age_days,
        "development_net_positive": development[selected]["net_pnl"] > 0,
        "holdout_net_positive": holdout["net_pnl"] > 0,
        "beats_same_allocation_hold": holdout["net_pnl"] > benchmark["net_pnl"],
        "holdout_sharpe": holdout["sharpe"] >= config.min_sharpe,
        "holdout_drawdown": holdout["maxDD"] >= -config.max_drawdown,
        "holdout_activity": holdout["trades_count"] >= config.min_trades,
        "all_holdout_folds_positive": all(f["net_pnl"] > 0 for f in folds),
        "positive_with_doubled_fees_and_slippage": stressed["net_pnl"] > 0,
    }
    report = {
        "as_of": str(as_of), "dataset": path.name,
        "dataset_sha256": dataset_hash,
        "data_source": source, "rows": len(df), "start": str(df.timestamp.iloc[0]),
        "provenance": provenance,
        "end": str(df.timestamp.iloc[-1]), "age_days": age_days,
        "config": asdict(config), "selection_metric": "development net_pnl",
        "selected_strategy": selected, "holdout_start": str(cutoff),
        "development": development, "holdout": holdout, "stress_holdout": stressed,
        "holdout_folds": folds, "same_allocation_buy_hold": benchmark,
        "full_buy_hold": buy_and_hold(df.iloc[split:], config, 1.0), "cash_return": 0.0,
        "paper_screen_passed": all(gates.values()), "gates": gates,
        "live_eligible": False,
        "limitations": [
            "Historical screening thresholds are not statistical proof of an edge.",
            "The bundled historical dataset has been used in previous research; confirmation needs fresh or forward data.",
            "The stress run recomputes cost-dependent decisions, so higher costs can change which trades occur.",
            "Limit fills use OHLC, without order-book queue or partial-fill evidence.",
            "Untouched limits expire to market at bar close; zero-latency close fills are an approximation.",
            "Terminal equity includes open positions, without forced liquidation.",
            "The existing sell guard may hold losing positions; it does not prevent drawdowns.",
            "LSTM uses existing untrained deterministic weights, not a learned trading model.",
            "Passing this report does not authorize or enable live execution.",
        ],
    }
    return report, equity, trades


def render_markdown(report):
    c = report["config"]
    lines = ["# ThetaBot chronological evaluation", "",
             f"As of: {report['as_of']}. Dataset: `{report['dataset']}` ({report['rows']} bars).",
             f"Range: {report['start']} to {report['end']}. Data age: {report['age_days']:.1f} days.",
             f"SHA-256: `{report['dataset_sha256']}`.", "",
             f"Costs per fill: fee {c['fee_rate']:.4%}, slippage {c['slippage_bps']:g} bps, half of {c['spread_bps']:g} bps spread on market fills.",
             f"Initial capital: {c['initial_usdt']:g} USDT. Maximum target exposure: {c['max_exposure']:.0%}.",
             "The full reproducible configuration is in `summary.json`.", "",
             "## Development: predeclared candidates", "",
             "| Strategy | Net return | Sharpe | Max drawdown | Fills | Fees (USDT) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, s in report["development"].items():
        lines.append(f"| {name} | {s['total_return']:.2%} | {s['sharpe']:.2f} | {s['maxDD']:.2%} | {s['trades_count']:.0f} | {s['fees_paid_total']:.2f} |")
    s = report["holdout"]
    lines += ["", f"Selected by development net P&L: **{report['selected_strategy']}**.",
              f"Chronological holdout starts {report['holdout_start']}; prior history warms features without trading.", "",
              "## Frozen-choice holdout", "",
              "| Measurement | Result |", "|---|---:|",
              f"| Net return | {s['total_return']:.2%} |",
              f"| Net P&L | {s['net_pnl']:.2f} USDT |",
              f"| Sharpe | {s['sharpe']:.2f} |",
              f"| Max drawdown | {s['maxDD']:.2%} |",
              f"| Same allocation buy-and-hold | {report['same_allocation_buy_hold']['total_return']:.2%} |",
              f"| Full buy-and-hold | {report['full_buy_hold']['total_return']:.2%} |",
              f"| Cash | 0.00% |",
              f"| Doubled-fee/slippage return | {report['stress_holdout']['total_return']:.2%} |", "",
              "## Holdout subperiods", "", "| Start | End | Net return |", "|---|---|---:|"]
    for fold in report["holdout_folds"]:
        lines.append(f"| {fold['start']} | {fold['end']} | {fold['total_return']:.2%} |")
    lines += ["", "## Screening", "", f"Paper screening: **{'PASS' if report['paper_screen_passed'] else 'FAIL'}**. Live eligibility: **false**.", "",
              "| Check | Passed |", "|---|---|"]
    for name, passed in report["gates"].items():
        lines.append(f"| {name} | {passed} |")
    lines += ["", "## Limits", ""] + [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=BUNDLED_DATA)
    parser.add_argument("--out", type=Path, default=Path("bench_out/evaluation"))
    parser.add_argument("--data-source", choices=["unverified", "binance_archive", "synthetic"], default="unverified")
    parser.add_argument("--as-of")
    parser.add_argument("--require-paper-pass", action="store_true")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--psi-mode", choices=["none", "scale_phase"], default="scale_phase")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--spread-bps", type=float, default=2)
    parser.add_argument("--max-exposure", type=float, default=0.3)
    args = parser.parse_args(argv)
    config = EvaluationConfig(timeframe=args.timeframe, psi_mode=args.psi_mode,
                              fee_rate=args.fee_rate, slippage_bps=args.slippage_bps,
                              spread_bps=args.spread_bps, max_exposure=args.max_exposure)
    try:
        report, equity, trades = evaluate(args.csv, config, as_of=args.as_of, data_source=args.data_source)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "report.md").write_text(render_markdown(report))
    equity.to_csv(args.out / "holdout_equity.csv", index=False)
    trades.to_csv(args.out / "holdout_trades.csv", index=False)
    print(f"Selected: {report['selected_strategy']}; holdout net return: {report['holdout']['total_return']:.2%}; paper screening: {report['paper_screen_passed']}")
    print(f"Evidence: {args.out / 'report.md'}")
    return 2 if args.require_paper_pass and not report["paper_screen_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
