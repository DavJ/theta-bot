#!/usr/bin/env python3
"""
One-shot benchmark over multiple symbols.

What it does:
- For each symbol: runs spot_bot.run_live in dryrun mode to export features CSV
- Loads exported CSV and computes a compact summary table:
  psi uniqueness, S range, risk_state counts, mean RV by risk_state, exposure proxy, etc.
- Prints a sorted table + optionally writes a summary CSV.

Usage example:
  python bench/benchmark_pairs.py --limit-total 8000 --out benchmark_summary.csv
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "AVAX/USDT",
]


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    match = re.fullmatch(r"(\d+)([mhd])", timeframe.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    value = int(match.group(1))
    unit = match.group(2).lower()
    unit_map = {"m": "minutes", "h": "hours", "d": "days"}
    return pd.Timedelta(**{unit_map[unit]: value})


def _bars_per_year(timeframe: str, index: pd.Index | None) -> float:
    try:
        delta = _timeframe_to_timedelta(timeframe)
        if delta.total_seconds() > 0:
            return float(pd.Timedelta(days=365) / delta)
    except Exception:
        pass
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        delta = index.to_series().diff().median()
        if pd.notna(delta) and delta.total_seconds() > 0:
            return float(pd.Timedelta(days=365) / delta)
    return 24 * 365.0


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    safe_peak = peak.where(peak > 0, 1e-8)
    dd = (equity - peak) / safe_peak
    return float(dd.min())


def compute_equity_curve(
    df: pd.DataFrame,
    exposure: pd.Series,
    *,
    fee_rate: float,
    slippage_bps: float,
    max_exposure: float,
    initial_equity: float = 1.0,
) -> Tuple[pd.Series, float, pd.Series]:
    if df is None or df.empty:
        empty = pd.Series(dtype=float)
        return empty, 0.0, empty

    closes = pd.to_numeric(df["close"], errors="coerce")
    returns = closes.pct_change().shift(-1).fillna(0.0)
    exp = pd.to_numeric(exposure, errors="coerce").fillna(0.0).clip(lower=0.0)
    exp = exp.clip(upper=max_exposure)

    if len(closes) < 2:
        empty = pd.Series(dtype=float, index=closes.index)
        return empty, 0.0, exp.iloc[:0]

    n_steps = len(closes) - 1
    turnover = 0.0
    equity = float(initial_equity)
    prev_exp = 0.0
    fee_mult = fee_rate + slippage_bps / 10000.0
    equity_vals = []
    equity_index = []

    for i in range(n_steps):
        exp_t = float(exp.iloc[i]) if i < len(exp) else 0.0
        exp_t = min(max(exp_t, 0.0), max_exposure)
        delta = exp_t - prev_exp
        turnover += abs(delta)
        equity -= equity * abs(delta) * fee_mult

        ret = float(returns.iloc[i]) if pd.notna(returns.iloc[i]) else 0.0
        equity *= 1.0 + exp_t * ret

        equity_vals.append(equity)
        equity_index.append(df.index[i + 1])
        prev_exp = exp_t

    equity_series = pd.Series(equity_vals, index=equity_index, dtype=float)
    used_exp = exp.iloc[:n_steps].copy()
    return equity_series, turnover, used_exp


def compute_equity_metrics(
    equity: pd.Series,
    exposure: pd.Series,
    turnover: float,
    *,
    timeframe: str,
) -> dict:
    if equity is None or equity.empty:
        return {
            "final_return": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "time_in_market": 0.0,
        }

    ret_series = equity.pct_change().dropna()
    periods_per_year = _bars_per_year(timeframe, equity.index)
    initial_equity = 1.0
    final_equity = float(equity.iloc[-1])
    final_return = float(final_equity / initial_equity - 1.0)
    cagr = float((final_equity / initial_equity) ** (periods_per_year / max(len(equity), 1)) - 1.0)
    volatility = float(ret_series.std(ddof=0) * np.sqrt(periods_per_year)) if not ret_series.empty else 0.0
    sharpe = (
        float(ret_series.mean() * periods_per_year / (ret_series.std(ddof=0) + 1e-12))
        if not ret_series.empty
        else 0.0
    )
    return {
        "final_return": final_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "turnover": float(turnover),
        "time_in_market": float((exposure.abs() > 1e-12).mean()) if len(exposure) else 0.0,
    }


def run_features_export(
    symbol: str,
    timeframe: str,
    limit_total: int,
    out_csv: Path,
    *,
    psi_mode: str,
    psi_window: int,
    cepstrum_domain: str,
    cepstrum_min_bin: int,
    cepstrum_max_frac: float,
    rv_window: int,
    conc_window: int,
    base: float,
    fee_rate: float,
    slippage_bps: float,
    max_exposure: float,
) -> None:
    cmd = [
        "python",
        "-m",
        "spot_bot.run_live",
        "--mode",
        "dryrun",
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--limit-total",
        str(limit_total),
        "--psi-mode",
        psi_mode,
        "--psi-window",
        str(psi_window),
        "--cepstrum-domain",
        cepstrum_domain,
        "--cepstrum-min-bin",
        str(cepstrum_min_bin),
        "--cepstrum-max-frac",
        str(cepstrum_max_frac),
        "--rv-window",
        str(rv_window),
        "--conc-window",
        str(conc_window),
        "--base",
        str(base),
        "--fee-rate",
        str(fee_rate),
        "--slippage-bps",
        str(slippage_bps),
        "--max-exposure",
        str(max_exposure),
        "--csv-out",
        str(out_csv),
        "--csv-out-mode",
        "features",
    ]

    print(f"\n=== Exporting features for {symbol} -> {out_csv} ===")
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Feature export failed for {symbol} (exit {res.returncode})")


def summarize_features_csv(
    symbol: str,
    csv_path: Path,
    *,
    timeframe: str,
    max_exposure: float,
    fee_rate: float,
    slippage_bps: float,
) -> tuple[dict, pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    row = {"symbol": symbol, "rows": int(len(df))}

    # Core sanity metrics
    if "psi" in df.columns:
        psi = pd.to_numeric(df["psi"], errors="coerce")
        row["psi_uniq_3dp"] = int(psi.round(3).nunique(dropna=True))
        row["psi_min"] = float(psi.min())
        row["psi_max"] = float(psi.max())
    else:
        row["psi_uniq_3dp"] = None
        row["psi_min"] = None
        row["psi_max"] = None

    if "S" in df.columns:
        S = pd.to_numeric(df["S"], errors="coerce")
        row["S_min"] = float(S.min())
        row["S_max"] = float(S.max())
    else:
        row["S_min"] = None
        row["S_max"] = None

    # Risk state counts and mean RV per state
    if "risk_state" in df.columns:
        vc = df["risk_state"].value_counts(dropna=False)
        row["ON"] = int(vc.get("ON", 0))
        row["REDUCE"] = int(vc.get("REDUCE", 0))
        row["OFF"] = int(vc.get("OFF", 0))
    else:
        row["ON"] = row["REDUCE"] = row["OFF"] = None

    if "rv" in df.columns and "risk_state" in df.columns:
        grp = df.groupby("risk_state")["rv"].mean()
        row["rv_ON"] = float(grp.get("ON", float("nan")))
        row["rv_REDUCE"] = float(grp.get("REDUCE", float("nan")))
        row["rv_OFF"] = float(grp.get("OFF", float("nan")))
    else:
        row["rv_ON"] = row["rv_REDUCE"] = row["rv_OFF"] = None

    # Exposure proxy (if present)
    # target_exposure is best; fallback to intent_exposure
    exp_col = None
    for c in ("target_exposure", "intent_exposure"):
        if c in df.columns:
            exp_col = c
            break

    if exp_col:
        exp = pd.to_numeric(df[exp_col], errors="coerce").fillna(0.0)
        row["exposure_mean"] = float(exp.mean())
        row["exposure_frac_on"] = float((exp > 1e-12).mean())
        row["exposure_col"] = exp_col
    else:
        row["exposure_mean"] = None
        row["exposure_frac_on"] = None
        row["exposure_col"] = None

    if "S" in df.columns and "rv" in df.columns:
        S = pd.to_numeric(df["S"], errors="coerce")
        rv = pd.to_numeric(df["rv"], errors="coerce")
        q = pd.qcut(S, 5, duplicates="drop")
        g = rv.groupby(q, observed=True).mean()
        row["rv_S_q0"] = float(g.iloc[0])
        row["rv_S_q4"] = float(g.iloc[-1])

    if "S" in df.columns and "rv" in df.columns:
        S = pd.to_numeric(df["S"], errors="coerce")
        rv = pd.to_numeric(df["rv"], errors="coerce")
        q = pd.qcut(S, 5, duplicates="drop")
        g = rv.groupby(q, observed=True).mean()
        row["rv_S_low"] = float(g.iloc[0])   # nejnižší kvantil S
        row["rv_S_high"] = float(g.iloc[-1]) # nejvyšší kvantil S
    else:
        row["rv_S_low"] = None
        row["rv_S_high"] = None

    equity = pd.Series(dtype=float)
    exp_used = pd.Series(dtype=float)
    if "close" in df.columns and exp_col:
        equity, turnover, exp_used = compute_equity_curve(
            df,
            df[exp_col],
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            max_exposure=max_exposure,
        )
        metrics = compute_equity_metrics(equity, exp_used, turnover, timeframe=timeframe)
        row.update(metrics)
        row["final_equity"] = float(equity.iloc[-1]) if not equity.empty else 1.0
    else:
        row.update(
            {
                "final_return": None,
                "cagr": None,
                "volatility": None,
                "sharpe": None,
                "max_drawdown": None,
                "turnover": None,
                "time_in_market": None,
                "final_equity": None,
            }
        )

    return row, equity, exp_used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                    help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--limit-total", type=int, default=8000)
    ap.add_argument("--workdir", default="bench_out", help="Where to store per-pair feature CSVs")
    ap.add_argument("--out", default="", help="Optional summary CSV output path")
    # Fixed research params (keep constant across pairs)
    ap.add_argument("--psi-mode", default="complex_cepstrum")
    ap.add_argument("--psi-window", type=int, default=256)
    ap.add_argument("--cepstrum-domain", default="logtime")
    ap.add_argument("--cepstrum-min-bin", type=int, default=4)
    ap.add_argument("--cepstrum-max-frac", type=float, default=0.2)
    ap.add_argument("--rv-window", type=int, default=24)
    ap.add_argument("--conc-window", type=int, default=256)
    ap.add_argument("--base", type=float, default=10.0)
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--max-exposure", type=float, default=0.3)

    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sym in symbols:
        safe = sym.replace("/", "_")
        csv_path = workdir / f"features_{safe}.csv"

        run_features_export(
            sym,
            args.timeframe,
            args.limit_total,
            csv_path,
            psi_mode=args.psi_mode,
            psi_window=args.psi_window,
            cepstrum_domain=args.cepstrum_domain,
            cepstrum_min_bin=args.cepstrum_min_bin,
            cepstrum_max_frac=args.cepstrum_max_frac,
            rv_window=args.rv_window,
            conc_window=args.conc_window,
            base=args.base,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
            max_exposure=args.max_exposure,
        )

        summary_row, equity, exp_used = summarize_features_csv(
            sym,
            csv_path,
            timeframe=args.timeframe,
            max_exposure=args.max_exposure,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
        )
        rows.append(summary_row)

        if not equity.empty:
            eq_out = workdir / f"equity_{safe}.csv"
            eq_df = pd.DataFrame(
                {
                    "timestamp": equity.index,
                    "equity": equity.values,
                    "exposure": exp_used.reindex(equity.index, fill_value=0.0),
                }
            )
            eq_df.to_csv(eq_out, index=False)
            print(f"Saved equity curve for {sym}: {eq_out}")

    summary = pd.DataFrame(rows)

    # A simple "quality score" for sorting:
    # - prefer higher psi uniqueness
    # - prefer rv_ON lower than rv_REDUCE (risk gating separation)
    def quality_score(r):
        s = 0.0
        if pd.notna(r.get("psi_uniq_3dp")):
            s += float(r["psi_uniq_3dp"])
        if pd.notna(r.get("rv_ON")) and pd.notna(r.get("rv_REDUCE")):
            # bigger separation => better
            s += 1000.0 * max(0.0, float(r["rv_REDUCE"]) - float(r["rv_ON"]))
        return s

    summary["score"] = summary.apply(quality_score, axis=1)
    summary = summary.sort_values("score", ascending=False)

    # Pretty print
    cols = [
        "symbol", "rows",
        "psi_uniq_3dp", "psi_min", "psi_max",
        "S_min", "S_max",
        "ON", "REDUCE", "OFF",
        "rv_ON", "rv_REDUCE", "rv_OFF",
        "exposure_col", "exposure_mean", "exposure_frac_on",
        "final_equity", "final_return", "cagr", "volatility", "sharpe", "max_drawdown", "turnover", "time_in_market",
        "score"
    ]
    cols = [c for c in cols if c in summary.columns]
    print("\n\n================== BENCHMARK SUMMARY ==================")
    print(summary[cols].to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"\nSaved summary CSV: {out_path}")


if __name__ == "__main__":
    main()
