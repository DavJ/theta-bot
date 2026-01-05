#!/usr/bin/env python3
"""
Matrix benchmark across symbols, methods, and psi modes.

For each (symbol, method, psi_mode) combination:
- Export features via spot_bot.run_live (closed-bar, dryrun)
- Simulate simple exposure PnL with costs
- Record metrics into a single CSV
- Save per-run equity curve and config snapshot
- Compute rolling window stats (30d, 90d)

psi_mode variants are restricted to:
- none: baseline with method C only
- scale_phase: Mellin/log-scale psi with methods C and S
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import List

import pandas as pd

from bench.benchmark_pairs import (
    DEFAULT_SYMBOLS,
    compute_equity_curve,
    compute_equity_metrics,
    max_drawdown,
    run_features_export,
    _timeframe_to_timedelta,
)


DEFAULT_PSI_MODES = ["scale_phase", "none"]
DEFAULT_METHODS = ["C", "S"]
ALLOWED_PSI_MODES = {"none", "scale_phase"}
ALLOWED_METHODS = {"C", "S"}
WINDOW_SAMPLE_COUNT = 3


def _parse_list(val: str) -> List[str]:
    return [v.strip() for v in val.split(",") if v.strip()]


def _build_run_plan(symbols: List[str], psi_modes: List[str], methods: List[str]) -> List[tuple[str, str, str]]:
    norm_modes: List[str] = []
    for mode in psi_modes:
        mode_l = mode.lower()
        if mode_l in ALLOWED_PSI_MODES and mode_l not in norm_modes:
            norm_modes.append(mode_l)
    if not norm_modes:
        raise ValueError(f"psi_modes must be within {sorted(ALLOWED_PSI_MODES)}")

    norm_methods: List[str] = []
    for method in methods:
        method_u = method.upper()
        if method_u in ALLOWED_METHODS and method_u not in norm_methods:
            norm_methods.append(method_u)
    if not norm_methods:
        raise ValueError(f"methods must be within {sorted(ALLOWED_METHODS)}")

    plan: List[tuple[str, str, str]] = []
    for symbol in symbols:
        for psi_mode in norm_modes:
            allowed_methods = norm_methods if psi_mode != "none" else [m for m in norm_methods if m == "C"]
            for method in allowed_methods:
                plan.append((symbol, method, psi_mode))
    if not plan:
        raise ValueError("No valid (symbol, method, psi_mode) combinations to run.")
    return plan


def _select_exposure(df: pd.DataFrame, method: str, max_exposure: float) -> pd.Series:
    method = method.upper()
    if method == "C":
        base = pd.to_numeric(df.get("C"), errors="coerce").fillna(0.0).clip(lower=0.0)
        if base.max() <= 1.0 + 1e-12:
            base = base.clip(upper=1.0) * max_exposure
        else:
            base = base.clip(upper=max_exposure)
        return base

    if "target_exposure" in df.columns:
        return pd.to_numeric(df["target_exposure"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=max_exposure)

    base = pd.to_numeric(df.get("S"), errors="coerce").fillna(0.0).clip(lower=0.0)
    if base.max() <= 1.0 + 1e-12:
        base = base.clip(upper=1.0) * max_exposure
    else:
        base = base.clip(upper=max_exposure)
    return base


def _rolling_windows(equity: pd.Series, window_days: int, timeframe: str) -> List[dict]:
    if equity is None or equity.empty:
        return []
    try:
        tf_delta = _timeframe_to_timedelta(timeframe)
    except Exception:
        tf_delta = pd.Timedelta(hours=1)
    window = max(int(pd.Timedelta(days=window_days) / tf_delta), 1)
    if window <= 1 or len(equity) < window:
        return []
    rows = []
    for start in range(0, len(equity) - window + 1):
        seg = equity.iloc[start : start + window]
        ret = float(seg.iloc[-1] / seg.iloc[0] - 1.0)
        dd = max_drawdown(seg)
        rows.append({"start": seg.index[0], "end": seg.index[-1], "return": ret, "maxdd": dd})
    best = sorted(rows, key=lambda r: r["return"], reverse=True)[:WINDOW_SAMPLE_COUNT]
    worst = sorted(rows, key=lambda r: r["return"])[:WINDOW_SAMPLE_COUNT]
    return best + worst


def _save_command(workdir: Path) -> None:
    cmd = "python -m bench.benchmark_matrix " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "benchmark_cmd.txt").write_text(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark all symbol/method/psi combinations.")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--psi-modes", default=",".join(DEFAULT_PSI_MODES))
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--limit-total", type=int, default=8000)
    ap.add_argument("--workdir", default="bench_out")
    ap.add_argument("--out", default="bench_out/benchmark_matrix.csv")
    ap.add_argument("--windows-out", default="bench_out/benchmark_windows.csv")
    ap.add_argument("--rv-window", type=int, default=24)
    ap.add_argument("--conc-window", type=int, default=256)
    ap.add_argument("--psi-window", type=int, default=256)
    ap.add_argument("--cepstrum-min-bin", type=int, default=4)
    ap.add_argument("--cepstrum-max-frac", type=float, default=0.2)
    ap.add_argument("--base", type=float, default=10.0)
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--max-exposure", type=float, default=0.3)
    ap.add_argument("--cepstrum-domain", default="logtime")
    args = ap.parse_args()

    symbols = _parse_list(args.symbols)
    psi_modes = _parse_list(args.psi_modes)
    methods = _parse_list(args.methods)
    run_plan = _build_run_plan(symbols, psi_modes, methods)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    _save_command(workdir)

    rows = []
    window_rows = []

    for symbol, method, psi_mode in run_plan:
        safe_symbol = symbol.replace("/", "_")
        run_id = f"{safe_symbol}_{method}_{psi_mode}"
        feature_path = workdir / f"features_{run_id}.csv"
        run_features_export(
            symbol,
            args.timeframe,
            args.limit_total,
            feature_path,
            psi_mode=psi_mode,
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

        df = pd.read_csv(feature_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        exposure = _select_exposure(df, method, args.max_exposure)
        equity, turnover, exp_used = compute_equity_curve(
            df,
            exposure,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
            max_exposure=args.max_exposure,
        )
        metrics = compute_equity_metrics(equity, exp_used, turnover, timeframe=args.timeframe)

        eq_out = workdir / f"equity_{run_id}.csv"
        if not equity.empty:
            eq_df = pd.DataFrame(
                {
                    "timestamp": equity.index,
                    "equity": equity.values,
                    "exposure": exp_used.reindex(equity.index, fill_value=0.0),
                }
            )
            eq_df.to_csv(eq_out, index=False)

        config_path = workdir / f"config_{run_id}.json"
        with config_path.open("w") as f:
            json.dump(
                {
                    "symbol": symbol,
                    "method": method,
                    "psi_mode": psi_mode,
                    "timeframe": args.timeframe,
                    "limit_total": args.limit_total,
                    "rv_window": args.rv_window,
                    "conc_window": args.conc_window,
                    "psi_window": args.psi_window,
                    "cepstrum_min_bin": args.cepstrum_min_bin,
                    "cepstrum_max_frac": args.cepstrum_max_frac,
                    "base": args.base,
                    "fee_rate": args.fee_rate,
                    "slippage_bps": args.slippage_bps,
                    "max_exposure": args.max_exposure,
                },
                f,
                default=str,
                indent=2,
            )

        row = {
            "run_id": run_id,
            "symbol": symbol,
            "method": method,
            "psi_mode": psi_mode,
            "sharpe": metrics.get("sharpe", 0.0),
            "cagr": metrics.get("cagr", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "time_in_market": metrics.get("time_in_market", 0.0),
            "turnover": metrics.get("turnover", 0.0),
            "fee_rate": args.fee_rate,
            "slippage_bps": args.slippage_bps,
            "max_exposure": args.max_exposure,
            "timeframe": args.timeframe,
        }
        rows.append(row)

        for window_days in (30, 90):
            for window_row in _rolling_windows(equity, window_days=window_days, timeframe=args.timeframe):
                window_rows.append(
                    {
                        "run_id": run_id,
                        "symbol": symbol,
                        "method": method,
                        "psi_mode": psi_mode,
                        "window_days": window_days,
                        "start": window_row["start"],
                        "end": window_row["end"],
                        "return": window_row["return"],
                        "maxdd": window_row["maxdd"],
                    }
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved benchmark matrix: {out_path}")

    if window_rows:
        win_path = Path(args.windows_out)
        win_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(window_rows).to_csv(win_path, index=False)
        print(f"Saved rolling windows: {win_path}")


if __name__ == "__main__":
    main()
