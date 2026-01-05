#!/usr/bin/env python3
"""
Render a concise Quant Research Note from benchmark outputs.

Inputs:
- bench_out/benchmark_matrix.csv
- bench_out/benchmark_windows.csv
- optional benchmark_cmd.txt with the CLI used to run the benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap

import pandas as pd


def _to_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No data available._"
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return "_No data available._"
    return df[cols].to_markdown(index=False)


def _load_cmd(args: argparse.Namespace) -> str:
    if args.benchmark_cmd:
        return args.benchmark_cmd
    if args.benchmark_cmd_file and Path(args.benchmark_cmd_file).exists():
        try:
            return Path(args.benchmark_cmd_file).read_text().strip()
        except Exception:
            return ""
    return ""


def _best_runs(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_sharpe = matrix.sort_values("sharpe", ascending=False).head(3)
    top_dd = matrix.sort_values("max_drawdown").head(3)
    return top_sharpe, top_dd


def _window_section(windows: pd.DataFrame, run_ids: list[str]) -> str:
    subset = windows[windows["run_id"].isin(run_ids)]
    if subset.empty:
        return "_No window data._"
    rows = []
    for run_id, grp in subset.groupby("run_id"):
        best = grp.sort_values("return", ascending=False).head(3)
        worst = grp.sort_values("return").head(3)
        rows.append(f"**{run_id} — best windows**\n\n{_to_markdown_table(best, ['window_days','start','end','return','maxdd'])}\n")
        rows.append(f"**{run_id} — worst windows**\n\n{_to_markdown_table(worst, ['window_days','start','end','return','maxdd'])}\n")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render research note markdown.")
    ap.add_argument("--matrix", default="bench_out/benchmark_matrix.csv")
    ap.add_argument("--windows", default="bench_out/benchmark_windows.csv")
    ap.add_argument("--out", default="bench_out/research_note.md")
    ap.add_argument("--benchmark-cmd", default="")
    ap.add_argument("--benchmark-cmd-file", default="bench_out/benchmark_cmd.txt")
    args = ap.parse_args()

    matrix_path = Path(args.matrix)
    windows_path = Path(args.windows)
    matrix = pd.read_csv(matrix_path)
    windows = pd.read_csv(windows_path) if windows_path.exists() else pd.DataFrame()

    top_sharpe, top_dd = _best_runs(matrix)
    sharpe_runs = list(top_sharpe["run_id"]) if "run_id" in top_sharpe else []
    dd_runs = list(top_dd["run_id"]) if "run_id" in top_dd else []
    run_ids = sorted(set(sharpe_runs) | set(dd_runs))
    benchmark_cmd = _load_cmd(args)
    render_cmd = "python bench/render_research_note.py " + " ".join(sys.argv[1:])

    symbols = ", ".join(sorted(set(matrix["symbol"].astype(str))))
    timeframe = matrix.get("timeframe", pd.Series(["n/a"])).iloc[0]
    fee = matrix.get("fee_rate", pd.Series([0.0])).iloc[0]
    slippage = matrix.get("slippage_bps", pd.Series([0.0])).iloc[0]
    max_exp = matrix.get("max_exposure", pd.Series([0.0])).iloc[0]

    lines = [
        "# Quant Research Note",
        "",
        "## Executive Summary",
        "Top Sharpe and drawdown profiles summarized below. Costs applied to exposure changes (fee + slippage).",
        "",
        "## Definitions",
        "- $r_t = \\frac{P_t}{P_{t-1}} - 1$ (close-to-close return)",
        "- $RV$: rolling realized volatility",
        "- $\\phi$: log-phase of volatility",
        "- $C$: concentration of $\\phi$",
        "- $\\psi$: phase of $RV$ (cepstrum / Mellin variants)",
        "- $C_{int}$: internal concentration combining $C$ and $\\psi$",
        "- $S$: ensemble score from $C$ and $\\psi$",
        "",
        "## Experiment Setup",
        f"- Pairs: {symbols}",
        f"- Timeframe: {timeframe}",
        f"- Costs: fee_rate={fee}, slippage_bps={slippage}, max_exposure={max_exp}",
        f"- Benchmark command: `{benchmark_cmd}`" if benchmark_cmd else "- Benchmark command: _not provided_",
        f"- Render command: `{render_cmd}`",
        "",
        "## Top Runs by Sharpe",
        _to_markdown_table(top_sharpe, ["run_id", "symbol", "method", "psi_mode", "sharpe", "cagr", "max_drawdown", "turnover", "time_in_market"]),
        "",
        "## Top Runs by Max Drawdown (best)",
        _to_markdown_table(top_dd, ["run_id", "symbol", "method", "psi_mode", "max_drawdown", "sharpe", "cagr", "turnover", "time_in_market"]),
        "",
        "## Good / Bad Windows",
        _window_section(windows, run_ids),
        "",
        "## Conclusion",
        textwrap.dedent(
            """We model volatility and regime dynamics, not point forecasts of price direction.
            Exposure follows closed-bar signals with explicit transaction costs and risk gating,
            highlighting robustness across market conditions."""
        ).strip(),
        "",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote research note to {out_path}")


if __name__ == "__main__":
    main()
