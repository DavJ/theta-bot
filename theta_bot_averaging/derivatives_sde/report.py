#!/usr/bin/env python3
"""Reporting utilities for derivatives SDE decomposition."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def decomposition_report(symbol: str, df: pd.DataFrame) -> str:
    start, end = df.index.min(), df.index.max()
    mu_stats = df["mu"].describe()
    sigma_stats = df["sigma"].describe()
    lambda_stats = df["Lambda"].describe()
    active_share = float(df["active"].mean())

    top_lambda = df.nlargest(20, "Lambda")[["mu", "sigma", "Lambda", "r", "z_funding", "z_oi_change", "z_basis"]]
    lines = [
        f"# Derivatives SDE Decomposition - {symbol}",
        "",
        f"- Coverage: {start} to {end}",
        f"- Active share: {active_share:.3f}",
        "",
        "## Mu statistics",
        mu_stats.to_frame("mu").to_markdown(),
        "",
        "## Sigma statistics",
        sigma_stats.to_frame("sigma").to_markdown(),
        "",
        "## Lambda statistics",
        lambda_stats.to_frame("Lambda").to_markdown(),
        "",
        "## Top-20 Lambda timestamps",
        top_lambda.to_markdown(),
    ]
    return "\n".join(lines)


def evaluation_report(eval_results: Dict[str, dict]) -> str:
    lines = ["# Derivatives SDE Evaluation", ""]
    for symbol, metrics in eval_results.items():
        lines.append(f"## {symbol}")
        for horizon, res in metrics.items():
            lines.append(f"### Horizon {horizon}h")
            table = pd.DataFrame(
                [res["active"], res["inactive"], res["shuffled"]],
                index=["active", "inactive", "shuffled"],
            )
            table["active_share"] = res.get("active_share", float("nan"))
            lines.append(table.to_markdown())
            if res.get("lambda_deciles"):
                dec_table = pd.DataFrame(res["lambda_deciles"]).T
                lines.append("Lambda decile monotonicity:")
                lines.append(dec_table.to_markdown())
            lines.append("")
    return "\n".join(lines)


def write_report(content: str, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content)
