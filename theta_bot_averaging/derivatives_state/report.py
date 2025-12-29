#!/usr/bin/env python3
"""
Report generation helpers for drift analysis.

Generate markdown reports showing top timestamps by determinism.
"""

from __future__ import annotations

import pandas as pd


def generate_top_timestamps_report(
    df: pd.DataFrame,
    top_n: int = 20,
    symbol: str = "SYMBOL",
) -> str:
    """
    Generate markdown report of top timestamps by determinism D(t).
    
    Parameters
    ----------
    df : pd.DataFrame
        Drift data with columns: D, mu, z_funding, z_oi_change, z_basis
    top_n : int
        Number of top timestamps to include (default: 20)
    symbol : str
        Symbol name for report title
        
    Returns
    -------
    str
        Markdown formatted report
    """
    # Sort by D(t) descending
    df_sorted = df.sort_values("D", ascending=False)
    top_df = df_sorted.head(top_n)
    
    # Build report
    lines = [
        f"# Top {top_n} Timestamps by Determinism D(t) - {symbol}",
        "",
        "Showing timestamps with highest drift magnitude.",
        "",
        "| Timestamp | D(t) | mu(t) | z_funding | z_oi_change | z_basis |",
        "|-----------|------|-------|-----------|-------------|---------|",
    ]
    
    for idx, row in top_df.iterrows():
        timestamp = idx.strftime("%Y-%m-%d %H:%M")
        D_val = row.get("D", 0.0)
        mu_val = row.get("mu", 0.0)
        z_funding = row.get("z_funding", 0.0)
        z_oi_change = row.get("z_oi_change", 0.0)
        z_basis = row.get("z_basis", 0.0)
        
        lines.append(
            f"| {timestamp} | {D_val:.4f} | {mu_val:+.4f} | {z_funding:+.4f} | {z_oi_change:+.4f} | {z_basis:+.4f} |"
        )
    
    lines.append("")
    
    # Add summary statistics
    lines.extend([
        "## Summary Statistics",
        "",
        f"- Total records: {len(df)}",
        f"- Mean D(t): {df['D'].mean():.4f}",
        f"- Median D(t): {df['D'].median():.4f}",
        f"- Max D(t): {df['D'].max():.4f}",
        f"- 85th percentile D(t): {df['D'].quantile(0.85):.4f}",
        "",
    ])
    
    return "\n".join(lines)


def save_report(report: str, output_path: str) -> None:
    """
    Save markdown report to file.
    
    Parameters
    ----------
    report : str
        Markdown formatted report
    output_path : str
        Output file path
    """
    with open(output_path, "w") as f:
        f.write(report)
