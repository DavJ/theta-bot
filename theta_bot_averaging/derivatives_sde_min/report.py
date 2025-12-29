from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def _fmt(val, digits: int) -> str:
    if pd.isna(val):
        return "nan"
    return f"{val:.{digits}f}"


def write_decomposition_report(summaries: List[Dict], path: str = "reports/DERIVATIVES_SDE_DECOMPOSITION.md") -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Derivatives SDE Decomposition",
        "",
        "| Symbol | Rows | Window (UTC) | Active Share | Lambda Threshold |",
        "| --- | ---: | --- | ---: | ---: |",
    ]

    for s in summaries:
        window = f"{s['start']} → {s['end']}"
        lines.append(
            f"| {s['symbol']} | {s['rows']} | {window} | {s['active_share']:.3f} | {s['lambda_threshold']:.4f} |"
        )

    lines.append("")
    lines.append("## Top Lambda Events")
    for s in summaries:
        lines.append(f"### {s['symbol']}")
        if not s.get("top_events"):
            lines.append("No events available.")
            continue
        lines.append("| Timestamp (UTC) | Lambda | Mu | Sigma |")
        lines.append("| --- | ---: | ---: | ---: |")
        for evt in s["top_events"]:
            ts = pd.to_datetime(evt["timestamp"], unit="ms", utc=True)
            lines.append(
                f"| {ts} | {evt['lambda']:.4f} | {evt['mu']:.6f} | {evt['sigma']:.6f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote decomposition report to {out_path}")


def write_eval_report(eval_results: Dict[str, Dict], path: str = "reports/DERIVATIVES_SDE_EVAL.md") -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = ["# Derivatives SDE Evaluation", ""]
    for symbol, result in eval_results.items():
        lines.append(f"## {symbol}")
        lines.append(f"- Active share: {result.get('active_share', float('nan')):.3f}")
        lines.append(f"- Lambda threshold (q): {result.get('lambda_threshold', float('nan')):.4f}")
        lines.append("")
        lines.append("| Horizon (h) | Sign Agree | Effect Size | Inactive Mean | Shuffled Agree |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for h, metrics in sorted(result["per_horizon"].items()):
            lines.append(
                f"| {h} | {_fmt(metrics.get('sign_agree'), 4)} "
                f"| {_fmt(metrics.get('effect_size'), 6)} "
                f"| {_fmt(metrics.get('inactive_mean'), 6)} "
                f"| {_fmt(metrics.get('shuffled_sign_agree'), 4)} |"
            )
        lines.append("")
        lines.append("### Lambda Decile Means (active set)")
        for h, metrics in sorted(result["per_horizon"].items()):
            deciles = metrics.get("decile_means", [])
            if not deciles:
                lines.append(f"- Horizon {h}: insufficient data")
            else:
                decile_str = ", ".join(f"{x:.6f}" for x in deciles)
                lines.append(f"- Horizon {h}: {decile_str}")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote evaluation report to {out_path}")
