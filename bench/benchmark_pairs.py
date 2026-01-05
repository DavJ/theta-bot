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
import subprocess
from pathlib import Path
import pandas as pd


DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "AVAX/USDT",
]


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
        "--csv-out",
        str(out_csv),
        "--csv-out-mode",
        "features",
    ]

    print(f"\n=== Exporting features for {symbol} -> {out_csv} ===")
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Feature export failed for {symbol} (exit {res.returncode})")


def summarize_features_csv(symbol: str, csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
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
        g = rv.groupby(q, observed=False).mean()
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

    return row


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
        )

        rows.append(summarize_features_csv(sym, csv_path))

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
