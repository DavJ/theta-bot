#!/usr/bin/env python3
"""
Build basis series from spot close and mark close:
  basis(t) = log(mark_close(t)) - log(spot_close(t))
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _load_price_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip")
    ts_col = None
    for candidate in ("close_time_ms", "timestamp_ms", "timestamp"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {path}")
    df = df.rename(columns={ts_col: "timestamp_ms"})
    df = df.sort_values("timestamp_ms")
    df.index = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df


def build_basis_for_symbol(
    symbol: str,
    interval: str,
    spot_dir: Path,
    futures_dir: Path,
    force: bool = False,
) -> Optional[Path]:
    spot_path = spot_dir / f"{symbol}_{interval}.csv.gz"
    mark_path = futures_dir / f"{symbol}_mark_{interval}.csv.gz"
    out_path = futures_dir / f"{symbol}_basis.csv.gz"

    if out_path.exists() and not force:
        print(f"[cache] {out_path} exists; skip (use --force to recompute)")
        return out_path

    if not spot_path.exists() or not mark_path.exists():
        print(f"Missing inputs for {symbol}: {spot_path} or {mark_path}")
        return None

    spot_df = _load_price_series(spot_path).rename(columns={"close": "spot_close"})
    mark_df = _load_price_series(mark_path).rename(columns={"close": "mark_close"})

    merged = pd.concat([spot_df["spot_close"], mark_df["mark_close"]], axis=1, join="inner").dropna()
    if merged.empty:
        print(f"No overlapping data for {symbol}")
        return None

    merged["basis"] = np.log(merged["mark_close"]) - np.log(merged["spot_close"])
    merged["timestamp_ms"] = (merged.index.view("int64") // 1_000_000).astype("int64")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged[["timestamp_ms", "basis", "mark_close", "spot_close"]].to_csv(
        out_path, index=False, compression="gzip"
    )
    print(f"{symbol}: wrote basis to {out_path} ({len(merged)} rows)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build basis from spot and mark closes.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--interval", default="1h", help="Interval suffix (default: 1h)")
    parser.add_argument("--spot-dir", default="data/raw/spot", help="Directory containing spot klines")
    parser.add_argument("--futures-dir", default="data/raw/futures", help="Directory containing mark klines")
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()

    spot_dir = Path(args.spot_dir)
    futures_dir = Path(args.futures_dir)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    for sym in symbols:
        build_basis_for_symbol(sym, args.interval, spot_dir, futures_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
