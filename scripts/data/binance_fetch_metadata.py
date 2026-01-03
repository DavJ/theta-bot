#!/usr/bin/env python3
"""
Fetch Binance USD-M futures metadata and store exchangeInfo plus a compact symbols table.

Usage:
  python scripts/data/binance_fetch_metadata.py \
    --out data/metadata/futures_exchangeInfo.json \
    --symbols-table data/metadata/futures_symbols.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


BINANCE_FAPI_BASE = "https://fapi.binance.com"


def fetch_futures_exchange_info(
    session: requests.Session,
    timeout_s: int = 30,
    max_retries: int = 6,
) -> Dict[str, Any]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/exchangeInfo"
    backoff = 1.0
    for attempt in range(max_retries):
        resp = session.get(url, timeout=timeout_s)
        if resp.status_code in (429, 418) or resp.status_code >= 500:
            retry_after = float(resp.headers.get("Retry-After", "1"))
            backoff = min(max(retry_after, backoff), 20.0)
            session.headers.update({"X-Client-Attempt": str(attempt + 1)})
            time.sleep(backoff)
            backoff *= 1.8
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Failed to fetch exchangeInfo after retries")


def extract_symbols_table(exchange_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sym in exchange_info.get("symbols", []):
        rows.append(
            {
                "symbol": sym.get("symbol"),
                "pair": sym.get("pair"),
                "contractType": sym.get("contractType"),
                "quoteAsset": sym.get("quoteAsset"),
                "status": sym.get("status"),
            }
        )
    return rows


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Binance futures metadata.")
    parser.add_argument("--out", default="data/metadata/futures_exchangeInfo.json")
    parser.add_argument("--symbols-table", default="data/metadata/futures_symbols.csv")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    out_path = Path(args.out)
    table_path = Path(args.symbols_table)
    if out_path.exists() and not args.force:
        print(f"[cache] {out_path} exists; skip fetch (use --force to refresh)")
        if not table_path.exists():
            exchange_info = json.load(out_path.open())
            write_csv(table_path, extract_symbols_table(exchange_info))
        return 0

    session = requests.Session()
    exchange_info = fetch_futures_exchange_info(session)
    write_json(out_path, exchange_info)
    write_csv(table_path, extract_symbols_table(exchange_info))

    print(f"Saved exchangeInfo to {out_path}")
    print(f"Saved symbols table to {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
