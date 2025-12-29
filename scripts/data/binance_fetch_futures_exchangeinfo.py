#!/usr/bin/env python3
"""
Fetch Binance futures exchange info metadata and save to JSON.

This provides contract metadata including delivery dates, contract types, and expiry features.

Example:
  python scripts/data/binance_fetch_futures_exchangeinfo.py \
    --out data/metadata/futures_exchangeInfo.json

Endpoint: /fapi/v1/exchangeInfo
Public endpoint - no API key required.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict

import requests


BINANCE_FAPI_BASE = "https://fapi.binance.com"


def fetch_futures_exchange_info(
    timeout_s: int = 30,
    max_retries: int = 6,
) -> Dict[str, Any]:
    """
    Fetch futures exchange info from Binance API.
    
    Returns complete exchangeInfo response with all symbols and metadata.
    """
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/exchangeInfo"
    session = requests.Session()
    
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=timeout_s)
            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", "1"))
                time.sleep(max(retry_after, backoff))
                backoff = min(backoff * 1.8, 20.0)
                continue
            r.raise_for_status()
            data = r.json()
            return data
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 1.8, 20.0)
    
    return {}


def extract_contract_metadata(exchange_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant contract metadata from exchangeInfo.
    
    Returns a simplified structure focusing on:
    - symbol
    - contractType (PERPETUAL, CURRENT_QUARTER, NEXT_QUARTER, etc.)
    - deliveryDate
    - onboardDate
    - status
    """
    symbols = exchange_info.get("symbols", [])
    
    metadata = {
        "fetchTime": exchange_info.get("serverTime"),
        "timezone": exchange_info.get("timezone", "UTC"),
        "contracts": []
    }
    
    for sym in symbols:
        contract = {
            "symbol": sym.get("symbol"),
            "pair": sym.get("pair"),
            "contractType": sym.get("contractType"),
            "deliveryDate": sym.get("deliveryDate"),
            "onboardDate": sym.get("onboardDate"),
            "status": sym.get("status"),
            "baseAsset": sym.get("baseAsset"),
            "quoteAsset": sym.get("quoteAsset"),
            "marginAsset": sym.get("marginAsset"),
            "pricePrecision": sym.get("pricePrecision"),
            "quantityPrecision": sym.get("quantityPrecision"),
        }
        metadata["contracts"].append(contract)
    
    return metadata


def write_json(path: str, data: Dict[str, Any], pretty: bool = True) -> None:
    """
    Write JSON data to file.
    """
    with open(path, "w") as f:
        if pretty:
            json.dump(data, f, indent=2)
        else:
            json.dump(data, f)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch Binance futures exchange info metadata"
    )
    ap.add_argument("--out", required=True, help="Output path, e.g. data/metadata/futures_exchangeInfo.json")
    ap.add_argument("--full", action="store_true", help="Save full response (default: extract metadata only)")
    ap.add_argument("--pretty", action="store_true", default=True, help="Pretty print JSON (default: True)")
    args = ap.parse_args()

    print("Fetching futures exchange info from Binance...")
    exchange_info = fetch_futures_exchange_info()
    
    if not exchange_info:
        print("ERROR: Failed to fetch exchange info", file=sys.stderr)
        return 1
    
    if args.full:
        data_to_save = exchange_info
        print(f"Fetched complete exchange info with {len(exchange_info.get('symbols', []))} symbols")
    else:
        data_to_save = extract_contract_metadata(exchange_info)
        print(f"Extracted metadata for {len(data_to_save['contracts'])} contracts")
    
    write_json(args.out, data_to_save, pretty=args.pretty)
    print(f"Saved to {args.out}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
