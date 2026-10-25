"""Explicit public spot-data snapshot; no credentials, orders or background job."""
import argparse
from pathlib import Path

import requests

from spot_bot.forward_validation import SOURCE, SYMBOLS, utc_now, utc, read_protocol, write_feed_snapshot


def download(protocol_path, out, fetch=None):
    protocol = read_protocol(protocol_path)
    observed = utc_now()
    first = int(utc(protocol['warmup_start']).timestamp() * 1000)
    end = int(observed.floor('4h').timestamp() * 1000) - 1

    def public_page(symbol, begin, finish):
        response = requests.get(SOURCE, params={'symbol': symbol, 'interval': '4h', 'timeZone': '0',
            'startTime': begin, 'endTime': finish, 'limit': 1000}, timeout=25)
        response.raise_for_status()
        return response.json()

    fetch = fetch or public_page
    raw = {}
    for symbol in SYMBOLS:
        cursor, rows = first, []
        for _ in range(20):
            if cursor > end:
                break
            page = fetch(symbol, cursor, end)
            if not isinstance(page, list) or not page or len(page) > 1000:
                raise ValueError('Missing or oversized public spot kline page')
            # Check page continuity here; the snapshot validates all prices and clocks.
            if any(not isinstance(row, list) or len(row) != 12 for row in page):
                raise ValueError('Malformed public spot page')
            times = [row[0] for row in page]
            expected = list(range(cursor, cursor + len(page) * 14400000, 14400000))
            if times != expected or times[-1] > end:
                raise ValueError('Public source returned a gap, duplicate or wrong time range')
            rows.extend(page)
            cursor = times[-1] + 14400000
        if cursor <= end:
            raise ValueError('Bounded public download did not cover the required account history')
        raw[symbol] = rows
    return write_feed_snapshot(out, raw, protocol, observed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('data/raw/forward_spot'))
    args = parser.parse_args()
    print(download(args.protocol, args.out))


if __name__ == '__main__':
    main()
