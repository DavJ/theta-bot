import csv, datetime

def parse_ts(ts):
    # supports unix ms or ISO8601
    try:
        ts = int(ts)
        if ts > 1e12: # ms
            ts = ts // 1000
        return ts
    except Exception:
        return int(datetime.datetime.fromisoformat(ts).timestamp())

class CSVFeed:
    def __init__(self, symbol: str, path: str):
        self.symbol = symbol
        self.rows = []
        with open(path, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                self.rows.append({
                    "timestamp": parse_ts(row["timestamp"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })
        self.idx = 0

    def next_row(self):
        if self.idx >= len(self.rows):
            return None
        row = self.rows[self.idx]
        self.idx += 1
        return row
