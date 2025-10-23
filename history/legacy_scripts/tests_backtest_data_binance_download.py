import time, csv, requests
ENDPOINT = "https://api.binance.com/api/v3/klines"
def fetch_klines(symbol, interval="5m", limit=1000, start_ms=None, end_ms=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None: params["startTime"] = int(start_ms)
    if end_ms is not None: params["endTime"] = int(end_ms)
    r = requests.get(ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    return r.json()
def download_to_csv(symbol, interval, out_csv, limit=10000, start_ms=None, end_ms=None):
    per_call = 1000; data = []; start = start_ms; remaining = limit if limit else None
    while True:
        need = per_call if (remaining is None) else min(per_call, remaining)
        batch = fetch_klines(symbol, interval, need, start_ms=start, end_ms=end_ms)
        if not batch: break
        data.extend(batch); last_t = batch[-1][0]; start = last_t + 1
        if remaining is not None:
            remaining -= len(batch)
            if remaining <= 0: break
        time.sleep(0.2)
        if len(batch) < per_call: break
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["timestamp","open","high","low","close","volume"])
        for k in data: w.writerow([k[0], k[1], k[2], k[3], k[4], k[5]])
    return out_csv
