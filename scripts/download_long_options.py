"""Build causal monthly option entry proxies from official public Deribit trades.

This downloader has no account credentials or order methods. A historical trade
is a pricing observation, not an executable historical ask or depth guarantee.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode

import pandas as pd
import requests

from scripts.download_spot_flow import BEGIN, END
from scripts.research_spot_flow import load_spot_flow

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / 'docs/evaluation/LONG_OPTIONS_RESEARCH_PLAN.md'
SIGNALS = ('call_atm', 'call_otm', 'direction_atm', 'direction_otm')


def api_get(host, method, params, path):
    """Cache raw immutable trade observations; surface API and HTTP failures."""
    url = f'https://{host}/api/v2/public/{method}?{urlencode(params)}'
    if not path.exists():
        for attempt in range(3):
            response = requests.get(url, timeout=45)
            if response.status_code in (429, 502, 503, 504) and attempt < 2:
                time.sleep(1 + attempt)
                continue
            response.raise_for_status()
            payload = response.content
            parsed = json.loads(payload)
            if 'error' in parsed or 'result' not in parsed:
                raise ValueError(f'Deribit API failure: {parsed.get("error")}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            break
        else:
            raise ValueError('Deribit request retries exhausted')
    payload = path.read_bytes()
    parsed = json.loads(payload)
    if 'error' in parsed or 'result' not in parsed:
        raise ValueError('Invalid cached Deribit response')
    return parsed['result'], {'url': url, 'file': path.name,
        'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def monthly_clock(start=BEGIN, end=END):
    for month in pd.date_range(start, end, freq='MS', inclusive='left'):
        entry = month + pd.Timedelta(days=(0 - month.dayofweek) % 7)
        last = month + pd.offsets.MonthEnd(0)
        expiry = last - pd.Timedelta(days=(last.dayofweek - 4) % 7) + pd.Timedelta(hours=8)
        yield entry, expiry


def select_contract(instruments, asset, family, side, strike_target, entry, expiry):
    """Contract choice cannot depend on its later trades or expiry payoff."""
    wanted = asset if family == 'inverse' else asset + '_USDC'
    eligible = [r for r in instruments
        if r['instrument_name'].split('-')[0] == wanted and r['option_type'] == side
        and r['creation_timestamp'] <= entry.value // 1_000_000
        and r['expiration_timestamp'] == expiry.value // 1_000_000]
    return min(eligible, key=lambda r: (abs(r['strike'] - strike_target), r['strike'])) if eligible else None


def download(folder, spot_folder):
    folder.mkdir(parents=True, exist_ok=True)
    frozen_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    (folder / 'frozen_protocol.json').write_text(json.dumps({'protocol_sha256': frozen_hash}, indent=2) + '\n')
    markets, spot_manifest = load_spot_flow(spot_folder)
    daily = {s: f.close.resample('1D').last() for s, f in markets.items()}
    momentum = {s: x.pct_change(30, fill_method=None) for s, x in daily.items()}
    source_files, instruments = [], []
    for currency in ('BTC', 'ETH', 'USDC'):
        result, proof = api_get('history.deribit.com', 'get_instruments',
            {'currency': currency, 'kind': 'option', 'expired': 'true'}, folder / f'{currency}_expired_instruments.json')
        instruments.extend(result)
        source_files.append(proof)
        print(f'{currency}: {len(result)} expired option metadata records', flush=True)
    delivery = {}
    for asset in ('BTC', 'ETH'):
        offset, prices = 0, {}
        while True:
            result, proof = api_get('www.deribit.com', 'get_delivery_prices',
                {'index_name': asset.lower() + '_usd', 'count': 1000, 'offset': offset},
                folder / f'{asset}_delivery_{offset}.json')
            source_files.append(proof)
            rows = result['data']
            for row in rows:
                prices[str(row['date'])] = float(row['delivery_price'])
            if not rows or min(pd.Timestamp(r['date'], tz='UTC') for r in rows) <= BEGIN:
                break
            offset += len(rows)
            if offset > 10000:
                raise ValueError('Delivery history paging did not reach study start')
        delivery[asset] = prices
    grouped = {}
    for r in instruments:
        prefix = r['instrument_name'].split('-')[0]
        if prefix in ('BTC', 'ETH', 'BTC_USDC', 'ETH_USDC'):
            grouped.setdefault((prefix, r['expiration_timestamp'], r['option_type']), []).append(r)
    observations, tasks = [], {}
    for entry, expiry in monthly_clock():
        for family in ('inverse', 'linear'):
            for policy in SIGNALS:
                for asset in ('BTC', 'ETH'):
                    symbol, previous = asset + 'USDT', entry - pd.Timedelta(days=1)
                    value = momentum[symbol].get(previous, float('nan'))
                    base = {'entry': str(entry), 'expiry': str(expiry), 'family': family,
                            'policy': policy, 'asset': asset, 'momentum_30d': float(value) if pd.notna(value) else None}
                    if pd.isna(value) or value == 0 or (policy.startswith('call') and value < 0):
                        observations.append({**base, 'status': 'inactive_signal'})
                        continue
                    side = 'call' if value > 0 else 'put'
                    close = float(daily[symbol].loc[previous])
                    target = close * (1.05 if side == 'call' else .95) if policy.endswith('otm') else close
                    prefix = asset if family == 'inverse' else asset + '_USDC'
                    candidates = grouped.get((prefix, expiry.value // 1_000_000, side), [])
                    contract = select_contract(candidates, asset, family, side, target, entry, expiry)
                    if contract is None:
                        observations.append({**base, 'status': 'no_listed_contract'})
                        continue
                    if float(contract['contract_size']) != 1.:
                        raise ValueError('BTC/ETH contract multiplier requires explicit modeling')
                    expiry_price = delivery[asset].get(expiry.strftime('%Y-%m-%d'))
                    if expiry_price is None or expiry_price <= 0:
                        raise ValueError(f'Missing official delivery price {asset} {expiry}')
                    for delay in (0, 1):
                        when = entry + pd.Timedelta(hours=4 * delay)
                        key = contract['instrument_name'], when.value // 1_000_000
                        tasks[key] = (key, when)
                        observations.append({**base, 'status': 'pending_trade', 'delay_bars': delay,
                            'execution_time': str(when), 'signal_close': close, 'contract': contract,
                            'official_delivery_price': expiry_price, 'task_key': list(key)})
    cache = folder / 'trade_cache'
    cache.mkdir(exist_ok=True)
    def get_trade(task):
        key, when = task
        params = {'instrument_name': key[0], 'start_timestamp': (when - pd.Timedelta(hours=4)).value // 1_000_000,
                  'end_timestamp': when.value // 1_000_000 - 1, 'count': 1, 'sorting': 'desc', 'include_old': 'true'}
        name = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest() + '.json'
        result, proof = api_get('history.deribit.com', 'get_last_trades_by_instrument_and_time', params, cache / name)
        rows = result['trades']
        if len(rows) > 1:
            raise ValueError('More than the requested last trade returned')
        trade = rows[0] if rows else None
        if trade is not None and not (params['start_timestamp'] <= trade['timestamp'] <= params['end_timestamp']):
            raise ValueError('Noncausal trade timestamp')
        return key, trade, proof
    trade_results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, (key, trade, proof) in enumerate(pool.map(get_trade, tasks.values()), 1):
            trade_results[key] = (trade, proof)
            if n % 50 == 0:
                print(f'Historical option entry windows {n}/{len(tasks)}', flush=True)
    for row in observations:
        if row['status'] != 'pending_trade':
            continue
        trade, proof = trade_results[tuple(row.pop('task_key'))]
        row['trade_source'] = proof
        if trade is None:
            row['status'] = 'no_recent_trade'
        elif not all(float(trade.get(k, 0)) > 0 for k in ('price', 'index_price', 'iv')):
            row['status'] = 'invalid_trade_model_inputs'
        else:
            row.update(status='observed_trade', trade=trade,
                trade_age_seconds=(pd.Timestamp(row['execution_time']).value // 1_000_000 - trade['timestamp']) / 1000)
    result = {'source': 'deribit_public_historical_trades', 'protocol_sha256': frozen_hash,
        'retrieved_at': str(pd.Timestamp.now(tz='UTC')), 'source_files': source_files,
        'spot_dataset_hashes': {s: r['sha256'] for s, r in spot_manifest['datasets'].items()},
        'trade_request_count': len(tasks), 'records': observations,
        'execution_prices_are_trade_proxies': True, 'historical_order_book_available': False}
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != frozen_hash:
        raise ValueError('Protocol changed during download')
    target = folder / 'entry_observations.json'
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {len(observations)} monthly signal/contract observations to {target}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw/long_options'))
    parser.add_argument('--spot-data', type=Path, default=Path('data/raw/spot_flow'))
    args = parser.parse_args()
    download(args.data, args.spot_data)


if __name__ == '__main__':
    main()
