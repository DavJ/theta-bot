"""Freeze, capture and audit bounded public spot-depth cost probes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import zipfile

import requests

from theta_bot_averaging.depth_paper import SYMBOLS, Book, Rules, json_values, round_trip_probe

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'docs/evaluation/DEPTH_PAPER_EXECUTION_PLAN.md'
BASE = 'https://data-api.binance.vision'
ENDPOINTS = ('/api/v3/exchangeInfo','/api/v3/depth')
SCENARIOS = {'nominal': {'fee_rate': '0.001', 'reserve_bps': '0'},
             'stress': {'fee_rate': '0.002', 'reserve_bps': '5'}}
SETTINGS = {'symbols': SYMBOLS, 'budgets': ['25','100','1000','2500'], 'capital': '10000',
    'depth_limit': 100, 'scenarios': SCENARIOS, 'maximum_request_ms': 5000,
    'clock_tolerance_ms': 250, 'maximum_metadata_age_ms': 3600000,
    'minimum_batches_per_market': 200, 'minimum_span_days': 30,
    'source_book_age_known': False, 'strategy_profitability_test': False}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def now():
    return datetime.now(timezone.utc)


def utc(value):
    try:
        ts = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Explicit ISO timestamp required') from exc
    if ts.tzinfo is None:
        raise ValueError('Explicit timestamp timezone required')
    return ts.astimezone(timezone.utc)


def fingerprints():
    paths = sorted((ROOT / 'spot_bot').rglob('*.py'))
    paths += [ROOT / p for p in ('theta_bot_averaging/depth_paper.py',
        'scripts/research_paper_execution.py', 'requirements.txt',
        'docs/evaluation/DEPTH_PAPER_EXECUTION_PLAN.md')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def runtime():
    return {'python': '.'.join(map(str, sys.version_info[:3])),
            'requests': importlib.metadata.version('requests')}


def exclusive(path, value):
    """An existing identical receipt is idempotent; a changed receipt is rejected."""
    path = Path(path)
    content = canonical(value) + b'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as f:
            f.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError('Existing immutable evidence differs')


def freeze(path):
    body = {'schema': 1, 'study_kind': 'public_depth_cost_calibration',
        'frozen_at': now().isoformat(), 'settings': json.loads(canonical(SETTINGS)),
        'source_sha256': fingerprints(), 'runtime': runtime(), 'live_eligible': False}
    proof = {**body, 'protocol_sha256': digest(body)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(canonical(proof).decode() + '\n')
    return proof


def read_protocol(path):
    proof = json.loads(Path(path).read_text())
    body = {k:v for k,v in proof.items() if k != 'protocol_sha256'}
    if (proof.get('protocol_sha256') != digest(body) or proof.get('schema') != 1
            or proof.get('study_kind') != 'public_depth_cost_calibration'
            or canonical(proof.get('settings')) != canonical(SETTINGS)
            or proof.get('source_sha256') != fingerprints() or proof.get('runtime') != runtime()
            or proof.get('live_eligible') is not False or utc(proof['frozen_at']) > now()):
        raise ValueError('Frozen source, runtime, protocol or settings changed')
    return proof


def envelope(record, endpoint, symbol, observed):
    if (record.get('source_url') not in (BASE + endpoint, 'https://api.binance.com' + endpoint)
            or record.get('method') != 'GET' or record.get('transport') not in ('public_https','binance_mcp')):
        raise ValueError('Unsupported public spot source')
    params = record.get('params', {})
    expected = {'symbol': symbol} if endpoint == ENDPOINTS[0] else {'symbol':symbol,'limit':100}
    extra = {'showPermissionSets'} if endpoint == ENDPOINTS[0] else {'symbolStatus'}
    if any(params.get(k) != v for k,v in expected.items()) or set(params)-set(expected)-extra:
        raise ValueError('Source request parameters changed')
    if ('symbolStatus' in params and params['symbolStatus'] != 'TRADING'
            or 'showPermissionSets' in params and type(params['showPermissionSets']) is not bool):
        raise ValueError('Unsupported public request filter')
    started, received = utc(record['request_started_at']), utc(record['received_at'])
    elapsed = record.get('elapsed_ms')
    wall = (received-started).total_seconds()*1000
    if (type(elapsed) not in (int,float) or isinstance(elapsed,bool)
            or not 0 <= elapsed <= SETTINGS['maximum_request_ms'] or wall < 0
            or abs(wall-elapsed) > SETTINGS['clock_tolerance_ms'] or received > observed):
        raise ValueError('Slow, future or inconsistent observation clock')
    if record.get('error') is not None or not isinstance(record.get('payload'), dict):
        raise ValueError('Public source returned an error or malformed JSON')
    return started, received, record['payload']


def public_get(endpoint, params):
    """The fixed public-data-only host and two GET paths are the entire network surface."""
    if endpoint not in ENDPOINTS:
        raise ValueError('Unsupported read-only public endpoint')
    started, clock = now(), time.monotonic()
    record = {'source_url': BASE+endpoint, 'method':'GET', 'transport':'public_https', 'params':params,
              'request_started_at': started.isoformat()}
    try:
        response = requests.get(BASE+endpoint, params=params, timeout=5, allow_redirects=False)
        if response.status_code != 200:
            raise ValueError(f'Public HTTP status {response.status_code}')
        record['payload'] = response.json()
    except (requests.RequestException, ValueError) as exc:
        record['error'] = str(exc)
    record.update(received_at=now().isoformat(), elapsed_ms=round((time.monotonic()-clock)*1000,3))
    return record


def collect(proof, fetch=public_get):
    records = {}
    for symbol in SYMBOLS:
        info = fetch(ENDPOINTS[0], {'symbol':symbol})
        depth = fetch(ENDPOINTS[1], {'symbol':symbol,'limit':100})
        records[symbol] = {'info':info, 'depth':depth}
    return {'schema':1, 'protocol_sha256':proof['protocol_sha256'],
            'observed_at':now().isoformat(), 'markets':records}


def audit_case(case, depth, rules, scenario):
    """Recalculate both fills with exact rational arithmetic, not the depth engine."""
    q = Fraction(case['base_quantity'])
    capital = Fraction(case['funded_quote'])
    cash, inventory, max_error = capital, Fraction(0), Fraction(0)
    rate, reserve = Fraction(scenario['fee_rate']), Fraction(scenario['reserve_bps'])/10000
    if q <= 0 or q % Fraction(rules.step) or q > Fraction(rules.maximum_qty):
        raise ValueError('Audit detected invalid filled quantity')
    spent = received = Fraction(0)
    for side, name, direction in (('buy','asks',1),('sell','bids',-1)):
        left, raw, count = q, Fraction(0), 0
        for price, size in depth[name]:
            count += 1
            take = min(left,Fraction(size))
            raw += take*Fraction(price)
            left -= take
            if left == 0:
                break
        if left:
            raise ValueError('Audit detected fabricated depth')
        notional = raw*(1+direction*reserve)
        unit = Fraction(rules.fee_quantum)
        units = notional*rate/unit
        commission = -(-units.numerator//units.denominator)*unit
        cash -= direction*notional+commission
        inventory += direction*q
        row = case[side]
        checks = {'quantity':q, 'raw_book_notional':raw, 'notional':notional,
            'fee_quote':commission, 'average_price':notional/q,
            'cash_after':cash, 'inventory_after':inventory}
        if row['side'] != side or row['levels_used'] != count or min(cash,inventory) < 0:
            raise ValueError('Independent paper ledger is invalid')
        for key,value in checks.items():
            max_error = max(max_error,abs(Fraction(row[key])-value))
        if side == 'buy': spent=notional+commission
        else: received=notional-commission
    expected = {'quote_spent':spent, 'quote_received':received, 'round_trip_cost_quote':spent-received,
        'cash_after_roundtrip':cash, 'round_trip_cost_bps':(spent-received)/spent*10000}
    for key,value in expected.items():
        max_error = max(max_error,abs(Fraction(case[key])-value))
    if (spent > Fraction(case['budget_including_fee']) or inventory != 0
            or Fraction(case['budget_including_fee']) > capital/4 or max_error > Fraction(1,10**24)):
        raise ValueError('Independent paper accounting does not reconcile')
    return {'reconciled':True, 'maximum_rational_residual':float(max_error)}


def analyze(proof, capture):
    observed = utc(capture['observed_at'])
    if (capture.get('schema') != 1 or capture.get('protocol_sha256') != proof['protocol_sha256']
            or set(capture.get('markets',{})) != set(SYMBOLS)
            or observed > now() or observed < utc(proof['frozen_at'])):
        raise ValueError('Wrong, future or prefrozen capture')
    results = {}
    for symbol in SYMBOLS:
        rows = capture['markets'][symbol]
        try:
            _, info_received, info = envelope(rows['info'],ENDPOINTS[0],symbol,observed)
            started, received, depth = envelope(rows['depth'],ENDPOINTS[1],symbol,observed)
            age = (started-info_received).total_seconds()*1000
            if started < utc(proof['frozen_at']) or not 0 <= age <= SETTINGS['maximum_metadata_age_ms']:
                raise ValueError('Metadata is stale or the book predates its frozen protocol')
            rules = Rules.from_info(info,symbol)
            book = Book.from_depth(depth,rules)
            cases = {}
            for scenario, costs in SCENARIOS.items():
                cases[scenario] = {}
                for budget in SETTINGS['budgets']:
                    try:
                        value = json_values(round_trip_probe(book,rules,budget,**costs))
                        value['audit'] = audit_case(value,depth,rules,costs)
                        cases[scenario][budget] = {'status':'conditional_simulation',**value}
                    except ValueError as exc:
                        cases[scenario][budget] = {'status':'rejected','reason':str(exc)}
            results[symbol] = {'status':'observed', 'quote_asset':rules.quote,
                'book_update_id':book.update_id, 'depth_payload_sha256':digest(depth),
                'received_at':received.isoformat(), 'request_elapsed_ms':rows['depth']['elapsed_ms'],
                'source_book_age_ms':None, 'spread_bps':str(book.spread_bps), 'cases':cases}
        except (ValueError, KeyError, TypeError) as exc:
            results[symbol] = {'status':'rejected','reason':str(exc), 'source_book_age_ms':None}
    return {'schema':1, 'study_kind':'public_depth_cost_calibration',
        'protocol_sha256':proof['protocol_sha256'], 'capture_sha256':digest(capture),
        'observed_at':capture['observed_at'], 'results':results,
        'observed_executable_fills':0, 'strategy_pnl_available':False,
        'source_book_age_known':False, 'account_eligibility_verified':False, 'live_eligible':False}


def save_and_analyze(proof, capture, out):
    out = Path(out)
    key = digest(capture)
    # Preserve raw failures as well as successful observations.
    exclusive(out/'captures'/f'{key}.json',capture)
    report = analyze(proof,capture)
    exclusive(out/'reports'/f'{key}.json',report)
    coverage = {}
    reports = [json.loads(p.read_text()) for p in sorted((out/'reports').glob('*.json'))]
    for row in reports:
        capture_path=out/'captures'/f"{row['capture_sha256']}.json"
        raw=json.loads(capture_path.read_text())
        if digest(raw)!=row['capture_sha256'] or row!=analyze(proof,raw):
            raise ValueError('Prior immutable paper evidence changed')
    for symbol in SYMBOLS:
        distinct = {}
        for row in reports:
            result=row['results'][symbol]
            if result['status']=='observed' and all(
                    c['status']=='conditional_simulation' for s in result['cases'].values() for c in s.values()):
                distinct.setdefault(result['depth_payload_sha256'],utc(result['received_at']))
        times=list(distinct.values())
        span=(max(times)-min(times)).total_seconds()/86400 if times else 0.
        coverage[symbol]={'distinct_successful_books':len(times),'span_days':span,
            'coverage_gate_passed':len(times)>=200 and span>=30}
    (out/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    return report,coverage


def export(report, coverage, out, destination):
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    stem='THETABOT_DEPTH_PAPER_2026-10-23'
    (destination/(stem+'.json.gz')).write_bytes(gzip.compress(canonical({'report':report,'coverage':coverage}),mtime=0))
    lines=['# Public spot-depth paper calibration — 2026-10-23','',
        '**Conditional static execution-cost probes; observed exchange fills: 0.**','',
        'Six markets, four all-in buy budgets and two frozen fee/reserve scenarios.',
        'Each independently funded probe buys visible asks then sells the acquired inventory into the same book.',
        'USDT and USDC accounts are separate; actual commissions and account eligibility are unknown.','',
        '| Market | Full spread (bps) | 100 nominal round-trip (bps) | 2,500 nominal (bps) | 2,500 stress (bps) | Request ms |',
        '|---|---:|---:|---:|---:|---:|']
    accepted = rejected = 0
    for symbol,row in report['results'].items():
        if row['status']!='observed':
            lines += [f"| {symbol} | rejected: {row['reason']} | — | — | — | — |"]
            continue
        values=[]
        for scenario,budget in (('nominal','100'),('nominal','2500'),('stress','2500')):
            case=row['cases'][scenario][budget]
            values.append(f"{float(case['round_trip_cost_bps']):.4f}" if case['status']=='conditional_simulation' else 'rejected')
        lines += [f"| {symbol} | {float(row['spread_bps']):.4f} | "+' | '.join(values)+f" | {row['request_elapsed_ms']:.1f} |"]
        for cases in row['cases'].values():
            for case in cases.values():
                accepted+=case['status']=='conditional_simulation'
                rejected+=case['status']=='rejected'
    lines += ['',f'Accepted conditional probes: {accepted}; rejected probes on valid books: {rejected}.',
        'Both legs of every accepted probe independently reconcile using rational arithmetic.','',
        f"Protocol: `{report['protocol_sha256']}`. Capture: `{report['capture_sha256']}`.",
        f"Observed at: {report['observed_at']}.",'',
        '## Interpretation','',
        '- Nominal modeled fees are 0.1% per leg in quote; stress uses 0.2% plus a hypothetical 5bps adverse reserve per leg.',
        '- Depth REST replies contain no exchange book timestamp. Source book age remains unknown; the receipt clock is only local observation time.',
        '- These static round trips measure spread/depth/cost, not a momentum strategy return or realized exchange execution.',
        '- Market quantity filters are checked. VWAP notional checks do not establish reference-price, execution-rule or account-specific eligibility.',
        '- One batch does not estimate tail costs or justify reducing historical slippage assumptions.',
        '- The coverage gate requires 200 distinct successful books per market spanning 30 days.',
        '- No background collector or order submission is started. The earlier candle-based prospective protocol stays separate.','',
        '[Frozen plan](DEPTH_PAPER_EXECUTION_PLAN.md).','',
        '## Coverage','', '| Market | Distinct successful books | Span days | Coverage gate |','|---|---:|---:|:---:|']
    lines += [f"| {s} | {c['distinct_successful_books']} | {c['span_days']:.4f} | {c['coverage_gate_passed']} |" for s,c in coverage.items()]
    lines += ['', '## Reproduce / explicitly collect another batch','', '```bash',
        'python -m scripts.research_paper_execution freeze --protocol artifacts/depth_paper/protocol.json',
        'python -m scripts.research_paper_execution collect --protocol artifacts/depth_paper/protocol.json --out artifacts/depth_paper',
        '# Or replay a saved connector capture:',
        'python -m scripts.research_paper_execution replay --protocol artifacts/depth_paper/protocol.json --capture CAPTURE.json --out artifacts/depth_paper',
        '```','']
    (destination/(stem+'.md')).write_text('\n'.join(lines))
    archive=destination/(stem+'_EVIDENCE.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(out/'protocol.json','frozen_protocol.json')
        z.write(PLAN,PLAN.name)
        z.write(out/'coverage.json','coverage.json')
        for folder in ('captures','reports'):
            for path in sorted((out/folder).glob('*.json')):
                z.write(path,str(path.relative_to(out)))
    (destination/(stem+'_EVIDENCE.sha256')).write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','collect','replay'))
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--out',type=Path,default=Path('artifacts/depth_paper'))
    parser.add_argument('--report-out',type=Path,default=Path('docs/evaluation'))
    args=parser.parse_args()
    if args.action=='freeze':
        print(freeze(args.protocol)['protocol_sha256'])
        return
    proof=read_protocol(args.protocol)
    if args.action=='replay' and args.capture is None:
        parser.error('--capture is required for replay')
    capture=collect(proof) if args.action=='collect' else json.loads(args.capture.read_text())
    report,coverage=save_and_analyze(proof,capture,args.out)
    # Export a copy under a stable internal evidence name even with --protocol elsewhere.
    exclusive(args.out/'protocol.json',proof)
    export(report,coverage,args.out,args.report_out)
    print(json.dumps({'capture_sha256':report['capture_sha256'],'markets':
        {s:r['status'] for s,r in report['results'].items()},'coverage':coverage,
        'observed_executable_fills':0,'strategy_pnl_available':False,'live_eligible':False},indent=2))


if __name__=='__main__':
    main()
