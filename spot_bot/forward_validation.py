"""Hash-bound prospective spot replay; never submits an exchange order."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4

import numpy as np
import pandas as pd

from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.execution_risk import completed_risk_covariances
from spot_bot.portfolio.momentum_refinement import momentum_refinement_targets
from spot_bot.portfolio.risk import PortfolioRisk

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://api.binance.com/api/v3/klines'
SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'BNBUSDT')
POLICIES = ('spot_control', 'guarded_edge')
SCENARIOS = ('nominal', 'cost_stress', 'delayed', 'joint_stress')
BAR = pd.Timedelta('4h')
SETTINGS = {'symbols': list(SYMBOLS), 'policies': list(POLICIES), 'scenarios': list(SCENARIOS),
    'initial_usdt': 10000., 'minimum_forward_days': 180, 'warmup_days': 120,
    'maximum_feed_age_hours': 8, 'max_exposure': .5, 'asset_cap': .25,
    'rebalance_band': .01, 'min_notional': 10., 'volatility_target': .15,
    'volatility_window': 60, 'covariance_shrinkage': .25,
    'drawdown_budget': .20, 'fee_rate': .001, 'slippage_bps': 5., 'spread_bps': 2.}


def utc_now():
    return pd.Timestamp.now(tz='UTC')


def utc(value):
    ts = pd.Timestamp(value)
    if ts.tz is None:
        raise ValueError('An explicit timezone is required')
    return ts.tz_convert('UTC').as_unit('ns')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def source_fingerprints():
    paths = sorted((ROOT / 'spot_bot').rglob('*.py'))
    paths += [ROOT / name for name in ('scripts/forward_spot.py', 'scripts/download_forward_spot.py',
              'docs/evaluation/FORWARD_VALIDATION_PLAN.md', 'requirements.txt')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def runtime_versions():
    return {'python': '.'.join(map(str, sys.version_info[:3])),
        **{p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy')}}


def freeze_protocol(path, start=None):
    now = utc_now()
    start = utc(start) if start is not None else now.normalize() + pd.Timedelta('1D')
    if start <= now or start != start.normalize():
        raise ValueError('Forward start must be a future UTC midnight')
    body = {'schema_version': 1, 'study_kind': 'prospectively_frozen_model_replay',
        'frozen_at': now.isoformat(), 'forward_start': start.isoformat(),
        'warmup_start': (start - pd.Timedelta(days=SETTINGS['warmup_days'])).isoformat(),
        'settings': json.loads(canonical(SETTINGS)), 'source_sha256': source_fingerprints(), 'runtime': runtime_versions(),
        'publication_timestamp_requires_external_anchor': True, 'live_eligible': False}
    protocol = {**body, 'protocol_sha256': digest(body)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(protocol, indent=2, allow_nan=False) + '\n')
    return protocol


def read_protocol(path):
    protocol = json.loads(Path(path).read_text())
    body = {k: v for k, v in protocol.items() if k != 'protocol_sha256'}
    if protocol.get('protocol_sha256') != digest(body):
        raise ValueError('Frozen protocol checksum mismatch')
    if (protocol.get('schema_version') != 1 or protocol.get('study_kind') != 'prospectively_frozen_model_replay'
            or canonical(protocol.get('settings')) != canonical(SETTINGS)
            or protocol.get('live_eligible') is not False
            or protocol.get('publication_timestamp_requires_external_anchor') is not True):
        raise ValueError('Unsupported frozen forward protocol')
    start, frozen = utc(protocol['forward_start']), utc(protocol['frozen_at'])
    if (frozen >= start or frozen > utc_now() or start != start.normalize()
            or utc(protocol['warmup_start']) != start - pd.Timedelta(days=SETTINGS['warmup_days'])):
        raise ValueError('Protocol is not frozen before its future-data boundary')
    if protocol['source_sha256'] != source_fingerprints():
        raise ValueError('Frozen strategy/evaluator source changed; create a separate new freeze')
    if protocol['runtime'] != runtime_versions():
        raise ValueError('Frozen runtime changed; reproduce it or create a separate new freeze')
    return protocol


def parse_closed_klines(rows, as_of):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty raw spot klines required')
    records = []
    for row in rows:
        if (not isinstance(row, list) or len(row) != 12
                or not isinstance(row[0], int) or isinstance(row[0], bool)
                or not isinstance(row[6], int) or isinstance(row[6], bool)
                or row[0] % 14400000 != 0 or row[6] != row[0] + 14399999):
            raise ValueError('Invalid UTC 4h spot open/close clock')
        values = np.array([row[i] for i in (1, 2, 3, 4, 5, 7, 8, 9, 10)], dtype=float)
        o, h, lo, c, volume, quote, count, buy, buy_quote = values
        if (not np.isfinite(values).all() or min(o, h, lo, c) <= 0
                or min(volume, quote, count, buy, buy_quote) < 0
                or lo > min(o, c) or h < max(o, c) or count % 1
                or buy > volume + 1e-8 or buy_quote > quote + 1e-8):
            raise ValueError('Invalid spot OHLCV/taker data')
        records.append([pd.to_datetime(row[0], unit='ms', utc=True), *values])
    frame = pd.DataFrame(records, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trade_count', 'taker_buy_base', 'taker_buy_quote']).set_index('timestamp')
    frame.index = frame.index.as_unit('ns')
    if (not frame.index.is_unique or not frame.index.is_monotonic_increasing
            or not frame.index.to_series().diff().dropna().eq(BAR).all()
            or frame.index[-1] + BAR > utc(as_of)):
        raise ValueError('Missing, duplicate, unordered or unfinished spot candle')
    return frame


def write_feed_snapshot(root, raw, protocol, observed_at):
    observed = utc(observed_at)
    if observed > utc_now() or set(raw) != set(SYMBOLS):
        raise ValueError('Future snapshot time or missing spot market')
    frames = {s: parse_closed_klines(raw[s], observed) for s in SYMBOLS}
    first = frames[SYMBOLS[0]].index
    expected = pd.date_range(utc(protocol['warmup_start']), observed.floor('4h'), freq='4h', inclusive='left')
    if not first.equals(expected) or any(not f.index.equals(first) for f in frames.values()):
        raise ValueError('Snapshot must include complete aligned warmup through the latest closed bar')
    payloads = {s: canonical(raw[s]) for s in SYMBOLS}
    manifest = {'source': SOURCE, 'timeframe': '4h', 'observed_at': observed.isoformat(),
        'protocol_sha256': protocol['protocol_sha256'],
        'datasets': {s: {'file': f'{s}_4h.json', 'rows': len(raw[s]),
            'sha256': hashlib.sha256(payloads[s]).hexdigest()} for s in SYMBOLS}}
    destination = Path(root) / ('snapshot-' + observed.strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(parents=True, exist_ok=False)
    for s in SYMBOLS:
        (destination / manifest['datasets'][s]['file']).write_bytes(payloads[s])
    # A snapshot is usable only once its final manifest has been written.
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return destination


def load_feed_snapshot(folder, protocol, as_of):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    observed = utc(manifest['observed_at'])
    now, as_of = utc_now(), utc(as_of)
    if (as_of > now or observed > as_of or as_of < utc(protocol['frozen_at'])
            or manifest.get('source') != SOURCE or manifest.get('timeframe') != '4h'
            or manifest.get('protocol_sha256') != protocol['protocol_sha256']
            or set(manifest.get('datasets', {})) != set(SYMBOLS)
            or as_of - observed.floor('4h') > pd.Timedelta(hours=SETTINGS['maximum_feed_age_hours'])):
        raise ValueError('Wrong source/protocol, future time or stale forward feed')
    frames, raw = {}, {}
    expected = pd.date_range(utc(protocol['warmup_start']), observed.floor('4h'), freq='4h', inclusive='left')
    for s in SYMBOLS:
        entry = manifest['datasets'][s]
        if entry['file'] != f'{s}_4h.json':
            raise ValueError('Unexpected snapshot filename')
        content = (folder / entry['file']).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError('Raw source snapshot checksum mismatch')
        raw[s] = json.loads(content)
        frames[s] = parse_closed_klines(raw[s], observed)
        if len(raw[s]) != entry['rows'] or not frames[s].index.equals(expected):
            raise ValueError('Complete identical warmup/forward history required')
    return frames, raw, manifest


@contextmanager
def evaluation_lock(out):
    lock = out / 'evaluation.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError('Another evaluation is active, or a prior lock needs inspection') from exc
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink()


def verify_previous(out, state, raw, protocol):
    if state['protocol_sha256'] != protocol['protocol_sha256']:
        raise ValueError('Cannot change a funded account protocol')
    if not re.fullmatch(r'run-[0-9a-f]{32}', state['last_run']):
        raise ValueError('Invalid receipt pointer')
    run = out / 'runs' / state['last_run']
    content = (run / 'receipt.json').read_bytes()
    if hashlib.sha256(content).hexdigest() != state['receipt_sha256']:
        raise ValueError('Previous receipt checksum mismatch')
    receipt = json.loads(content)
    for name, sha in receipt['ledger_sha256'].items():
        if Path(name).name != name or hashlib.sha256((run / name).read_bytes()).hexdigest() != sha:
            raise ValueError('Previous account ledger changed')
    for s in SYMBOLS:
        count = state['source_rows'][s]
        if len(raw[s]) < count or digest(raw[s][:count]) != state['prefix_sha256'][s]:
            raise ValueError('Previously accepted data prefix changed or shrank')
    return receipt


def evaluate_forward(protocol_path, data, out, as_of=None):
    protocol = read_protocol(protocol_path)
    as_of = utc(as_of) if as_of is not None else utc_now()
    markets, raw, manifest = load_feed_snapshot(data, protocol, as_of)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(out):
        state_path = out / 'state.json'
        previous = json.loads(state_path.read_text()) if state_path.exists() else None
        if previous is not None:
            receipt = verify_previous(out, previous, raw, protocol)
            if as_of < utc(previous['as_of']):
                raise ValueError('Cannot rewind the forward account clock')
            if all(len(raw[s]) == previous['source_rows'][s] for s in SYMBOLS):
                return receipt['report']
        elif (out / 'runs').exists() and any((out / 'runs').iterdir()):
            raise ValueError('Existing receipts without state: inspect before resetting the account')
        start = utc(protocol['forward_start'])
        end = next(iter(markets.values())).index[-1] + BAR
        forward_bars = max(0, int((end - start) / BAR))
        run_name = 'run-' + uuid4().hex
        run = out / 'runs' / run_name
        run.mkdir(parents=True, exist_ok=False)
        summaries = {}
        if forward_bars:
            target = momentum_refinement_targets(markets)[0]['capped_allocation']
            risk = PortfolioRisk(volatility_target=.15)
            cov = completed_risk_covariances(markets, risk)
            for scenario in SCENARIOS:
                summaries[scenario] = {}
                for policy in POLICIES:
                    guarded = policy == 'guarded_edge'
                    stress = scenario in ('cost_stress', 'joint_stress')
                    eq, fills, summary = run_intraday_spot(markets, target, start=start, end=end,
                        initial_usdt=SETTINGS['initial_usdt'], daily_control=True,
                        fee_rate=.002 if stress else .001, slippage_bps=10. if stress else 5., spread_bps=2.,
                        signal_delay_bars=int(scenario in ('delayed', 'joint_stress')),
                        rebalance_destination='band_edge' if guarded else 'target',
                        risk=risk if guarded else None, completed_covariances=cov if guarded else None)
                    summary['accounting'] = reconcile_spot_path(markets, eq, fills, SETTINGS['initial_usdt'])
                    eq.to_csv(run / f'{scenario}_{policy}_equity.csv', index=False)
                    fills.to_csv(run / f'{scenario}_{policy}_trades.csv', index=False)
                    summaries[scenario][policy] = summary
        enough = forward_bars >= SETTINGS['minimum_forward_days'] * 6
        drawdown_ok = bool(summaries) and all(s['guarded_edge']['max_intraday_drawdown_bound'] >= -.20 for s in summaries.values())
        gates = {'enough_closed_forward_days': enough, 'candidate_drawdowns_within_budget': drawdown_ok,
            'candidate_positive_after_all_cost_stresses': bool(summaries) and all(s['guarded_edge']['net_pnl'] > 0 for s in summaries.values()),
            'beats_control_nominal_and_joint': bool(summaries) and all(summaries[s]['guarded_edge']['net_pnl'] > summaries[s]['spot_control']['net_pnl'] for s in ('nominal', 'joint_stress')),
            'accounting': bool(summaries) and all(row['accounting']['reconciled'] for scenario in summaries.values() for row in scenario.values())}
        passed = bool(all(gates.values()))
        phase = ('waiting_for_start' if not forward_bars else 'rejected_drawdown' if not drawdown_ok
                 else 'collecting' if not enough else 'preliminary_pass' if passed else 'preliminary_fail')
        report = {'protocol_sha256': protocol['protocol_sha256'], 'as_of': as_of.isoformat(),
            'snapshot_observed_at': manifest['observed_at'], 'forward_start': start.isoformat(),
            'forward_closed_bars': forward_bars, 'forward_full_days': forward_bars // 6,
            'phase': phase, 'gates': gates, 'preliminary_screen_passed': passed,
            'live_eligible': False, 'results': summaries,
            'publication_timestamp_requires_external_anchor': True,
            'limitations': ['Fixed model replay, not observed executable paper orders.',
                'Local checksums are not an unforgeable publication timestamp.',
                'Modeled fees, impact and terminal marks; no custody/tax/USDT-risk model.',
                'Historical risk screens and requested volatility do not cap future losses.']}
        ledger_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.glob('*.csv'))}
        receipt = {'report': report, 'source_manifest': manifest, 'ledger_sha256': ledger_hashes,
            'previous_receipt_sha256': previous['receipt_sha256'] if previous else None}
        content = (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode()
        (run / 'receipt.json').write_bytes(content)
        state = {'protocol_sha256': protocol['protocol_sha256'], 'last_run': run_name,
            'receipt_sha256': hashlib.sha256(content).hexdigest(), 'as_of': as_of.isoformat(),
            'source_rows': {s: len(raw[s]) for s in SYMBOLS}, 'prefix_sha256': {s: digest(raw[s]) for s in SYMBOLS}}
        temporary = out / ('state-' + uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(temporary, state_path)
        return report
