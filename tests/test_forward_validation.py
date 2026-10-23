"""Prospective clocks, continuous funding, immutable evidence and fail-closed inputs."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spot_bot.forward_validation as fv
import scripts.download_forward_spot as downloader


@pytest.fixture
def study(tmp_path, monkeypatch):
    start = pd.Timestamp('2024-05-01', tz='UTC')
    clock = [start - pd.Timedelta('12h')]
    monkeypatch.setattr(fv, 'utc_now', lambda: clock[0])
    monkeypatch.setattr(downloader, 'utc_now', lambda: clock[0])
    protocol_path = tmp_path / 'protocol.json'
    protocol = fv.freeze_protocol(protocol_path, start)
    dates = pd.date_range(fv.utc(protocol['warmup_start']), periods=6 * 310, freq='4h')
    raw = {}
    for j, symbol in enumerate(fv.SYMBOLS):
        prices = (100 + j * 10) * np.exp(np.arange(len(dates)) * .0002)
        rows = []
        for i, ts in enumerate(dates):
            o, c = prices[max(0, i - 1)], prices[i]
            ms = int(ts.timestamp() * 1000)
            rows.append([ms, str(o), str(max(o, c) * 1.001), str(min(o, c) * .999), str(c),
                '100', ms + 14399999, str(100 * c), 100, '50', str(50 * c), '0'])
        raw[symbol] = rows

    def snapshot(days, mutate=None):
        clock[0] = start + pd.Timedelta(days=days)
        count = int((clock[0] - dates[0]) / fv.BAR)
        selected = {s: copy.deepcopy(rows[:count]) for s, rows in raw.items()}
        if mutate: mutate(selected)
        folder = fv.write_feed_snapshot(tmp_path / 'feed', selected, protocol, clock[0])
        return folder

    return {'start': start, 'clock': clock, 'protocol_path': protocol_path, 'protocol': protocol,
        'raw': raw, 'snapshot': snapshot, 'out': tmp_path / 'account', 'tmp': tmp_path}


def test_freeze_refuses_known_boundary_and_overwriting(study):
    with pytest.raises(ValueError, match='future UTC midnight'):
        fv.freeze_protocol(study['tmp'] / 'past.json', study['start'] - pd.Timedelta('1D'))
    with pytest.raises(FileExistsError):
        fv.freeze_protocol(study['protocol_path'], study['start'])
    assert fv.read_protocol(study['protocol_path'])['settings']['minimum_forward_days'] == 180


def test_no_forward_data_is_reported_as_waiting_not_as_a_win(study):
    snap = study['snapshot'](-1/6)
    report = fv.evaluate_forward(study['protocol_path'], snap, study['out'])
    assert report['phase'] == 'waiting_for_start'
    assert report['forward_closed_bars'] == 0 and report['results'] == {}
    assert not report['preliminary_screen_passed'] and not report['live_eligible']


def test_account_extends_without_reset_and_preserves_every_earlier_fill(study):
    a = study['snapshot'](12)
    first = fv.evaluate_forward(study['protocol_path'], a, study['out'])
    old_state = json.loads((study['out'] / 'state.json').read_text())
    old_run = study['out'] / 'runs' / old_state['last_run']
    b = study['snapshot'](20)
    later = fv.evaluate_forward(study['protocol_path'], b, study['out'])
    state = json.loads((study['out'] / 'state.json').read_text())
    run = study['out'] / 'runs' / state['last_run']
    receipt = json.loads((run / 'receipt.json').read_text())
    assert receipt['previous_receipt_sha256'] == old_state['receipt_sha256']
    assert first['forward_closed_bars'] == 72 and later['forward_closed_bars'] == 120
    assert later['phase'] == 'collecting' and not later['preliminary_screen_passed']
    for old_path in old_run.glob('*.csv'):
        x, y = pd.read_csv(old_path), pd.read_csv(run / old_path.name)
        pd.testing.assert_frame_equal(x, y.iloc[:len(x)], check_dtype=False)
    for results in later['results'].values():
        assert all(row['accounting']['reconciled'] and row['minimum_cash'] >= 0
                   and row['minimum_inventory'] >= -1e-10 for row in results.values())
    assert fv.evaluate_forward(study['protocol_path'], b, study['out']) == later
    assert json.loads((study['out'] / 'state.json').read_text()) == state


def test_revised_warmup_history_is_not_accepted_as_new_forward_evidence(study):
    a = study['snapshot'](8)
    fv.evaluate_forward(study['protocol_path'], a, study['out'])
    original = (study['out'] / 'state.json').read_bytes()

    def revise(raw):
        raw['BTCUSDT'][10][5] = '101'  # valid OHLCV; old history was revised.
    b = study['snapshot'](9, revise)
    with pytest.raises(ValueError, match='prefix changed'):
        fv.evaluate_forward(study['protocol_path'], b, study['out'])
    assert (study['out'] / 'state.json').read_bytes() == original


def test_old_receipt_or_csv_corruption_blocks_progress(study):
    a = study['snapshot'](8)
    fv.evaluate_forward(study['protocol_path'], a, study['out'])
    state = json.loads((study['out'] / 'state.json').read_text())
    run = study['out'] / 'runs' / state['last_run']
    ledger = next(run.glob('*.csv'))
    ledger.write_bytes(ledger.read_bytes() + b'\n')
    b = study['snapshot'](9)
    with pytest.raises(ValueError, match='ledger changed'):
        fv.evaluate_forward(study['protocol_path'], b, study['out'])


def test_a_clock_jump_does_not_create_180_days_of_observations(study):
    a = study['snapshot'](8)
    study['clock'][0] += pd.Timedelta('190D')
    with pytest.raises(ValueError, match='stale forward feed'):
        fv.evaluate_forward(study['protocol_path'], a, study['out'])


def test_real_180_closed_days_only_unlock_the_preliminary_screen(study):
    a = study['snapshot'](180)
    report = fv.evaluate_forward(study['protocol_path'], a, study['out'])
    assert report['forward_closed_bars'] == 1080
    assert report['gates']['enough_closed_forward_days']
    assert not report['live_eligible']
    assert all(row['accounting']['reconciled'] for results in report['results'].values() for row in results.values())


def test_a_gap_loss_rejects_early_and_does_not_reset_or_invent_a_stop_fill(study):
    event = int((study['start'] + pd.Timedelta('8D') - fv.utc(study['protocol']['warmup_start'])) / fv.BAR)

    def crash(raw):
        for rows in raw.values():
            for row in rows[event:]:
                for col in (1, 2, 3, 4, 7, 10): row[col] = str(float(row[col]) * .01)
    a = study['snapshot'](12, crash)
    report = fv.evaluate_forward(study['protocol_path'], a, study['out'])
    assert report['phase'] == 'rejected_drawdown'
    assert not report['gates']['candidate_drawdowns_within_budget']
    assert not report['preliminary_screen_passed']
    assert report['results']['nominal']['guarded_edge']['max_intraday_drawdown_bound'] < -.20
    assert report['results']['nominal']['guarded_edge']['minimum_cash'] >= 0


def test_a_failed_state_commit_leaves_the_last_funded_account_unchanged(study, monkeypatch):
    a = study['snapshot'](8)
    fv.evaluate_forward(study['protocol_path'], a, study['out'])
    original = (study['out'] / 'state.json').read_bytes()
    b = study['snapshot'](9)
    replace = fv.os.replace
    monkeypatch.setattr(fv.os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('disk interrupted')))
    with pytest.raises(OSError, match='disk interrupted'):
        fv.evaluate_forward(study['protocol_path'], b, study['out'])
    assert (study['out'] / 'state.json').read_bytes() == original
    assert not (study['out'] / 'evaluation.lock').exists()
    monkeypatch.setattr(fv.os, 'replace', replace)
    assert fv.evaluate_forward(study['protocol_path'], b, study['out'])['forward_full_days'] == 9


def test_concurrent_evaluation_cannot_overwrite_an_account(study):
    a = study['snapshot'](8)
    study['out'].mkdir()
    (study['out'] / 'evaluation.lock').touch()
    with pytest.raises(ValueError, match='evaluation is active'):
        fv.evaluate_forward(study['protocol_path'], a, study['out'])
    assert not (study['out'] / 'state.json').exists()


@pytest.mark.parametrize('kind', ['gap', 'duplicate', 'close_clock', 'negative', 'future', 'nan'])
def test_bad_candles_never_enter_a_forward_snapshot(study, kind):
    rows = copy.deepcopy(study['raw']['BTCUSDT'][:30])
    as_of = fv.utc(study['protocol']['warmup_start']) + pd.Timedelta('5D')
    if kind == 'gap': del rows[10]
    elif kind == 'duplicate': rows[10] = rows[9]
    elif kind == 'close_clock': rows[10][6] -= 1
    elif kind == 'negative': rows[10][5] = '-1'
    elif kind == 'nan': rows[10][4] = 'nan'
    elif kind == 'future': as_of -= fv.BAR
    with pytest.raises(ValueError): fv.parse_closed_klines(rows, as_of)


@pytest.mark.parametrize('change', ['configuration', 'source', 'runtime'])
def test_changed_protocol_or_implementation_requires_a_separate_freeze(study, monkeypatch, change):
    if change == 'configuration':
        protocol = json.loads(study['protocol_path'].read_text())
        protocol['settings']['initial_usdt'] *= 2
        study['protocol_path'].write_text(json.dumps(protocol))
    elif change == 'source': monkeypatch.setattr(fv, 'source_fingerprints', lambda: {})
    else: monkeypatch.setattr(fv, 'runtime_versions', lambda: {})
    with pytest.raises(ValueError): fv.read_protocol(study['protocol_path'])


def test_public_spot_pagination_keeps_a_complete_utc_snapshot(study):
    study['clock'][0] = study['start'] + pd.Timedelta('70D')
    calls = []

    def page(symbol, begin, end):
        calls.append((symbol, begin, end))
        return [row for row in study['raw'][symbol] if begin <= row[0] <= end][:1000]
    snapshot = downloader.download(study['protocol_path'], study['tmp'] / 'download', fetch=page)
    markets, _, manifest = fv.load_feed_snapshot(snapshot, study['protocol'], study['clock'][0])
    assert len(calls) == 6 and manifest['source'] == fv.SOURCE
    assert all(len(frame) == 1140 for frame in markets.values())


def test_repeated_public_pages_are_rejected_instead_of_looping(study):
    study['clock'][0] = study['start'] + pd.Timedelta('70D')
    with pytest.raises(ValueError, match='gap, duplicate or wrong time range'):
        downloader.download(study['protocol_path'], study['tmp'] / 'bad-download',
            fetch=lambda symbol, begin, end: study['raw'][symbol][:1000])
