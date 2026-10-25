"""Owned funding, true depth arithmetic and immutable observation evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import json

import pytest

from theta_bot_averaging.depth_paper import (
    SYMBOLS, Book, Rules, common_step, floor_step, json_values, number,
    round_trip_probe, simulate_fill, walk,
)
from scripts import research_paper_execution as research


def info(symbol='BTCUSDT'):
    quote='USDC' if symbol.endswith('USDC') else 'USDT'
    return {'symbols':[{'symbol':symbol,'baseAsset':symbol[:-len(quote)],'quoteAsset':quote,
        'status':'TRADING','isSpotTradingAllowed':True,'orderTypes':['MARKET'],
        'quoteCommissionPrecision':8,'filters':[
            {'filterType':'LOT_SIZE','minQty':'0.001','maxQty':'10000','stepSize':'0.001'},
            {'filterType':'MARKET_LOT_SIZE','minQty':'0','maxQty':'1000','stepSize':'0'},
            {'filterType':'PRICE_FILTER','tickSize':'0.01'},
            {'filterType':'NOTIONAL','minNotional':'5','maxNotional':'1000000',
             'applyMinToMarket':True,'applyMaxToMarket':False,'avgPriceMins':5}]}]}


def depth():
    return {'lastUpdateId':19,'bids':[['99','0.5'],['98','100']],
            'asks':[['101','0.5'],['102','100']]}


def setup():
    rules=Rules.from_info(info(),'BTCUSDT')
    return rules,Book.from_depth(depth(),rules)


def test_depth_walk_uses_multiple_levels_and_refuses_invented_liquidity():
    rules,book=setup()
    assert walk(book,'buy',D('1'))==(D('101.5'),2)
    assert walk(book,'sell',D('1'))==(D('98.5'),2)
    with pytest.raises(ValueError,match='insufficient_visible_depth'): walk(book,'buy',D('101'))
    assert book.asks[0]==(D('101'),D('.5'))


def test_floor_and_combined_quantity_steps_are_exact():
    assert common_step([D('.003'),D('.002'),D('0')])==D('.006')
    assert floor_step(D('.0119'),D('.006'))==D('.006')
    rules,_=setup()
    assert rules.step==D('.001')  # zero market step does not disable LOT_SIZE


@pytest.mark.parametrize('bad',[True,1.2,'NaN','Infinity','-1',None])
def test_nonexact_or_invalid_numbers_fail(bad):
    with pytest.raises(ValueError): number(bad)


@pytest.mark.parametrize('field,value',[('status','HALT'),('isSpotTradingAllowed',False),
    ('baseAsset','ETH'),('orderTypes',['LIMIT'])])
def test_unverified_spot_markets_reject(field,value):
    source=info()
    source['symbols'][0][field]=value
    with pytest.raises(ValueError): Rules.from_info(source,'BTCUSDT')


@pytest.mark.parametrize('kind',['crossed','duplicate','unordered','zero','wrong_tick','bad_id','empty'])
def test_invalid_depth_rejects(kind):
    rules,_=setup()
    raw=depth()
    if kind=='crossed': raw['bids'][0][0]='101'
    elif kind=='duplicate': raw['asks'][1][0]='101'
    elif kind=='unordered': raw['asks'].reverse()
    elif kind=='zero': raw['bids'][0][1]='0'
    elif kind=='wrong_tick': raw['asks'][0][0]='101.001'
    elif kind=='bad_id': raw['lastUpdateId']=True
    else: raw['asks']=[]
    with pytest.raises(ValueError): Book.from_depth(raw,rules)


def test_conditional_round_trip_reconciles_fees_depth_and_owned_balances():
    rules,book=setup()
    case=json_values(round_trip_probe(book,rules,'1000','.001'))
    assert case['buy']['levels_used']==case['sell']['levels_used']==2
    assert D(case['quote_spent'])<=1000
    assert D(case['cash_after_roundtrip'])==D('10000')-D(case['round_trip_cost_quote'])
    assert case['sell']['inventory_after']=='0.000'
    assert case['exchange_filters_fully_verified'] is False
    assert case['observed_exchange_fill'] is False
    assert research.audit_case(case,depth(),rules,research.SCENARIOS['nominal'])['reconciled']
    bad=deepcopy(case)
    bad['buy']['cash_after']=str(D(case['buy']['cash_after'])+D('.0001'))
    with pytest.raises(ValueError,match='does not reconcile'):
        research.audit_case(bad,depth(),rules,research.SCENARIOS['nominal'])


def test_stress_costs_rise_without_changing_the_book():
    rules,book=setup()
    a=round_trip_probe(book,rules,'1000','.001')
    b=round_trip_probe(book,rules,'1000','.002','5')
    assert b['round_trip_cost_bps']>a['round_trip_cost_bps']
    assert b['base_quantity']<=a['base_quantity']
    assert book==Book.from_depth(depth(),rules)


def test_fills_refuse_borrowing_shorting_and_unsafe_quantity_rounding():
    rules,book=setup()
    with pytest.raises(ValueError,match='insufficient_owned_cash'):
        simulate_fill(book,rules,'buy','1','100','0','.001')
    with pytest.raises(ValueError,match='insufficient_owned_inventory'):
        simulate_fill(book,rules,'sell','1','10000','.999','.001')
    with pytest.raises(ValueError,match='quantity_filter'):
        simulate_fill(book,rules,'buy','1.0001','10000','0','.001')
    with pytest.raises(ValueError,match='25%'):
        round_trip_probe(book,rules,'2501','.001')
    tiny=depth()
    tiny['asks']=[['101','.1']]
    with pytest.raises(ValueError,match='insufficient_visible_depth'):
        round_trip_probe(Book.from_depth(tiny,rules),rules,'1000','.001')


def fixture(tmp_path,monkeypatch):
    clock=datetime(2026,1,1,tzinfo=timezone.utc)
    monkeypatch.setattr(research,'now',lambda:clock)
    proof=research.freeze(tmp_path/'protocol.json')
    def record(endpoint,params):
        is_info=endpoint.endswith('exchangeInfo')
        t=clock+timedelta(seconds=1 if is_info else 2)
        return {'source_url':research.BASE+endpoint,'method':'GET','transport':'public_https',
            'params':params,'request_started_at':t.isoformat(),
            'received_at':(t+timedelta(milliseconds=100)).isoformat(),'elapsed_ms':100,
            'payload':info(params['symbol']) if is_info else depth()}
    monkeypatch.setattr(research,'now',lambda:clock+timedelta(seconds=3))
    return proof,research.collect(proof,record)


def test_capture_replays_with_conditional_labels_and_an_append_only_receipt(tmp_path,monkeypatch):
    proof,capture=fixture(tmp_path,monkeypatch)
    a,coverage=research.save_and_analyze(proof,capture,tmp_path)
    assert a['observed_executable_fills']==0 and a['strategy_pnl_available'] is False
    assert all(r['source_book_age_ms'] is None for r in a['results'].values())
    assert all(c['distinct_successful_books']==1 and not c['coverage_gate_passed'] for c in coverage.values())
    assert all(c['status']=='conditional_simulation' for r in a['results'].values()
        for scenario in r['cases'].values() for c in scenario.values())
    assert research.save_and_analyze(proof,capture,tmp_path)==(a,coverage)
    path=next((tmp_path/'reports').glob('*.json'))
    old=json.loads(path.read_text())
    old['strategy_pnl_available']=True
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError,match='immutable'): research.save_and_analyze(proof,capture,tmp_path)


@pytest.mark.parametrize('kind',['slow','future','clock_mismatch','source','prefrozen','metadata_stale','source_error'])
def test_bad_observation_times_and_sources_are_preserved_and_rejected(tmp_path,monkeypatch,kind):
    proof,capture=fixture(tmp_path,monkeypatch)
    row=capture['markets']['BTCUSDT']['depth']
    if kind=='slow': row['elapsed_ms']=5001
    elif kind=='future': row['received_at']='2026-01-02T00:00:00+00:00'
    elif kind=='clock_mismatch': row['elapsed_ms']=1000
    elif kind=='source': row['source_url']='https://example.com/api/v3/depth'
    elif kind=='prefrozen':
        row['request_started_at']='2025-12-31T23:59:59+00:00'
        row['received_at']='2025-12-31T23:59:59.1+00:00'
    elif kind=='metadata_stale':
        meta=capture['markets']['BTCUSDT']['info']
        meta['request_started_at']='2025-12-31T22:00:00+00:00'
        meta['received_at']='2025-12-31T22:00:00.1+00:00'
    else: row['error']='HTTP 429'
    result,coverage=research.save_and_analyze(proof,capture,tmp_path)
    assert result['results']['BTCUSDT']['status']=='rejected'
    assert coverage['BTCUSDT']['distinct_successful_books']==0
    assert json.loads(next((tmp_path/'captures').glob('*.json')).read_text())==capture


def test_duplicate_books_cannot_inflate_coverage(tmp_path,monkeypatch):
    proof,capture=fixture(tmp_path,monkeypatch)
    research.save_and_analyze(proof,capture,tmp_path)
    capture['observed_at']='2026-01-01T00:00:03.001+00:00'
    monkeypatch.setattr(research,'now',lambda:datetime(2026,1,1,0,0,4,tzinfo=timezone.utc))
    _,coverage=research.save_and_analyze(proof,capture,tmp_path)
    assert all(r['distinct_successful_books']==1 for r in coverage.values())


def test_frozen_protocol_rejects_rehashed_settings_and_runtime_changes(tmp_path,monkeypatch):
    proof,_=fixture(tmp_path,monkeypatch)
    assert research.read_protocol(tmp_path/'protocol.json')==proof
    with pytest.raises(FileExistsError): research.freeze(tmp_path/'protocol.json')
    proof['settings']['maximum_request_ms']=100000
    proof['protocol_sha256']=research.digest({k:v for k,v in proof.items() if k!='protocol_sha256'})
    (tmp_path/'protocol.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='Frozen'): research.read_protocol(tmp_path/'protocol.json')


def test_public_collector_has_only_the_two_fixed_get_paths(monkeypatch):
    calls=[]
    class Response:
        status_code=302
    def fake_get(url,**kwargs):
        calls.append((url,kwargs))
        return Response()
    monkeypatch.setattr(research.requests,'get',fake_get)
    row=research.public_get('/api/v3/depth',{'symbol':'BTCUSDT','limit':100})
    assert '302' in row['error']
    assert calls[0][0].startswith(research.BASE)
    assert calls[0][1]['allow_redirects'] is False
    with pytest.raises(ValueError): research.public_get('/api/v3/order',{})
    assert len(calls)==1


def test_full_evidence_export_contains_raw_capture_and_audited_receipts(tmp_path,monkeypatch):
    import hashlib
    import zipfile
    proof,capture=fixture(tmp_path,monkeypatch)
    report,coverage=research.save_and_analyze(proof,capture,tmp_path)
    destination=tmp_path/'published'
    research.export(report,coverage,tmp_path,destination)
    archive=next(destination.glob('*.zip'))
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==next(destination.glob('*.sha256')).read_text().split()[0]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert json.loads(z.read('frozen_protocol.json'))==proof
        assert json.loads(z.read('captures/'+research.digest(capture)+'.json'))==capture
        assert json.loads(z.read('reports/'+research.digest(capture)+'.json'))==report
