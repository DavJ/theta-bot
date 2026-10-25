"""Prospective boundaries, publication identity and continuous funded accounts."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from theta_bot_averaging import entry_forward as forward


@pytest.fixture
def cohort(tmp_path,monkeypatch):
    clock={'now':pd.Timestamp('2022-03-20T12:00:00Z')}
    monkeypatch.setattr(forward.legacy,'utc_now',lambda:clock['now'])
    protocol_path=tmp_path/'protocol.json'
    protocol=forward.freeze_protocol(protocol_path)
    clock['now']+=pd.Timedelta('1min')
    run={'id':123,'created_at':clock['now'].isoformat(),'head_sha':'a'*40,
         'repository':{'full_name':forward.REPOSITORY},'path':forward.WORKFLOW,
         'head_commit':{'id':'a'*40,'tree_id':'b'*40},
         'html_url':f'https://github.com/{forward.REPOSITORY}/actions/runs/123'}
    blobs=forward.expected_git_blobs(protocol) | {forward.PROTOCOL_GIT_PATH:forward.git_blob(protocol_path.read_bytes())}
    tree={'sha':'b'*40,'truncated':False,
          'tree':[{'path':p,'sha':h,'type':'blob','mode':'100644'} for p,h in blobs.items()]}
    anchor_path=tmp_path/'publication.json'
    anchor=forward.write_publication(protocol_path,run,tree,anchor_path)
    return {'root':tmp_path,'clock':clock,'protocol_path':protocol_path,'protocol':protocol,
            'run':run,'tree':tree,'anchor_path':anchor_path,'anchor':anchor}


def raw_bars(observed):
    index=pd.date_range(forward.HISTORY_START,observed.floor('4h'),freq='4h',inclusive='left')
    raw={}
    for j,symbol in enumerate(forward.SYMBOLS):
        rows=[]
        for i,timestamp in enumerate(index):
            price=100+(j+1)*i*.1
            begin=int(timestamp.timestamp()*1000)
            rows.append([begin,str(price),str(price*1.002),str(price*.998),str(price*1.0002),
                         '100',begin+14399999,'10000',10,'50','5000','0'])
        raw[symbol]=rows
    return raw


def snapshot(cohort,observed,raw=None):
    observed=pd.Timestamp(observed)
    cohort['clock']['now']=observed
    return forward.import_snapshot(cohort['protocol_path'],raw if raw is not None else raw_bars(observed),
        cohort['root']/'feed',observed,'binance_mcp_public_spot')


def evaluate(cohort,data,out=None):
    return forward.evaluate_forward(cohort['protocol_path'],cohort['anchor_path'],data,
        out if out is not None else cohort['root']/'accounts')


def test_future_freeze_is_exclusive_and_separate_from_old_cohort(cohort):
    assert forward.read_protocol(cohort['protocol_path'])==cohort['protocol']
    assert cohort['protocol']['forward_start']=='2022-03-21T00:00:00+00:00'
    assert cohort['protocol']['warmup_start']==forward.HISTORY_START
    assert cohort['protocol']['minimum_assessment_at']=='2022-09-17T00:00:00+00:00'
    with pytest.raises(FileExistsError): forward.freeze_protocol(cohort['protocol_path'])
    with pytest.raises(ValueError): forward.legacy.read_protocol(cohort['protocol_path'])


@pytest.mark.parametrize('start',['2022-03-20T00:00:00Z','2022-03-21T04:00:00Z','2022-03-21'])
def test_retroactive_nonmidnight_or_naive_start_rejects(cohort,start):
    with pytest.raises(ValueError): forward.freeze_protocol(cohort['root']/'bad.json',start)


@pytest.mark.parametrize('field',['settings','runtime','source_sha256'])
def test_rehashed_protocol_change_is_rejected(cohort,field):
    proof=copy.deepcopy(cohort['protocol'])
    proof[field]={}
    proof['protocol_sha256']=forward.legacy.digest({k:v for k,v in proof.items() if k!='protocol_sha256'})
    cohort['protocol_path'].write_bytes(forward.json_bytes(proof))
    with pytest.raises(ValueError): forward.read_protocol(cohort['protocol_path'])


@pytest.mark.parametrize('problem',['late','repository','workflow','source_blob','protocol_blob','truncated'])
def test_publication_run_must_bind_prestart_source_and_exact_protocol(cohort,problem):
    run,tree=copy.deepcopy(cohort['run']),copy.deepcopy(cohort['tree'])
    if problem=='late': run['created_at']=cohort['protocol']['forward_start']
    elif problem=='repository': run['repository']['full_name']='someone/else'
    elif problem=='workflow': run['path']='other.yml'
    elif problem=='source_blob': tree['tree'][0]['sha']='0'*40
    elif problem=='protocol_blob': tree['tree']=[r for r in tree['tree'] if r['path']!=forward.PROTOCOL_GIT_PATH]
    else: tree['truncated']=True
    with pytest.raises(ValueError): forward.write_publication(cohort['protocol_path'],run,tree,cohort['root']/'bad-anchor.json')


def test_rehashed_anchor_date_cannot_move_publication_after_start(cohort):
    anchor=copy.deepcopy(cohort['anchor'])
    anchor['run_created_at']=cohort['protocol']['forward_start']
    anchor['anchor_sha256']=forward.legacy.digest({k:v for k,v in anchor.items() if k!='anchor_sha256'})
    cohort['anchor_path'].write_bytes(forward.json_bytes(anchor))
    with pytest.raises(ValueError): forward.read_publication(cohort['anchor_path'],cohort['protocol'])


def test_prestart_snapshot_has_no_scored_pnl_or_vacuously_passing_economic_gates(cohort):
    data=snapshot(cohort,'2022-03-20T16:00:00Z')
    result=evaluate(cohort,data)
    assert result['phase']=='waiting_for_start' and result['forward_closed_bars']==0
    assert result['results']=={} and result['independent_csv_audits']=={}
    assert result['gates']['accounting'] is None
    assert result['gates']['candidate_positive_all_scenarios'] is None
    assert not result['gates']['enough_closed_forward_days']
    assert result['preliminary_screen_passed'] is False and result['live_eligible'] is False
    assert evaluate(cohort,data)==result


def test_first_closed_bar_boundary_is_not_elapsed_time(cohort):
    data=snapshot(cohort,'2022-03-21T01:00:00Z')
    result=evaluate(cohort,data)
    assert result['phase']=='waiting_for_closed_bar' and result['forward_closed_bars']==0


def test_full_account_replay_extends_exact_ledgers_over_a_partial_day_boundary(cohort):
    first=snapshot(cohort,'2022-03-21T20:00:00Z')
    result=evaluate(cohort,first)
    assert result['forward_closed_bars']==5 and result['phase']=='collecting'
    assert result['gates']['accounting'] and result['gates']['independent_csv_accounting']
    assert sum(map(len,result['results'].values()))==16
    out=cohort['root']/'accounts'
    state=json.loads((out/'state.json').read_text())
    old_run=out/'runs'/state['last_run']
    old={p.name:p.read_bytes() for p in old_run.glob('*.csv')}
    assert len(old)==32
    equity=pd.read_csv(old_run/'nominal_entry_20_equity.csv')
    assert pd.to_datetime(equity.timestamp,utc=True).iloc[0]==pd.Timestamp(cohort['protocol']['forward_start'])
    assert equity.usdt.min()>=0 and equity.filter(like='base_').min().min()>=0
    for observed,count in [('2022-03-22T00:00:00Z',6),('2022-03-22T04:00:00Z',7)]:
        data=snapshot(cohort,observed)
        result=evaluate(cohort,data)
        assert result['forward_closed_bars']==count
        new_state=json.loads((out/'state.json').read_text())
        new_run=out/'runs'/new_state['last_run']
        assert all((new_run/name).read_bytes().startswith(payload) for name,payload in old.items())
    assert new_state['sequence']==3 and new_state['anchor_sha256']==cohort['anchor']['anchor_sha256']


@pytest.mark.parametrize('problem',['history','ledger','receipt','state_reset','orphan','rollback','anchor'])
def test_accepted_data_receipts_and_funded_identity_cannot_be_rewritten(cohort,problem):
    first=snapshot(cohort,'2022-03-20T16:00:00Z')
    evaluate(cohort,first)
    out=cohort['root']/'accounts'
    old_state=(out/'state.json').read_bytes()
    if problem=='ledger':
        data=snapshot(cohort,'2022-03-21T20:00:00Z')
        evaluate(cohort,data)
    data=snapshot(cohort,'2022-03-22T04:00:00Z')
    if problem=='history':
        raw=raw_bars(cohort['clock']['now'])
        raw['BTCUSDT'][0][5]='101'
        data=snapshot(cohort,'2022-03-22T04:00:01Z',raw)
    elif problem=='ledger':
        state=json.loads((out/'state.json').read_text())
        (out/'runs'/state['last_run']/'nominal_entry_20_equity.csv').write_text('changed')
    elif problem=='receipt':
        state=json.loads((out/'state.json').read_text())
        with (out/'runs'/state['last_run']/'receipt.json').open('a') as f: f.write(' ')
    elif problem=='state_reset': (out/'state.json').unlink()
    elif problem=='orphan': (out/'runs'/('run-'+'f'*32)).mkdir()
    elif problem=='rollback':
        evaluate(cohort,data)
        (out/'state.json').write_bytes(old_state)
    else:
        anchor=copy.deepcopy(cohort['anchor'])
        anchor['verified_at']=cohort['clock']['now'].isoformat()
        anchor['anchor_sha256']=forward.legacy.digest({k:v for k,v in anchor.items() if k!='anchor_sha256'})
        cohort['anchor_path'].write_bytes(forward.json_bytes(anchor))
    with pytest.raises(ValueError): evaluate(cohort,data)


@pytest.mark.parametrize('problem',['stale','gap','future','transport'])
def test_invalid_or_stale_public_source_rejects(cohort,problem):
    observed=pd.Timestamp('2022-03-22T04:00:00Z')
    cohort['clock']['now']=observed
    raw=raw_bars(observed)
    if problem=='gap': raw['ETHUSDT'].pop(100)
    if problem=='future': observed+=pd.Timedelta('1h')
    if problem=='transport':
        with pytest.raises(ValueError): forward.import_snapshot(cohort['protocol_path'],raw,cohort['root']/'feed',observed,'unknown')
    elif problem in ('gap','future'):
        with pytest.raises(ValueError): forward.import_snapshot(cohort['protocol_path'],raw,cohort['root']/'feed',observed,'direct_https')
    else:
        data=snapshot(cohort,observed)
        cohort['clock']['now']+=pd.Timedelta('9h')
        with pytest.raises(ValueError): evaluate(cohort,data)


def test_bounded_collector_uses_only_fixed_spot_pages(cohort):
    observed=pd.Timestamp('2022-03-20T16:00:00Z')
    cohort['clock']['now']=observed
    raw=raw_bars(observed)
    calls=[]
    def fetch(symbol,begin,end):
        calls.append((symbol,begin,end))
        return [row for row in raw[symbol] if begin<=row[0]<=end][:1000]
    data=forward.collect(cohort['protocol_path'],cohort['root']/'feed',fetch=fetch)
    assert len(calls)==3 and set(c[0] for c in calls)==set(forward.SYMBOLS)
    assert json.loads((data/'manifest.json').read_text())['transport']=='direct_https'


def test_bounded_collector_cannot_silently_accept_a_short_history(cohort):
    cohort['clock']['now']=pd.Timestamp('2022-03-20T16:00:00Z')
    raw=raw_bars(cohort['clock']['now'])
    def fetch(symbol,begin,end):
        return [row for row in raw[symbol] if row[0]==begin]
    with pytest.raises(ValueError,match='Bounded collection'):
        forward.collect(cohort['protocol_path'],cohort['root']/'feed',fetch=fetch)


def test_screen_needs_real_closed_days_and_complete_independent_accounts():
    rows={s:{p:{'net_pnl':2 if p=='entry_20' else 1,'max_intraday_drawdown_bound':-.1,
                'accounting':{'reconciled':True}} for p in forward.POLICIES} for s in forward.SCENARIOS}
    audits={s:{p:{'reconciled':True} for p in forward.POLICIES} for s in forward.SCENARIOS}
    assert forward.screen(rows,audits,180*6)[1]
    assert not forward.screen(rows,audits,180*6-1)[1]
    rows['joint_stress']['entry_20']['max_intraday_drawdown_bound']=-.201
    assert not forward.screen(rows,audits,180*6)[1]
    with pytest.raises(ValueError): forward.screen(rows,{},180*6)


def test_concurrent_evaluation_lock_fails_closed(cohort):
    data=snapshot(cohort,'2022-03-20T16:00:00Z')
    out=cohort['root']/'accounts'
    out.mkdir()
    (out/'evaluation.lock').write_text('active')
    with pytest.raises(ValueError,match='Another evaluation'): evaluate(cohort,data)
    assert not (out/'state.json').exists()
