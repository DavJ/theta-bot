"""Separately frozen prospective entry-policy model accounts; public data only."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import pandas as pd
import requests

from spot_bot import forward_validation as legacy
from spot_bot.portfolio.risk import PortfolioRisk
from scripts import research_risk_utilization as research

ROOT=research.ROOT
SOURCE=legacy.SOURCE
SYMBOLS=legacy.SYMBOLS
POLICIES=research.POLICIES
SCENARIOS=legacy.SCENARIOS
CANDIDATE='entry_20'
PROTOCOL_GIT_PATH='docs/evaluation/THETABOT_ENTRY_FORWARD_PROTOCOL_2026-10-25.json'
REPOSITORY='DavJ/theta-bot'
WORKFLOW='.github/workflows/eval-metrics.yml'
HISTORY_START='2022-01-01T00:00:00+00:00'
SETTINGS={'candidate':CANDIDATE,'policies':POLICIES,'scenarios':SCENARIOS,
    'symbols':list(SYMBOLS),'initial_usdt':10000.,'entry_hurdle':research.entry.ENTRY_HURDLE,
    'costs':{'nominal':research.entry.NOMINAL,'stress':research.entry.STRESS},
    'max_exposure':.5,'asset_cap':.25,'rebalance_band':.01,'minimum_notional':10.,
    'rebalance_destination':'band_edge','volatility_window':60,'covariance_shrinkage':.25,
    'historical_drawdown_selection_budget':.20,
    'minimum_forward_days':180,'maximum_feed_age_hours':8,'history_start':HISTORY_START,
    'maximum_pages_per_symbol':20,'source':SOURCE,'publication_repository':REPOSITORY,
    'publication_protocol_path':PROTOCOL_GIT_PATH,'publication_workflow':WORKFLOW}


def utc_now():
    return legacy.utc_now()


def source_fingerprints():
    paths=[ROOT/'theta_bot_averaging/entry_forward.py',ROOT/'scripts/research_entry_forward.py',
           ROOT/'docs/evaluation/ENTRY_FORWARD_VALIDATION_PLAN.md']
    return legacy.source_fingerprints() | research.fingerprints() | {
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def runtime_versions():
    return {**legacy.runtime_versions(),'requests':importlib.metadata.version('requests')}


def json_bytes(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def git_blob(payload):
    return hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()


def freeze_protocol(path,start=None):
    now=utc_now()
    start=legacy.utc(start) if start is not None else now.normalize()+pd.Timedelta('1D')
    if start<=now or start!=start.normalize() or legacy.utc(HISTORY_START)>=start:
        raise ValueError('A future UTC midnight after fixed feature history is required')
    body={'schema_version':1,'study_kind':'prospective_fixed_entry_model_replay',
        'frozen_at':now.isoformat(),'forward_start':start.isoformat(),'warmup_start':HISTORY_START,
        'minimum_assessment_at':(start+pd.Timedelta(days=SETTINGS['minimum_forward_days'])).isoformat(),
        'settings':json.loads(legacy.canonical(SETTINGS)),'source_sha256':source_fingerprints(),
        'runtime':runtime_versions(),'publication_receipt_required':True,'live_eligible':False}
    protocol={**body,'protocol_sha256':legacy.digest(body)}
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        f.write(json_bytes(protocol))
    return protocol


def read_protocol(path):
    payload=Path(path).read_bytes()
    protocol=json.loads(payload)
    body={k:v for k,v in protocol.items() if k!='protocol_sha256'}
    if (protocol.get('protocol_sha256')!=legacy.digest(body)
            or protocol.get('schema_version')!=1 or protocol.get('study_kind')!='prospective_fixed_entry_model_replay'
            or legacy.canonical(protocol.get('settings'))!=legacy.canonical(SETTINGS)
            or protocol.get('publication_receipt_required') is not True or protocol.get('live_eligible') is not False
            or payload!=json_bytes(protocol)):
        raise ValueError('Unsupported or altered fixed entry protocol')
    start,frozen=legacy.utc(protocol['forward_start']),legacy.utc(protocol['frozen_at'])
    if (frozen>=start or frozen>utc_now() or start!=start.normalize()
            or protocol.get('warmup_start')!=HISTORY_START
            or legacy.utc(protocol['minimum_assessment_at'])!=start+pd.Timedelta(days=180)):
        raise ValueError('Invalid prospective time boundary or feature history')
    if protocol.get('source_sha256')!=source_fingerprints() or protocol.get('runtime')!=runtime_versions():
        raise ValueError('Frozen source/runtime changed; preserve this cohort and create a separate freeze')
    return protocol


def expected_git_blobs(protocol):
    return {p:git_blob((ROOT/p).read_bytes()) for p in protocol['source_sha256']}


def write_publication(protocol_path,run,tree,path):
    protocol=read_protocol(protocol_path)
    created=legacy.utc(run['created_at'])
    commit=run['head_sha']
    run_id=run['id']
    if (type(run_id) is not int or run_id<=0 or not re.fullmatch('[0-9a-f]{40}',commit)
            or run.get('repository',{}).get('full_name')!=REPOSITORY or run.get('path')!=WORKFLOW
            or run.get('head_commit',{}).get('id')!=commit
            or not re.fullmatch('[0-9a-f]{40}',tree.get('sha',''))
            or tree.get('sha')!=run.get('head_commit',{}).get('tree_id') or tree.get('truncated') is not False
            or not legacy.utc(protocol['frozen_at'])<=created<legacy.utc(protocol['forward_start'])
            or created>utc_now()):
        raise ValueError('GitHub publication run/tree or timestamp does not anchor this prospective freeze')
    entries={r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob' and r['mode'] in ('100644','100755')}
    blobs=expected_git_blobs(protocol)
    protocol_blob=git_blob(json_bytes(protocol))
    expected=blobs | {PROTOCOL_GIT_PATH:protocol_blob}
    if any(entries.get(p)!=h for p,h in expected.items()):
        raise ValueError('Published Git tree differs from pinned source or exact protocol blob')
    url=f'https://github.com/{REPOSITORY}/actions/runs/{run_id}'
    if run.get('html_url')!=url:
        raise ValueError('Unexpected publication URL')
    body={'schema_version':1,'protocol_sha256':protocol['protocol_sha256'],
        'verified_at':utc_now().isoformat(),'repository':REPOSITORY,'workflow':WORKFLOW,
        'run_id':run_id,'run_created_at':created.isoformat(),'run_url':url,'source_commit':commit,
        'tree_sha':tree['sha'],'protocol_git_path':PROTOCOL_GIT_PATH,'protocol_git_blob_sha1':protocol_blob,
        'source_git_blobs':blobs,'verification':'Inspected public GitHub API run timestamp and Git tree; local receipt is not an unforgeable clock.'}
    anchor={**body,'anchor_sha256':legacy.digest(body)}
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        f.write(json_bytes(anchor))
    return anchor


def read_publication(path,protocol):
    anchor=json.loads(Path(path).read_text())
    body={k:v for k,v in anchor.items() if k!='anchor_sha256'}
    created,verified=legacy.utc(anchor['run_created_at']),legacy.utc(anchor['verified_at'])
    if (anchor.get('anchor_sha256')!=legacy.digest(body) or anchor.get('schema_version')!=1
            or anchor.get('protocol_sha256')!=protocol['protocol_sha256']
            or anchor.get('repository')!=REPOSITORY or anchor.get('workflow')!=WORKFLOW
            or type(anchor.get('run_id')) is not int or anchor['run_id']<=0
            or not re.fullmatch('[0-9a-f]{40}',anchor.get('source_commit',''))
            or not re.fullmatch('[0-9a-f]{40}',anchor.get('tree_sha',''))
            or anchor.get('run_url')!=f"https://github.com/{REPOSITORY}/actions/runs/{anchor['run_id']}"
            or anchor.get('protocol_git_path')!=PROTOCOL_GIT_PATH
            or anchor.get('protocol_git_blob_sha1')!=git_blob(json_bytes(protocol))
            or anchor.get('source_git_blobs')!=expected_git_blobs(protocol)
            or not legacy.utc(protocol['frozen_at'])<=created<legacy.utc(protocol['forward_start'])
            or not created<=verified<=utc_now()):
        raise ValueError('Publication receipt no longer binds this prospective cohort')
    return anchor


def import_snapshot(protocol_path,raw,out,observed_at,transport):
    protocol=read_protocol(protocol_path)
    if transport not in ('direct_https','binance_mcp_public_spot'):
        raise ValueError('Explicit supported public spot transport required')
    observed=legacy.utc(observed_at)
    if observed>utc_now() or observed<legacy.utc(protocol['frozen_at']) or set(raw)!=set(SYMBOLS):
        raise ValueError('Invalid snapshot observation or fixed spot markets')
    expected=pd.date_range(legacy.utc(HISTORY_START),observed.floor('4h'),freq='4h',inclusive='left')
    for symbol in SYMBOLS:
        if not legacy.parse_closed_klines(raw[symbol],observed).index.equals(expected):
            raise ValueError('Complete aligned fixed feature history through the latest closed bar required')
    payloads={s:legacy.canonical(raw[s]) for s in SYMBOLS}
    manifest={'source':SOURCE,'timeframe':'4h','observed_at':observed.isoformat(),
        'protocol_sha256':protocol['protocol_sha256'],'transport':transport,
        'encoding':'Canonical JSON of full returned 12-field arrays; not HTTP wire bytes',
        'datasets':{s:{'file':f'{s}_4h.json','rows':len(raw[s]),
            'sha256':hashlib.sha256(payloads[s]).hexdigest()} for s in SYMBOLS}}
    destination=Path(out)/('snapshot-'+observed.strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(parents=True,exist_ok=False)
    for s in SYMBOLS:
        (destination/manifest['datasets'][s]['file']).write_bytes(payloads[s])
    with (destination/'manifest.json').open('xb') as f:
        f.write(json_bytes(manifest))
    return destination


def collect(protocol_path,out,fetch=None):
    protocol=read_protocol(protocol_path)
    observed=utc_now()
    first=int(legacy.utc(protocol['warmup_start']).timestamp()*1000)
    end=int(observed.floor('4h').timestamp()*1000)-1

    def public_page(symbol,begin,finish):
        response=requests.get(SOURCE,params={'symbol':symbol,'interval':'4h','timeZone':'0',
            'startTime':begin,'endTime':finish,'limit':1000},timeout=25)
        response.raise_for_status()
        return response.json()

    fetch=fetch or public_page
    raw={}
    for symbol in SYMBOLS:
        cursor,rows=first,[]
        for _ in range(SETTINGS['maximum_pages_per_symbol']):
            if cursor>end:
                break
            page=fetch(symbol,cursor,end)
            if (not isinstance(page,list) or not page or len(page)>1000
                    or any(not isinstance(r,list) or len(r)!=12 for r in page)):
                raise ValueError('Missing or oversized public spot page')
            times=[r[0] for r in page]
            if times!=list(range(cursor,cursor+len(page)*14400000,14400000)) or times[-1]>end:
                raise ValueError('Public spot page has a gap, duplicate or wrong range')
            rows.extend(page)
            cursor=times[-1]+14400000
        if cursor<=end:
            raise ValueError('Bounded collection did not cover the fixed full feature history')
        raw[symbol]=rows
    return import_snapshot(protocol_path,raw,out,observed,'direct_https')


def verify_history(out,state,raw,protocol,anchor):
    if type(state.get('sequence')) is not int or state['sequence']<=0:
        raise ValueError('Invalid funded receipt sequence')
    receipt=legacy.verify_previous(out,state,raw,protocol)
    if state.get('anchor_sha256')!=anchor['anchor_sha256']:
        raise ValueError('Cannot change the publication anchor of a funded cohort')
    records=[]
    for run in (out/'runs').iterdir():
        if (not run.is_dir() or not re.fullmatch('run-[0-9a-f]{32}',run.name)
                or not (run/'receipt.json').is_file()):
            raise ValueError('Orphaned or invalid run requires inspection before continuing')
        payload=(run/'receipt.json').read_bytes()
        saved=json.loads(payload)
        if type(saved.get('sequence')) is not int or saved['sequence']<=0:
            raise ValueError('Invalid receipt sequence')
        records.append((saved['sequence'],run,saved,hashlib.sha256(payload).hexdigest()))
    records.sort(key=lambda r:r[0])
    if [r[0] for r in records]!=list(range(1,state.get('sequence',0)+1)):
        raise ValueError('State is not the latest complete immutable receipt sequence')
    previous_hash=None
    for sequence,run,saved,sha in records:
        if (saved.get('previous_receipt_sha256')!=previous_hash
                or saved.get('anchor_sha256')!=anchor['anchor_sha256']
                or saved['report'].get('protocol_sha256')!=protocol['protocol_sha256']):
            raise ValueError('Receipt chain or prospective identity changed')
        expected={f'{s}_{p}_{kind}.csv' for s in SCENARIOS for p in POLICIES for kind in ('equity','trades')}
        if not saved['report']['forward_closed_bars']:
            expected=set()
        if set(saved['ledger_sha256'])!=expected:
            raise ValueError('Previous funded ledger set is incomplete or altered')
        for name,sha256 in saved['ledger_sha256'].items():
            if hashlib.sha256((run/name).read_bytes()).hexdigest()!=sha256:
                raise ValueError('Previous immutable account ledger changed')
        previous_hash=sha
    if (not records or records[-1][1].name!=state['last_run'] or previous_hash!=state['receipt_sha256']
            or receipt['report']['as_of']!=state['as_of']):
        raise ValueError('State does not point to the latest accepted receipt')
    for s in SYMBOLS:
        entry=receipt['source_manifest']['datasets'][s]
        if state['source_rows'][s]!=entry['rows'] or state['prefix_sha256'][s]!=entry['sha256']:
            raise ValueError('Accepted source-prefix identity changed')
    return receipt


def screen(summaries,audits,forward_bars):
    available=bool(summaries)
    if available and (set(summaries)!=set(SCENARIOS) or set(audits)!=set(SCENARIOS)
            or any(set(summaries[s])!=set(POLICIES) or set(audits[s])!=set(POLICIES) for s in SCENARIOS)):
        raise ValueError('Complete fixed policy/scenario summaries and audits are required')
    if not available and audits:
        raise ValueError('Audit results cannot exist without funded summaries')
    drawdown=all(rows[CANDIDATE]['max_intraday_drawdown_bound']>=-.20 for rows in summaries.values()) if available else None
    gates={'enough_closed_forward_days':forward_bars>=180*6,
        'candidate_drawdowns_within_budget':drawdown,
        'candidate_positive_all_scenarios':all(rows[CANDIDATE]['net_pnl']>0 for rows in summaries.values()) if available else None,
        'candidate_beats_both_15_controls_nominal_joint':all(
            summaries[s][CANDIDATE]['net_pnl']>summaries[s][p]['net_pnl']
            for s in ('nominal','joint_stress') for p in ('control_15','entry_15')) if available else None,
        'accounting':all(row['accounting']['reconciled'] for rows in summaries.values() for row in rows.values()) if available else None,
        'independent_csv_accounting':all(row['reconciled'] for rows in audits.values() for row in rows.values()) if available else None}
    return gates,bool(all(value is True for value in gates.values()))


def evaluate_forward(protocol_path,anchor_path,data,out,as_of=None):
    protocol=read_protocol(protocol_path)
    anchor=read_publication(anchor_path,protocol)
    as_of=legacy.utc(as_of) if as_of is not None else utc_now()
    markets,raw,manifest=legacy.load_feed_snapshot(data,protocol,as_of)
    if manifest.get('transport') not in ('direct_https','binance_mcp_public_spot'):
        raise ValueError('Unknown public spot snapshot transport')
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    with legacy.evaluation_lock(out):
        state_path=out/'state.json'
        previous=json.loads(state_path.read_text()) if state_path.exists() else None
        old_receipt=None
        if previous is not None:
            old_receipt=verify_history(out,previous,raw,protocol,anchor)
            if as_of<legacy.utc(previous['as_of']):
                raise ValueError('Cannot rewind a funded cohort clock')
            if all(len(raw[s])==previous['source_rows'][s] for s in SYMBOLS):
                return old_receipt['report']
        elif (out/'runs').exists() and any((out/'runs').iterdir()):
            raise ValueError('Existing runs without state require inspection; cannot reset accounts')
        start=legacy.utc(protocol['forward_start'])
        end=next(iter(markets.values())).index[-1]+legacy.BAR
        forward_bars=max(0,int((end-start)/legacy.BAR))
        run_name='run-'+uuid4().hex
        run=out/'runs'/run_name
        run.mkdir(parents=True,exist_ok=False)
        summaries,audits={},{}
        if forward_bars:
            targets=research.entry.completed_entry_targets(markets)
            covariance=research.entry.completed_risk_covariances(markets,PortfolioRisk(volatility_target=.15))
            selection=next(iter(markets.values())).index>=start
            independent={lag:research.entry.independent_covariances(markets,lag)[selection] for lag in (1,2)}
            for scenario in SCENARIOS:
                summaries[scenario],audits[scenario]={},{}
                delayed=scenario in ('delayed','joint_stress')
                costs=research.entry.STRESS if scenario in ('cost_stress','joint_stress') else research.entry.NOMINAL
                for policy,(signal,vol) in POLICIES.items():
                    equity,trades,row=research.entry.run_intraday_spot(markets,targets[signal],start=start,end=end,
                        initial_usdt=10000,daily_control=True,signal_delay_bars=int(delayed),
                        rebalance_destination='band_edge',risk=PortfolioRisk(volatility_target=vol),
                        completed_covariances=covariance,**costs)
                    row['accounting']=research.entry.reconcile_spot_path(markets,equity,trades,10000)
                    stem=f'{scenario}_{policy}'
                    equity.to_csv(run/f'{stem}_equity.csv',index=False)
                    trades.to_csv(run/f'{stem}_trades.csv',index=False)
                    eq,fills=(pd.read_csv(run/f'{stem}_{kind}.csv') for kind in ('equity','trades'))
                    eq.timestamp=pd.to_datetime(eq.timestamp,utc=True)
                    fills.timestamp=pd.to_datetime(fills.timestamp,utc=True)
                    audits[scenario][policy]=research.audit_configured_risk(markets,eq,fills,10000,costs,
                        independent[2 if delayed else 1],vol)
                    summaries[scenario][policy]=row
        if old_receipt is not None:
            old_run=out/'runs'/previous['last_run']
            for name in old_receipt['ledger_sha256']:
                if not (run/name).read_bytes().startswith((old_run/name).read_bytes()):
                    raise ValueError('Full replay changed a previously accepted equity/fill prefix')
        gates,passed=screen(summaries,audits,forward_bars)
        phase=('waiting_for_start' if as_of<start else 'waiting_for_closed_bar') if not forward_bars else (
            'rejected_drawdown' if gates['candidate_drawdowns_within_budget'] is False else
            'collecting' if not gates['enough_closed_forward_days'] else
            'preliminary_pass' if passed else 'preliminary_fail')
        report={'protocol_sha256':protocol['protocol_sha256'],'publication_anchor_sha256':anchor['anchor_sha256'],
            'as_of':as_of.isoformat(),'snapshot_observed_at':manifest['observed_at'],
            'forward_start':start.isoformat(),'minimum_assessment_at':protocol['minimum_assessment_at'],
            'feature_history_start':HISTORY_START,'forward_closed_bars':forward_bars,
            'forward_full_days':forward_bars//6,'phase':phase,'gates':gates,
            'preliminary_screen_passed':passed,'live_eligible':False,'results':summaries,
            'independent_csv_audits':audits,'candidate':CANDIDATE,
            'limitations':['Prospective model replay, not observed executable paper or exchange orders.',
                'Publication receipt records inspected public GitHub API evidence; local checksums are not an unforgeable clock.',
                'Warmup contributes features only; each forward account starts with its own cash and no positions.',
                'Fixed modeled fees/impact and terminal marks; account/EEA, commission tier, tax, operating costs, custody and quote FX are unverified.',
                'Historical drawdown screens and requested volatility do not cap future losses; diagnostics never replace the fixed candidate.']}
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.glob('*.csv'))}
        sequence=previous['sequence']+1 if previous else 1
        receipt={'sequence':sequence,'anchor_sha256':anchor['anchor_sha256'],'report':report,
            'source_manifest':manifest,'ledger_sha256':hashes,
            'previous_receipt_sha256':previous['receipt_sha256'] if previous else None}
        payload=json_bytes(receipt)
        (run/'receipt.json').write_bytes(payload)
        state={'protocol_sha256':protocol['protocol_sha256'],'anchor_sha256':anchor['anchor_sha256'],
            'sequence':sequence,'last_run':run_name,'receipt_sha256':hashlib.sha256(payload).hexdigest(),
            'as_of':as_of.isoformat(),'source_rows':{s:len(raw[s]) for s in SYMBOLS},
            'prefix_sha256':{s:legacy.digest(raw[s]) for s in SYMBOLS}}
        temporary=out/('state-'+uuid4().hex+'.tmp')
        temporary.write_bytes(json_bytes(state))
        os.replace(temporary,state_path)
        return report
