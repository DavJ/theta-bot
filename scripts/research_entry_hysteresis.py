"""Frozen cost-aware momentum-entry comparison on already studied spot history."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from scripts.research_allocation_benchmarks import canonical, runtime, CAPITALS, PERIODS, SCENARIOS, RISK
from scripts.research_cost_risk import compare_control, verify_archive_cache
from scripts.research_spot_flow import load_spot_flow, NOMINAL, STRESS
from scripts.audit_cost_risk import audit_ledger, independent_covariances
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.execution_risk import completed_risk_covariances
from spot_bot.portfolio.momentum_refinement import momentum_refinement_targets
from theta_bot_averaging.allocation_benchmarks import realized_daily_volatility
from theta_bot_averaging.entry_hysteresis import VARIANTS, ENTRY_HURDLE, completed_entry_targets

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'docs/evaluation/ENTRY_HYSTERESIS_RESEARCH_PLAN.md'
CANDIDATE = 'entry_hysteresis'
CONTROL = 'guarded_edge'
SETTINGS = {'candidate':CANDIDATE,'policies':VARIANTS,'entry_hurdle':ENTRY_HURDLE,
    'capitals':CAPITALS,'periods':PERIODS,'scenarios':SCENARIOS,'risk':asdict(RISK),
    'costs':{'nominal':NOMINAL,'stress':STRESS},'rebalance_destination':'band_edge',
    'rebalance_band':.01,'minimum_notional':10,'total_target_cap':.5,'asset_target_cap':.25}
_WORKER = None


def fingerprints():
    paths = sorted((ROOT/'spot_bot').rglob('*.py'))
    paths += [ROOT/p for p in ('theta_bot_averaging/entry_hysteresis.py',
        'theta_bot_averaging/allocation_benchmarks.py','scripts/research_entry_hysteresis.py',
        'scripts/research_allocation_benchmarks.py','scripts/research_cost_risk.py',
        'scripts/research_spot_flow.py','scripts/audit_cost_risk.py','scripts/download_spot_flow.py',
        'scripts/download_binance_archive.py','scripts/download_external_signals.py',
        'requirements.txt','docs/evaluation/ENTRY_HYSTERESIS_RESEARCH_PLAN.md')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def freeze(manifest,out):
    out.mkdir(parents=True,exist_ok=True)
    body = {'schema':1,'frozen_at':datetime.now(timezone.utc).isoformat(),
        'settings':json.loads(canonical(SETTINGS)),'source_sha256':fingerprints(),
        'runtime':runtime(),'source_manifest':manifest,'dates_previously_studied':True,'live_eligible':False}
    proof = {**body,'protocol_sha256':hashlib.sha256(canonical(body)).hexdigest()}
    with (out/'frozen_protocol.json').open('x') as f:
        f.write(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    return proof


def read_freeze(out,manifest):
    proof = json.loads((out/'frozen_protocol.json').read_text())
    body = {k:v for k,v in proof.items() if k!='protocol_sha256'}
    if (proof.get('protocol_sha256')!=hashlib.sha256(canonical(body)).hexdigest()
            or canonical(proof.get('settings'))!=canonical(SETTINGS)
            or proof.get('source_sha256')!=fingerprints() or proof.get('runtime')!=runtime()
            or proof.get('source_manifest')!=manifest or proof.get('live_eligible') is not False
            or proof.get('dates_previously_studied') is not True):
        raise ValueError('Frozen source, runtime, settings or input changed')
    return proof


def initialize(markets,targets,covariance,out):
    global _WORKER
    _WORKER = markets,targets,covariance,out


def replay(task):
    capital,scenario,period,policy = task
    markets,targets,covariance,out = _WORKER
    start,end = (pd.Timestamp(v,tz='UTC') for v in PERIODS[period])
    equity,trades,row = run_intraday_spot(markets,targets[policy],start=start,end=end,
        initial_usdt=capital,daily_control=True,signal_delay_bars=int(scenario in ('delayed','joint_stress')),
        rebalance_destination='band_edge',risk=RISK,completed_covariances=covariance,
        **(STRESS if scenario in ('cost_stress','joint_stress') else NOMINAL))
    row['accounting'] = reconcile_spot_path(markets,equity,trades,capital)
    risk = trades.risk_reduction.astype(bool)
    extra = {'realized_daily_annual_volatility':realized_daily_volatility(equity,capital),
        'costs_by_annotation':{name:{'fills':int(mask.sum()),
            'fees_plus_impact':float(trades.loc[mask,'fee'].sum()+trades.loc[mask,'slippage'].sum())}
            for name,mask in (('risk_reduction',risk),('other',~risk))}}
    if capital==10000 and period=='full_continuous' and scenario in ('nominal','joint_stress'):
        stem=f'{capital}_{scenario}_{policy}'
        equity.to_csv(out/f'{stem}_equity.csv',index=False)
        trades.to_csv(out/f'{stem}_trades.csv',index=False)
    return task,row,extra


def evaluate(tasks,executor):
    results,diagnostics = {},{}
    for task,row,extra in executor.map(replay,tasks) if executor else map(replay,tasks):
        results[task],diagnostics[task] = row,extra
        print(f"{task}: net {row['total_return']:+.2%}; DD {row['max_intraday_drawdown_bound']:.2%}",flush=True)
    return results,diagnostics


def nest(paths):
    return {str(c):{s:{period:{p:paths[c,s,period,p] for p in VARIANTS}
        for period in PERIODS} for s in SCENARIOS} for c in CAPITALS}


def audit_exports(markets,out,paths):
    covariance = {lag:independent_covariances(markets,lag) for lag in (1,2)}
    result = {}
    for scenario in ('nominal','joint_stress'):
        costs = STRESS if scenario=='joint_stress' else NOMINAL
        for policy in VARIANTS:
            stem=f'10000_{scenario}_{policy}'
            equity,trades = (pd.read_csv(out/f'{stem}_{kind}.csv') for kind in ('equity','trades'))
            equity.timestamp=pd.to_datetime(equity.timestamp,utc=True)
            trades.timestamp=pd.to_datetime(trades.timestamp,utc=True)
            result[stem]=audit_ledger(markets,equity,trades,10000,costs,
                covariance[2 if scenario=='joint_stress' else 1])
            row=paths[10000,scenario,'full_continuous',policy]
            pairs=((equity.equity.iloc[-1],row['final_equity']),
                (equity.intraday_drawdown_bound.min(),row['max_intraday_drawdown_bound']),
                (trades.fee.sum(),row['fees_paid_total']),(trades.slippage.sum(),row['slippage_paid_total']))
            if len(trades)!=row['trades_count'] or not all(np.isclose(a,b,rtol=1e-12,atol=1e-8) for a,b in pairs):
                raise ValueError('Exported ledger differs from reported economics')
    return result


def research(data,out,jobs=4):
    markets,manifest = load_spot_flow(data)
    proof = read_freeze(out,manifest)
    if (out/'frozen_selection.json').exists():
        raise ValueError('Prior payoffs exist; preserve the fixed selection')
    archives=verify_archive_cache(data,manifest)
    old=json.loads(gzip.decompress((ROOT/'docs/evaluation/THETABOT_COST_RISK_2026-10-13.json.gz').read_bytes()))
    for symbol,entry in manifest['datasets'].items():
        prior=old['source_manifest']['datasets'][symbol]
        if [(a['month'],a['sha256'],a['rows']) for a in entry['archives']]!=[
                (a['month'],a['sha256'],a['rows']) for a in prior['archives']]:
            raise ValueError('Official archive history changed')
    targets=completed_entry_targets(markets)
    control=momentum_refinement_targets(markets)[0]['capped_allocation']
    pd.testing.assert_frame_equal(targets[CONTROL],control,rtol=1e-13,atol=1e-14)
    for name,target in targets.items():
        target.to_csv(out/f'targets_{name}.csv',index_label='bar_open')
    covariance=completed_risk_covariances(markets,RISK)
    initialize(markets,targets,covariance,out)
    pool=ProcessPoolExecutor(max_workers=jobs,initializer=initialize,
        initargs=(markets,targets,covariance,out)) if jobs>1 else None
    try:
        dev,dev_extra=evaluate([(c,s,'development',p) for c in CAPITALS for s in SCENARIOS for p in VARIANTS],pool)
        selection={'protocol_sha256':proof['protocol_sha256'],'candidate':CANDIDATE,
            'frozen_at':datetime.now(timezone.utc).isoformat(),'criterion':'Candidate fixed before all payoffs; no reselection',
            'development':{str(c):{s:{p:dev[c,s,'development',p] for p in VARIANTS} for s in SCENARIOS} for c in CAPITALS}}
        with (out/'frozen_selection.json').open('x') as f:
            f.write(json.dumps(selection,indent=2,allow_nan=False)+'\n')
        later,later_extra=evaluate([(c,s,period,p) for c in CAPITALS for s in SCENARIOS
            for period in ('later_continuous','full_continuous') for p in VARIANTS],pool)
    finally:
        if pool:
            pool.shutdown()
    paths=dev|later
    checked=0
    for (c,s,period,p),row in paths.items():
        if p==CONTROL:
            compare_control(row,old['results'][str(c)][s][period][CONTROL],f'{c}/{s}/{period}')
            checked+=1
    gates={'accounting':all(r['accounting']['reconciled'] for r in paths.values()),
        'candidate_all_drawdowns_within_budget':all(paths[c,s,p,CANDIDATE]['max_intraday_drawdown_bound']>=-.20
            for c in CAPITALS for s in SCENARIOS for p in PERIODS),
        'candidate_later_full_positive_all_stresses':all(paths[c,s,p,CANDIDATE]['net_pnl']>0
            for c in CAPITALS for s in SCENARIOS for p in ('later_continuous','full_continuous')),
        'candidate_beats_control_later_full_nominal_joint':all(paths[c,s,p,CANDIDATE]['net_pnl']>paths[c,s,p,CONTROL]['net_pnl']
            for c in CAPITALS for s in ('nominal','joint_stress') for p in ('later_continuous','full_continuous'))}
    turnover={'lower_candidate_costs':all(
        paths[c,s,p,CANDIDATE]['fees_paid_total']+paths[c,s,p,CANDIDATE]['slippage_paid_total']
        < paths[c,s,p,CONTROL]['fees_paid_total']+paths[c,s,p,CONTROL]['slippage_paid_total']
        for c in CAPITALS for s in ('nominal','joint_stress') for p in ('later_continuous','full_continuous')),
        'fewer_candidate_fills':all(paths[c,s,p,CANDIDATE]['trades_count']<paths[c,s,p,CONTROL]['trades_count']
            for c in CAPITALS for s in ('nominal','joint_stress') for p in ('later_continuous','full_continuous'))}
    report={'frozen_protocol':proof,'frozen_selection':selection,'verified_archives':archives,
        'archived_control_checks':checked,'reconciled_paths':len(paths),'results':nest(paths),
        'diagnostics':nest(dev_extra|later_extra),'profit_gates':gates,'turnover_diagnostics':turnover,
        'historical_profit_screen_passed':bool(all(gates.values())),
        'turnover_hypothesis_supported':bool(all(gates.values()) and all(turnover.values())),
        'independent_csv_audit':audit_exports(markets,out,paths),'live_eligible':False,
        'limitations':['Known dates and surviving three-asset universe; no fresh holdout.',
            'Trailing 30-day strength is not an estimated future net edge; eligibility state is not filled inventory.',
            'Diagnostics do not replace the fixed candidate after observing outcomes.',
            'Shared requested risk settings do not ensure equal realized exposure or volatility.',
            'Other-cost annotations include all non-risk fills, including required cap sales.',
            'Modeled opens, fixed fees and terminal marks; no order-book, account/EEA, tax, operating-cost, custody or quote-FX validation.',
            'The historical 20% drawdown screen and 15% volatility request do not cap future losses.']}
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}
    (out/'csv_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    return report


def export_report(report,out,destination):
    destination.mkdir(parents=True,exist_ok=True)
    stem='THETABOT_ENTRY_HYSTERESIS_2026-10-25'
    (destination/(stem+'.json.gz')).write_bytes(gzip.compress(canonical(report),compresslevel=9,mtime=0))
    proof=report['frozen_protocol']
    (destination/'THETABOT_ENTRY_PROTOCOL_2026-10-25.json').write_text(json.dumps(proof,indent=2)+'\n')
    lines=['# Fixed momentum-entry hysteresis — 2026-10-25','',
        '**Previously studied history; fixed candidate, no new holdout or parameter search.**','',
        f"Historical profit screen: **{report['historical_profit_screen_passed']}**.",
        f"Cost/turnover hypothesis supported: **{report['turnover_hypothesis_supported']}**.",'',
        'Candidate enters above 0.64% completed 30-day momentum and stays eligible until momentum is nonpositive.',
        'All variants keep the same 15% requested volatility guard, caps, 1pp band-edge execution and modeled costs.',
        'Full: January 2022–September 2026; later: independently funded January 2025–September 2026.',
        'Net is after modeled trading costs, before tax and operating costs.','']
    for capital,scenarios in report['results'].items():
        lines += [f'## Initial {capital} USDT','',
            '| Policy | Full nominal | Full joint stress | Worst full DD | Later nominal | Later joint stress |',
            '|---|---:|---:|---:|---:|---:|']
        for p in VARIANTS:
            full=scenarios['nominal']['full_continuous'][p]
            joint=scenarios['joint_stress']['full_continuous'][p]
            dd=min(s['full_continuous'][p]['max_intraday_drawdown_bound'] for s in scenarios.values())
            later=scenarios['nominal']['later_continuous'][p]
            latejoint=scenarios['joint_stress']['later_continuous'][p]
            lines += [f"| {p} | {full['total_return']:+.2%} | {joint['total_return']:+.2%} | {dd:.2%} | {later['total_return']:+.2%} | {latejoint['total_return']:+.2%} |"]
        lines += ['', '| Policy | Full fees + impact | Full fills | Risk-reduction fills | Mean exposure | Realized annual vol |',
            '|---|---:|---:|---:|---:|---:|']
        for p in VARIANTS:
            row=scenarios['nominal']['full_continuous'][p]
            vol=report['diagnostics'][capital]['nominal']['full_continuous'][p]['realized_daily_annual_volatility']
            lines += [f"| {p} | {row['fees_paid_total']+row['slippage_paid_total']:.2f} | {row['trades_count']} | {row['risk_reduction_fills']} | {row['mean_exposure']:.2%} | {vol:.2%} |"]
        lines += ['']
    audit=report['independent_csv_audit']
    lines += ['## Fixed gates and evidence','',
        *[f'- {k}: **{v}**' for k,v in report['profit_gates'].items()],
        *[f'- {k}: **{v}**' for k,v in report['turnover_diagnostics'].items()],
        f"- {report['verified_archives']} official monthly archives rechecked; {report['archived_control_checks']} archived control paths reproduce.",
        f"- {report['reconciled_paths']} funded paths reconcile cash, owned inventory and NAV.",
        f"- {len(audit)} complete CSV accounts independently audited: {sum(a['bars'] for a in audit.values()):,} bars and {sum(a['fills'] for a in audit.values()):,} fills.",
        f"- Maximum independent NAV residual: {max(a['max_nav_error'] for a in audit.values()):.3g} USDT.",
        f"- Protocol SHA-256: `{proof['protocol_sha256']}`.",'',
        '## Limits','',*['- '+s for s in report['limitations']],'',
        'Live eligibility remains false. [Frozen plan](ENTRY_HYSTERESIS_RESEARCH_PLAN.md).','',
        '## Reproduce','', '```bash','python -m scripts.download_spot_flow',
        'python -m scripts.research_entry_hysteresis freeze',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_entry_hysteresis run --jobs 4','```','']
    (destination/(stem+'.md')).write_text('\n'.join(lines))
    archive=out/'evidence.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(out.iterdir()):
            if path.suffix in ('.csv','.json'):
                z.write(path,path.name)
        z.write(PLAN,PLAN.name)
    payload=archive.read_bytes()
    name=stem+'_EVIDENCE.zip'
    parts=[]
    for i,start in enumerate(range(0,len(payload),5_000_000),1):
        path=destination/f'{name}.part{i:03d}'
        part=payload[start:start+5_000_000]
        path.write_bytes(part)
        parts.append({'file':path.name,'bytes':len(part),'sha256':hashlib.sha256(part).hexdigest()})
    (destination/(name+'.manifest.json')).write_text(json.dumps({'file':name,'bytes':len(payload),
        'sha256':hashlib.sha256(payload).hexdigest(),'parts':parts,
        'reassemble':'Concatenate listed parts and verify complete SHA-256 before unzipping.'},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--data',type=Path,default=Path('data/raw/spot_flow'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/entry_hysteresis_20261025'))
    parser.add_argument('--report-out',type=Path,default=Path('docs/evaluation'))
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    if args.jobs<1:
        parser.error('--jobs must be positive')
    if args.action=='freeze':
        _,manifest=load_spot_flow(args.data)
        print(freeze(manifest,args.out)['protocol_sha256'])
    else:
        report=research(args.data,args.out,args.jobs)
        export_report(report,args.out,args.report_out)
        print(json.dumps({'profit_gates':report['profit_gates'],'turnover':report['turnover_diagnostics'],
                         'live_eligible':False},indent=2))


if __name__=='__main__':
    main()
