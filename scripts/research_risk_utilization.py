"""Fixed volatility-request comparison on reused spot history; offline only."""
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

from scripts import research_entry_hysteresis as entry
from spot_bot.portfolio.risk import PortfolioRisk

ROOT = entry.ROOT
PLAN = ROOT/'docs/evaluation/RISK_UTILIZATION_RESEARCH_PLAN.md'
PRIOR = ROOT/'docs/evaluation/THETABOT_ENTRY_HYSTERESIS_2026-10-25.json.gz'
POLICIES = {'control_15':('guarded_edge',.15), 'entry_15':('entry_hysteresis',.15),
            'entry_175':('entry_hysteresis',.175), 'entry_20':('entry_hysteresis',.20)}
CANDIDATE = 'entry_20'
SETTINGS = {'candidate':CANDIDATE,'policies':POLICIES,'capitals':entry.CAPITALS,
    'periods':entry.PERIODS,'scenarios':entry.SCENARIOS,'entry_hurdle':entry.ENTRY_HURDLE,
    'risk_template':asdict(entry.RISK),'costs':{'nominal':entry.NOMINAL,'stress':entry.STRESS},
    'total_target_cap':.5,'asset_target_cap':.25,'rebalance_destination':'band_edge',
    'rebalance_band':.01,'minimum_notional':10,'daily_control':True}
_WORKER = None


def fingerprints():
    paths = [Path(__file__).resolve(),PLAN]
    return entry.fingerprints() | {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prior_identity():
    payload = PRIOR.read_bytes()
    prior = json.loads(gzip.decompress(payload))
    return {'file':str(PRIOR.relative_to(ROOT)), 'sha256':hashlib.sha256(payload).hexdigest(),
            'protocol_sha256':prior['frozen_protocol']['protocol_sha256']}


def freeze(manifest,out):
    out.mkdir(parents=True,exist_ok=True)
    body = {'schema':1,'frozen_at':datetime.now(timezone.utc).isoformat(),
        'settings':json.loads(entry.canonical(SETTINGS)),'source_sha256':fingerprints(),
        'runtime':entry.runtime(),'source_manifest':manifest,'preceding_report':prior_identity(),
        'adaptive_followup':True,'dates_previously_studied':True,'live_eligible':False}
    proof = {**body,'protocol_sha256':hashlib.sha256(entry.canonical(body)).hexdigest()}
    with (out/'frozen_protocol.json').open('x') as f:
        f.write(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    return proof


def read_freeze(out,manifest):
    proof = json.loads((out/'frozen_protocol.json').read_text())
    body = {k:v for k,v in proof.items() if k!='protocol_sha256'}
    if (proof.get('protocol_sha256')!=hashlib.sha256(entry.canonical(body)).hexdigest()
            or entry.canonical(proof.get('settings'))!=entry.canonical(SETTINGS)
            or proof.get('source_sha256')!=fingerprints() or proof.get('runtime')!=entry.runtime()
            or proof.get('source_manifest')!=manifest or proof.get('preceding_report')!=prior_identity()
            or proof.get('adaptive_followup') is not True
            or proof.get('dates_previously_studied') is not True or proof.get('live_eligible') is not False):
        raise ValueError('Frozen source, runtime, settings, preceding report or input changed')
    return proof


def audit_configured_risk(markets,equity,trades,capital,costs,covariance,target):
    """Direct NumPy audit of actual CSV fields, without invoking risk.constrain."""
    if not np.isfinite(target) or target<=0:
        raise ValueError('Positive finite risk request required')
    result = entry.audit_ledger(markets,equity,trades,capital,costs)
    index = pd.DatetimeIndex(equity.timestamp)
    if covariance.shape!=(len(index),len(markets),len(markets)):
        raise ValueError('Aligned covariance required')
    prices = np.column_stack([f.loc[index,'open'] for f in markets.values()])
    qty = equity[[f'base_{s}' for s in markets]].to_numpy(dtype=float)
    cash = equity.usdt.to_numpy(dtype=float)
    previous_qty = np.vstack([np.zeros(len(markets)),qty[:-1]])
    previous_cash = np.r_[capital,cash[:-1]]
    before_nav = previous_cash+np.einsum('ij,ij->i',previous_qty,prices)
    after_nav = cash+np.einsum('ij,ij->i',qty,prices)
    before_w = previous_qty*prices/before_nav[:,None]
    after_w = qty*prices/after_nav[:,None]
    known = np.isfinite(covariance).all(axis=(1,2))
    before_vol = np.zeros(len(index))
    before_vol[known] = np.sqrt(np.maximum(0.,np.einsum('ij,ijk,ik->i',
        before_w[known],covariance[known],before_w[known])*365))
    after_vol = np.sqrt(np.maximum(0.,np.einsum('ij,ijk,ik->i',
        after_w[known],covariance[known],after_w[known])*365))
    scale = np.zeros(len(index))
    scale[known] = np.minimum(1.,target/np.maximum(before_vol[known],1e-100))
    unscheduled = ~equity.scheduled_decision.to_numpy(dtype=bool)
    decision_peak = np.maximum.accumulate(np.maximum(before_nav,np.r_[capital,equity.high_bound.to_numpy()[:-1]]))

    def residual(actual,expected):
        a,b = np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite risk audit field')
        return float(np.max(np.abs(a-b))) if a.size else 0.

    errors = {'max_risk_volatility_error':residual(equity.loc[known,'estimated_annual_volatility_after_fills'],after_vol),
        'max_unscheduled_risk_scale_error':residual(equity.loc[unscheduled,'risk_scale'],scale[unscheduled]),
        'max_unscheduled_planned_exposure_error':residual(equity.loc[unscheduled,'planned_exposure'],
            before_w.sum(axis=1)[unscheduled]*scale[unscheduled]),
        'max_decision_nav_error':residual(equity.decision_nav,before_nav),
        'max_decision_peak_error':residual(equity.decision_peak_bound,decision_peak)}
    if equity.loc[~known,'estimated_annual_volatility_after_fills'].notna().any():
        raise ValueError('Volatility estimate appeared without completed history')
    if ((trades.side=='buy') & (trades.risk_reduction | ~trades.scheduled_decision)).any():
        raise ValueError('Risk reduction or unscheduled decision bought inventory')
    if max(errors.values())>1e-7:
        raise ValueError(f'Independent configured risk audit residual exceeds tolerance: {errors}')
    return {**result,**errors,'configured_annual_volatility':target}


def initialize(markets,targets,covariance,out):
    global _WORKER
    _WORKER = markets,targets,covariance,out


def replay(task):
    capital,scenario,period,policy = task
    markets,targets,covariance,out = _WORKER
    signal,target = POLICIES[policy]
    start,end = (pd.Timestamp(v,tz='UTC') for v in entry.PERIODS[period])
    equity,trades,row = entry.run_intraday_spot(markets,targets[signal],start=start,end=end,
        initial_usdt=capital,daily_control=True,signal_delay_bars=int(scenario in ('delayed','joint_stress')),
        rebalance_destination='band_edge',risk=PortfolioRisk(volatility_target=target),
        completed_covariances=covariance,
        **(entry.STRESS if scenario in ('cost_stress','joint_stress') else entry.NOMINAL))
    row['accounting'] = entry.reconcile_spot_path(markets,equity,trades,capital)
    extra = {'realized_daily_annual_volatility':entry.realized_daily_volatility(equity,capital)}
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
    return {str(c):{s:{period:{p:paths[c,s,period,p] for p in POLICIES}
        for period in entry.PERIODS} for s in entry.SCENARIOS} for c in entry.CAPITALS}


def audit_exports(markets,out,paths):
    covariance = {lag:entry.independent_covariances(markets,lag) for lag in (1,2)}
    result = {}
    for scenario in ('nominal','joint_stress'):
        costs = entry.STRESS if scenario=='joint_stress' else entry.NOMINAL
        for policy,(_,target) in POLICIES.items():
            stem=f'10000_{scenario}_{policy}'
            equity,trades = (pd.read_csv(out/f'{stem}_{kind}.csv') for kind in ('equity','trades'))
            equity.timestamp=pd.to_datetime(equity.timestamp,utc=True)
            trades.timestamp=pd.to_datetime(trades.timestamp,utc=True)
            result[stem]=audit_configured_risk(markets,equity,trades,10000,costs,
                covariance[2 if scenario=='joint_stress' else 1],target)
            row=paths[10000,scenario,'full_continuous',policy]
            pairs=((equity.equity.iloc[-1],row['final_equity']),
                (equity.intraday_drawdown_bound.min(),row['max_intraday_drawdown_bound']),
                (trades.fee.sum(),row['fees_paid_total']),(trades.slippage.sum(),row['slippage_paid_total']))
            if len(trades)!=row['trades_count'] or not all(np.isclose(a,b,rtol=1e-12,atol=1e-8) for a,b in pairs):
                raise ValueError('Exported ledger differs from reported economics')
    return result


def research(data,out,jobs=4):
    markets,manifest = entry.load_spot_flow(data)
    proof = read_freeze(out,manifest)
    if (out/'frozen_selection.json').exists():
        raise ValueError('Prior payoffs exist; preserve the fixed selection')
    archives = entry.verify_archive_cache(data,manifest)
    prior = json.loads(gzip.decompress(PRIOR.read_bytes()))
    if manifest!=prior['frozen_protocol']['source_manifest']:
        raise ValueError('Preceding input manifest changed')
    all_targets=entry.completed_entry_targets(markets)
    targets={s:all_targets[s] for s in ('guarded_edge','entry_hysteresis')}
    for name,target in targets.items():
        target.to_csv(out/f'targets_{name}.csv',index_label='bar_open')
    covariance=entry.completed_risk_covariances(markets,entry.RISK)
    initialize(markets,targets,covariance,out)
    pool=ProcessPoolExecutor(max_workers=jobs,initializer=initialize,
        initargs=(markets,targets,covariance,out)) if jobs>1 else None
    try:
        dev,dev_extra=evaluate([(c,s,'development',p) for c in entry.CAPITALS for s in entry.SCENARIOS for p in POLICIES],pool)
        selection={'protocol_sha256':proof['protocol_sha256'],'candidate':CANDIDATE,
            'frozen_at':datetime.now(timezone.utc).isoformat(),'criterion':'Candidate fixed before all payoffs; no reselection',
            'development':{str(c):{s:{p:dev[c,s,'development',p] for p in POLICIES} for s in entry.SCENARIOS} for c in entry.CAPITALS}}
        with (out/'frozen_selection.json').open('x') as f:
            f.write(json.dumps(selection,indent=2,allow_nan=False)+'\n')
        later,later_extra=evaluate([(c,s,period,p) for c in entry.CAPITALS for s in entry.SCENARIOS
            for period in ('later_continuous','full_continuous') for p in POLICIES],pool)
    finally:
        if pool:
            pool.shutdown()
    paths=dev|later
    checked=0
    for (c,s,period,p),row in paths.items():
        if p in ('control_15','entry_15'):
            entry.compare_control(row,prior['results'][str(c)][s][period][POLICIES[p][0]],f'{c}/{s}/{period}/{p}')
            checked+=1
    gates={'accounting':all(r['accounting']['reconciled'] for r in paths.values()),
        'candidate_all_drawdowns_within_budget':all(paths[c,s,p,CANDIDATE]['max_intraday_drawdown_bound']>=-.20
            for c in entry.CAPITALS for s in entry.SCENARIOS for p in entry.PERIODS),
        'candidate_later_full_positive_all_stresses':all(paths[c,s,p,CANDIDATE]['net_pnl']>0
            for c in entry.CAPITALS for s in entry.SCENARIOS for p in ('later_continuous','full_continuous')),
        'candidate_beats_both_15_controls_later_full_nominal_joint':all(
            paths[c,s,p,CANDIDATE]['net_pnl']>paths[c,s,p,control]['net_pnl']
            for c in entry.CAPITALS for s in ('nominal','joint_stress')
            for p in ('later_continuous','full_continuous') for control in ('control_15','entry_15'))}
    report={'frozen_protocol':proof,'frozen_selection':selection,'verified_archives':archives,
        'preceding_control_checks':checked,'reconciled_paths':len(paths),'results':nest(paths),
        'diagnostics':nest(dev_extra|later_extra),'profit_gates':gates,
        'historical_profit_screen_passed':bool(all(gates.values())),
        'independent_csv_audit':audit_exports(markets,out,paths),'live_eligible':False,
        'predictive_signal_changed':False,'limitations':[
            'Adaptive follow-up on previously studied dates and three surviving assets; no fresh holdout.',
            'Greater fixed risk utilization changes exposure and loss risk, not predictive information.',
            'No diagnostic replaces the fixed candidate after results.',
            'Modeled opens, fixed fees and terminal marks; no executable/account/EEA, tax, operating-cost, custody or quote-FX validation.',
            'Historical 20% drawdown selection and annual-volatility requests do not cap future losses.']}
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}
    (out/'csv_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    return report


def export_report(report,out,destination):
    destination.mkdir(parents=True,exist_ok=True)
    stem='THETABOT_RISK_UTILIZATION_2026-10-25'
    (destination/(stem+'.json.gz')).write_bytes(gzip.compress(entry.canonical(report),compresslevel=9,mtime=0))
    proof=report['frozen_protocol']
    (destination/'THETABOT_RISK_UTILIZATION_PROTOCOL_2026-10-25.json').write_text(json.dumps(proof,indent=2)+'\n')
    lines=['# Fixed risk utilization — 2026-10-25','',
        '**Adaptive follow-up on reused history; greater risk utilization is not new alpha.**','',
        f"Fixed candidate `entry_20` historical screen: **{report['historical_profit_screen_passed']}**.",'',
        'Full: January 2022–September 2026; later: independently funded January 2025–September 2026.',
        'Net is after modeled trading costs, before tax and operating costs. Terminal holdings are marked.','']
    for capital,scenarios in report['results'].items():
        lines += [f'## Initial {capital} USDT','',
            '| Policy | Full nominal | Full joint stress | Worst full DD | Nominal CAGR | Later nominal | Later joint |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for p in POLICIES:
            full=scenarios['nominal']['full_continuous'][p]
            joint=scenarios['joint_stress']['full_continuous'][p]
            dd=min(s['full_continuous'][p]['max_intraday_drawdown_bound'] for s in scenarios.values())
            later=scenarios['nominal']['later_continuous'][p]
            latejoint=scenarios['joint_stress']['later_continuous'][p]
            lines += [f"| {p} | {full['total_return']:+.2%} | {joint['total_return']:+.2%} | {dd:.2%} | {full['cagr']:.2%} | {later['total_return']:+.2%} | {latejoint['total_return']:+.2%} |"]
        lines += ['', '| Policy | Requested vol | Realized annual vol | Mean exposure | Fees + impact | Fills |',
            '|---|---:|---:|---:|---:|---:|']
        for p,(_,target) in POLICIES.items():
            row=scenarios['nominal']['full_continuous'][p]
            vol=report['diagnostics'][capital]['nominal']['full_continuous'][p]['realized_daily_annual_volatility']
            lines += [f"| {p} | {target:.1%} | {vol:.2%} | {row['mean_exposure']:.2%} | {row['fees_paid_total']+row['slippage_paid_total']:.2f} | {row['trades_count']} |"]
        lines += ['']
    audit=report['independent_csv_audit']
    lines += ['## Fixed gates and evidence','',
        *[f'- {k}: **{v}**' for k,v in report['profit_gates'].items()],
        f"- {report['verified_archives']} official archives rechecked; {report['preceding_control_checks']} preceding 15% controls reproduce.",
        f"- {report['reconciled_paths']} funded paths reconcile; {len(audit)} complete raw CSV accounts independently audited.",
        f"- Independent audit covers {sum(a['bars'] for a in audit.values()):,} bars and {sum(a['fills'] for a in audit.values()):,} fills.",
        f"- Maximum independent NAV residual: {max(a['max_nav_error'] for a in audit.values()):.3g} USDT.",
        f"- Maximum independent volatility residual: {max(a['max_risk_volatility_error'] for a in audit.values()):.3g}.",
        f"- Protocol SHA-256: `{proof['protocol_sha256']}`.",
        f"- Preceding report SHA-256: `{proof['preceding_report']['sha256']}`.",'',
        '## Limits','',*['- '+s for s in report['limitations']],'',
        'Live eligibility remains false. [Frozen plan](RISK_UTILIZATION_RESEARCH_PLAN.md).','',
        '## Reproduce','', '```bash','python -m scripts.download_spot_flow',
        'python -m scripts.research_entry_hysteresis freeze',
        'python -m scripts.research_entry_hysteresis run --jobs 4',
        'python -m scripts.research_risk_utilization freeze',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_risk_utilization run --jobs 4','```','']
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
    parser.add_argument('--out',type=Path,default=Path('artifacts/risk_utilization_20261025'))
    parser.add_argument('--report-out',type=Path,default=Path('docs/evaluation'))
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    if args.jobs<1:
        parser.error('--jobs must be positive')
    if args.action=='freeze':
        _,manifest=entry.load_spot_flow(args.data)
        print(freeze(manifest,args.out)['protocol_sha256'])
    else:
        report=research(args.data,args.out,args.jobs)
        export_report(report,args.out,args.report_out)
        print(json.dumps({'profit_gates':report['profit_gates'],'live_eligible':False},indent=2))


if __name__=='__main__':
    main()
