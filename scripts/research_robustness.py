"""Frozen calendar/asset-omission audit of fixed spot policies; offline only."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from scripts import research_risk_utilization as prior
from theta_bot_averaging.robustness import (
    SYMBOLS,UNIVERSES,omission_targets,calendar_diagnostics,attribute_spot_pnl,
)

entry=prior.entry
ROOT=entry.ROOT
PLAN=ROOT/'docs/evaluation/ROBUSTNESS_RESEARCH_PLAN.md'
PRIOR=ROOT/'docs/evaluation/THETABOT_RISK_UTILIZATION_2026-10-25.json.gz'
PRIOR_ARCHIVE=ROOT/'docs/evaluation/THETABOT_RISK_UTILIZATION_2026-10-25_EVIDENCE.zip.manifest.json'
POLICIES=prior.POLICIES
CANDIDATE='entry_20'
SCENARIOS=('nominal','joint_stress')
CAPITAL=10000
SETTINGS={'candidate':CANDIDATE,'policies':POLICIES,'universes':UNIVERSES,'capital':CAPITAL,
    'scenarios':SCENARIOS,'periods':entry.PERIODS,'execution':prior.SETTINGS,
    'omission_rule':'Zero omitted target; retained eligible names each receive 25% before unchanged risk scaling',
    'rolling_months':[12,24],'concentration_months':3,'drawdown_selection_budget':.20}
_WORKER=None


def fingerprints():
    paths=[Path(__file__).resolve(),PLAN,ROOT/'theta_bot_averaging/robustness.py']
    return prior.fingerprints() | {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prior_identity():
    payload=PRIOR.read_bytes()
    report=json.loads(gzip.decompress(payload))
    archive=json.loads(PRIOR_ARCHIVE.read_text())
    return {'report_file':str(PRIOR.relative_to(ROOT)),'report_sha256':hashlib.sha256(payload).hexdigest(),
        'protocol_sha256':report['frozen_protocol']['protocol_sha256'],
        'archive_manifest_sha256':hashlib.sha256(PRIOR_ARCHIVE.read_bytes()).hexdigest(),
        'archive_sha256':archive['sha256']}


def freeze(manifest,out):
    out.mkdir(parents=True,exist_ok=True)
    body={'schema':1,'frozen_at':datetime.now(timezone.utc).isoformat(),
        'settings':json.loads(entry.canonical(SETTINGS)),'source_sha256':fingerprints(),
        'runtime':entry.runtime(),'source_manifest':manifest,'preceding_evidence':prior_identity(),
        'adaptive_followup':True,'dates_previously_studied':True,'live_eligible':False}
    proof={**body,'protocol_sha256':hashlib.sha256(entry.canonical(body)).hexdigest()}
    with (out/'frozen_protocol.json').open('x') as f:
        f.write(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    return proof


def read_freeze(out,manifest):
    proof=json.loads((out/'frozen_protocol.json').read_text())
    body={k:v for k,v in proof.items() if k!='protocol_sha256'}
    if (proof.get('protocol_sha256')!=hashlib.sha256(entry.canonical(body)).hexdigest()
            or entry.canonical(proof.get('settings'))!=entry.canonical(SETTINGS)
            or proof.get('source_sha256')!=fingerprints() or proof.get('runtime')!=entry.runtime()
            or proof.get('source_manifest')!=manifest or proof.get('preceding_evidence')!=prior_identity()
            or proof.get('adaptive_followup') is not True or proof.get('dates_previously_studied') is not True
            or proof.get('live_eligible') is not False):
        raise ValueError('Frozen source, runtime, input, preceding evidence or settings changed')
    return proof


def verify_prior_archive():
    manifest=json.loads(PRIOR_ARCHIVE.read_text())
    pieces=[]
    for part in manifest['parts']:
        payload=(PRIOR_ARCHIVE.parent/part['file']).read_bytes()
        if len(payload)!=part['bytes'] or hashlib.sha256(payload).hexdigest()!=part['sha256']:
            raise ValueError('Preceding evidence part changed')
        pieces.append(payload)
    payload=b''.join(pieces)
    if len(payload)!=manifest['bytes'] or hashlib.sha256(payload).hexdigest()!=manifest['sha256']:
        raise ValueError('Preceding evidence archive changed')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if z.testzip() is not None:
            raise ValueError('Preceding evidence CRC failed')
        hashes=json.loads(z.read('csv_hashes.json'))
        if any(hashlib.sha256(z.read(p)).hexdigest()!=h for p,h in hashes.items()):
            raise ValueError('Preceding CSV evidence changed')
        if json.loads(z.read('summary.json'))!=json.loads(gzip.decompress(PRIOR.read_bytes())):
            raise ValueError('Preceding report and archive disagree')
    return manifest['sha256']


def initialize(markets,targets,covariance,out):
    global _WORKER
    _WORKER=markets,targets,covariance,out


def replay(task):
    universe,scenario,period,policy=task
    markets,targets,covariance,out=_WORKER
    signal,vol=POLICIES[policy]
    start,end=(pd.Timestamp(v,tz='UTC') for v in entry.PERIODS[period])
    equity,trades,row=entry.run_intraday_spot(markets,targets[universe][signal],start=start,end=end,
        initial_usdt=CAPITAL,daily_control=True,signal_delay_bars=int(scenario=='joint_stress'),
        rebalance_destination='band_edge',risk=prior.PortfolioRisk(volatility_target=vol),
        completed_covariances=covariance,**(entry.STRESS if scenario=='joint_stress' else entry.NOMINAL))
    row['accounting']=entry.reconcile_spot_path(markets,equity,trades,CAPITAL)
    extra={'realized_daily_annual_volatility':entry.realized_daily_volatility(equity,CAPITAL)}
    if period=='full_continuous' and (universe=='all_three' or policy==CANDIDATE):
        stem=f'{universe}_{scenario}_{policy}'
        equity.to_csv(out/f'{stem}_equity.csv',index=False)
        trades.to_csv(out/f'{stem}_trades.csv',index=False)
    return task,row,extra


def evaluate(tasks,executor):
    results,diagnostics={},{}
    for task,row,extra in executor.map(replay,tasks) if executor else map(replay,tasks):
        results[task],diagnostics[task]=row,extra
        print(f"{task}: net {row['total_return']:+.2%}; DD {row['max_intraday_drawdown_bound']:.2%}",flush=True)
    return results,diagnostics


def nest(paths):
    return {u:{s:{period:{p:paths[u,s,period,p] for p in POLICIES}
        for period in entry.PERIODS} for s in SCENARIOS} for u in UNIVERSES}


def audit_exports(markets,out,paths):
    covariance={lag:entry.independent_covariances(markets,lag) for lag in (1,2)}
    result={}
    for universe in UNIVERSES:
        for scenario in SCENARIOS:
            for policy,(_,vol) in POLICIES.items():
                if universe!='all_three' and policy!=CANDIDATE:
                    continue
                stem=f'{universe}_{scenario}_{policy}'
                equity,trades=(pd.read_csv(out/f'{stem}_{kind}.csv') for kind in ('equity','trades'))
                equity.timestamp=pd.to_datetime(equity.timestamp,utc=True)
                trades.timestamp=pd.to_datetime(trades.timestamp,utc=True)
                audit=prior.audit_configured_risk(markets,equity,trades,CAPITAL,
                    entry.STRESS if scenario=='joint_stress' else entry.NOMINAL,
                    covariance[2 if scenario=='joint_stress' else 1],vol)
                row=paths[universe,scenario,'full_continuous',policy]
                pairs=((equity.equity.iloc[-1],row['final_equity']),
                    (equity.intraday_drawdown_bound.min(),row['max_intraday_drawdown_bound']),
                    (trades.fee.sum(),row['fees_paid_total']),(trades.slippage.sum(),row['slippage_paid_total']))
                if len(trades)!=row['trades_count'] or not all(np.isclose(a,b,rtol=1e-12,atol=1e-8) for a,b in pairs):
                    raise ValueError('Exported ledger differs from reported economics')
                if universe!='all_three':
                    excluded=universe.removeprefix('without_')
                    if equity['base_'+excluded].ne(0).any() or trades.symbol.eq(excluded).any():
                        raise ValueError('Omitted asset was traded or held')
                calendar=calendar_diagnostics(equity,CAPITAL)
                attribution=attribute_spot_pnl(markets,equity,trades,CAPITAL)
                pd.DataFrame(calendar['months']).to_csv(out/f'{stem}_months.csv',index=False)
                pd.DataFrame.from_dict(attribution['assets'],orient='index').to_csv(out/f'{stem}_attribution.csv',index_label='symbol')
                result[stem]={'ledger':audit,'calendar':calendar,'attribution':attribution}
    return result


def research(data,out,jobs=4):
    markets,manifest=entry.load_spot_flow(data)
    proof=read_freeze(out,manifest)
    if (out/'frozen_selection.json').exists():
        raise ValueError('Prior payoffs exist; preserve the fixed selection')
    archives=entry.verify_archive_cache(data,manifest)
    old=json.loads(gzip.decompress(PRIOR.read_bytes()))
    old_proof=old['frozen_protocol']
    old_body={k:v for k,v in old_proof.items() if k!='protocol_sha256'}
    if (hashlib.sha256(entry.canonical(old_body)).hexdigest()!=old_proof['protocol_sha256']
            or old_proof['source_sha256']!=prior.fingerprints() or old_proof['runtime']!=entry.runtime()):
        raise ValueError('Preceding protocol/source/runtime identity changed')
    if manifest['datasets']!=old['frozen_protocol']['source_manifest']['datasets']:
        raise ValueError('Source CSV/archive identities differ from preceding study')
    archive_sha=verify_prior_archive()
    all_targets=entry.completed_entry_targets(markets)
    targets={u:{s:omission_targets(all_targets[s],u) for s in ('guarded_edge','entry_hysteresis')} for u in UNIVERSES}
    for universe,signals in targets.items():
        for name,target in signals.items():
            target.to_csv(out/f'targets_{universe}_{name}.csv',index_label='bar_open')
    covariance=entry.completed_risk_covariances(markets,entry.RISK)
    initialize(markets,targets,covariance,out)
    pool=ProcessPoolExecutor(max_workers=jobs,initializer=initialize,
        initargs=(markets,targets,covariance,out)) if jobs>1 else None
    try:
        dev,dev_extra=evaluate([(u,s,'development',p) for u in UNIVERSES for s in SCENARIOS for p in POLICIES],pool)
        selection={'protocol_sha256':proof['protocol_sha256'],'candidate':CANDIDATE,
            'frozen_at':datetime.now(timezone.utc).isoformat(),
            'criterion':'Candidate fixed before payoffs; no risk or asset-subset reselection',
            'development':{u:{s:{p:dev[u,s,'development',p] for p in POLICIES} for s in SCENARIOS} for u in UNIVERSES}}
        with (out/'frozen_selection.json').open('x') as f:
            f.write(json.dumps(selection,indent=2,allow_nan=False)+'\n')
        later,later_extra=evaluate([(u,s,period,p) for u in UNIVERSES for s in SCENARIOS
            for period in ('later_continuous','full_continuous') for p in POLICIES],pool)
    finally:
        if pool:
            pool.shutdown()
    paths=dev|later
    checked=0
    for (u,s,period,p),row in paths.items():
        if u=='all_three':
            entry.compare_control(row,old['results'][str(CAPITAL)][s][period][p],f'{u}/{s}/{period}/{p}')
            checked+=1
    omissions=UNIVERSES[1:]
    gates={'accounting':all(r['accounting']['reconciled'] for r in paths.values()),
        'original_control_reproduction':checked==24,
        'candidate_omission_drawdowns_within_budget':all(
            paths[u,s,p,CANDIDATE]['max_intraday_drawdown_bound']>=-.20
            for u in omissions for s in SCENARIOS for p in entry.PERIODS),
        'candidate_omission_later_full_positive':all(paths[u,s,p,CANDIDATE]['net_pnl']>0
            for u in omissions for s in SCENARIOS for p in ('later_continuous','full_continuous'))}
    report={'frozen_protocol':proof,'frozen_selection':selection,'verified_archives':archives,
        'verified_preceding_archive_sha256':archive_sha,'preceding_control_checks':checked,
        'reconciled_paths':len(paths),'results':nest(paths),'diagnostics':nest(dev_extra|later_extra),
        'omission_gates':gates,'historical_omission_screen_passed':bool(all(gates.values())),
        'audited_accounts':audit_exports(markets,out,paths),'live_eligible':False,
        'limitations':['Adaptive diagnostic on reused dates and three surviving assets; no new holdout or confirmatory significance.',
            'Rolling windows overlap; observed negative-window fractions are not future loss probabilities.',
            'Excluding the best three months is ex-post arithmetic, not an executable timing strategy.',
            'Additive asset contributions describe this funded shared-cash path, not stand-alone asset strategies.',
            'Omissions use only 10,000 USDT and nominal/joint scenarios; smaller accounts and isolated stresses are untested.',
            'Model fills, fixed fees, terminal marks, account/EEA execution, taxes, operating costs, custody and quote FX retain prior limitations.',
            'Historical drawdown/annual-volatility requests do not cap future losses; no post-result policy or universe reselection.']}
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}
    (out/'csv_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    return report


def export_report(report,out,destination):
    destination.mkdir(parents=True,exist_ok=True)
    stem='THETABOT_ROBUSTNESS_2026-10-25'
    (destination/(stem+'.json.gz')).write_bytes(gzip.compress(entry.canonical(report),compresslevel=9,mtime=0))
    proof=report['frozen_protocol']
    (destination/'THETABOT_ROBUSTNESS_PROTOCOL_2026-10-25.json').write_text(json.dumps(proof,indent=2)+'\n')
    lines=['# Fixed calendar and asset-omission audit — 2026-10-25','',
        '**Adaptive diagnostic on reused history; no policy or universe selection.**','',
        f"Fixed `entry_20` omission screen: **{report['historical_omission_screen_passed']}**.",'',
        'Each account starts with 10,000 USDT. Full: January 2022–September 2026;',
        'later: independently funded January 2025–September 2026. Net includes modeled',
        'trading costs, excludes operating costs/tax, and marks terminal inventory.','',
        '## Fixed candidate with one asset omitted','',
        '| Universe | Full nominal | Full joint | Worst full DD | Later nominal | Later joint |',
        '|---|---:|---:|---:|---:|---:|']
    for u in UNIVERSES:
        rows=report['results'][u]
        nominal=rows['nominal']['full_continuous'][CANDIDATE]
        joint=rows['joint_stress']['full_continuous'][CANDIDATE]
        dd=min(rows[s]['full_continuous'][CANDIDATE]['max_intraday_drawdown_bound'] for s in SCENARIOS)
        lines += [f"| {u} | {nominal['total_return']:+.2%} | {joint['total_return']:+.2%} | {dd:.2%} | {rows['nominal']['later_continuous'][CANDIDATE]['total_return']:+.2%} | {rows['joint_stress']['later_continuous'][CANDIDATE]['total_return']:+.2%} |"]
    lines += ['', 'Eligible retained assets each receive up to the existing 25% cap before risk scaling;',
        'this reallocates the remaining budget and is not subtraction of historical asset P&L.','',
        '## Rolling calendar years in the original universe','',
        '| Policy | Scenario | Worst 12 months | Negative 12-month windows | Worst month | Best-3 positive log-return share | Return excluding best 3 months* |',
        '|---|---|---:|---:|---:|---:|---:|']
    for p in POLICIES:
        for s in SCENARIOS:
            cal=report['audited_accounts'][f'all_three_{s}_{p}']['calendar']
            rolling=cal['rolling_months']['12']
            share=cal['best_three_share_of_positive_log_return']
            share_text='n/a' if share is None else f'{share:.2%}'
            lines += [f"| {p} | {s} | {rolling['worst_return']:.2%} | {rolling['negative_count']}/{rolling['count']} | {cal['worst_month']['return']:.2%} | {share_text} | {cal['mathematical_return_excluding_best_three_months']:+.2%} |"]
    lines += ['', '*Ex-post mathematical concentration diagnostic, not a tradable monthly switch.',
        'Windows overlap and are not independent observations or future loss probabilities.','',
        '## Nominal additive asset P&L attribution (USDT)','',
        '| Universe | Policy | BTC net contribution | ETH net contribution | BNB net contribution | Total net |',
        '|---|---|---:|---:|---:|---:|']
    for u in UNIVERSES:
        for p in POLICIES if u=='all_three' else (CANDIDATE,):
            a=report['audited_accounts'][f'{u}_nominal_{p}']['attribution']
            values=[a['assets'][s]['net_pnl_contribution'] for s in SYMBOLS]
            lines += [f"| {u} | {p} | {values[0]:+.2f} | {values[1]:+.2f} | {values[2]:+.2f} | {a['total_net_pnl']:+.2f} |"]
    audits=report['audited_accounts']
    lines += ['', 'Attribution includes holding gains/losses, fees and adverse execution impact.',
        'These are contributions to the shared-cash account, not stand-alone asset returns.','',
        '## Fixed gates and verification','',*[f'- {k}: **{v}**' for k,v in report['omission_gates'].items()],
        f"- {report['verified_archives']} official archives rechecked; {report['preceding_control_checks']} preceding summaries reproduce.",
        f"- {report['reconciled_paths']} funded paths reconcile; {len(audits)} complete CSV accounts independently audited.",
        f"- Audit covers {sum(a['ledger']['bars'] for a in audits.values()):,} bars and {sum(a['ledger']['fills'] for a in audits.values()):,} fills.",
        f"- Maximum attribution bar residual: {max(a['attribution']['max_bar_residual'] for a in audits.values()):.3g} USDT.",
        f"- Protocol SHA-256: `{proof['protocol_sha256']}`.",'',
        '## Limits','',*['- '+s for s in report['limitations']],'',
        'Live eligibility remains false. [Frozen plan](ROBUSTNESS_RESEARCH_PLAN.md).','',
        '## Reproduce','', '```bash','python -m scripts.download_spot_flow',
        'python -m scripts.research_robustness freeze',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_robustness run --jobs 4','```','']
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
    parser.add_argument('--out',type=Path,default=Path('artifacts/robustness_20261025'))
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
        print(json.dumps({'omission_gates':report['omission_gates'],'live_eligible':False},indent=2))


if __name__=='__main__':
    main()
