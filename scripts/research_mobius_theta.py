"""Frozen historical Möbius/theta forecast and funded-spot comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

from scripts.audit_cost_risk import audit_ledger, independent_covariances
from scripts.research_cost_risk import verify_archive_cache, compare_control
from scripts.research_spot_flow import load_spot_flow, NOMINAL, STRESS
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.execution_risk import completed_risk_covariances
from spot_bot.portfolio.momentum_refinement import momentum_refinement_targets
from spot_bot.portfolio.risk import PortfolioRisk
from theta_bot_averaging.mobius_research import (
    FAMILIES, WINDOW, HARMONICS, Q, HORIZON, LOG_HURDLE,
    rolling_forecast, predictive_metrics, buy_mask,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'docs/evaluation/MOBIUS_THETA_RESEARCH_PLAN.md'
PERIODS = {'development': ('2022-01-01', '2025-01-01'),
           'later_continuous': ('2025-01-01', '2026-10-01'),
           'full_continuous': ('2022-01-01', '2026-10-01')}
SCENARIOS = ('nominal', 'cost_stress', 'delayed', 'joint_stress')
POLICIES = {'guarded_edge': None, 'theta_full_gate': 'theta_full', 'mobius_gate': 'mobius',
            'fundamental_gate': 'fundamental', 'mean_gate': 'mean'}
ALIASES = {'fourier_full_gate': 'theta_full_gate', 'fourier_squarefree_gate': 'mobius_gate'}
SETTINGS = {'window': WINDOW, 'harmonics': HARMONICS, 'q': Q, 'horizon': HORIZON,
            'log_hurdle': LOG_HURDLE, 'candidate': 'mobius_gate', 'capitals': [1000, 10000],
            'periods': PERIODS, 'scenarios': SCENARIOS, 'policies': POLICIES, 'aliases': ALIASES}
_WORKER = None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fingerprints():
    paths = sorted((ROOT / 'spot_bot').rglob('*.py'))
    paths += [ROOT / p for p in ('theta_bot_averaging/mobius_research.py',
        'scripts/research_mobius_theta.py', 'scripts/research_spot_flow.py',
        'scripts/research_cost_risk.py', 'scripts/audit_cost_risk.py',
        'scripts/download_spot_flow.py', 'scripts/download_binance_archive.py',
        'requirements.txt', 'docs/evaluation/MOBIUS_THETA_RESEARCH_PLAN.md')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def runtime():
    return {'python': '.'.join(map(str, sys.version_info[:3])),
            **{p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy')}}


def freeze(manifest, out):
    out.mkdir(parents=True, exist_ok=True)
    body = {'schema': 1, 'frozen_at': datetime.now(timezone.utc).isoformat(),
            'settings': json.loads(canonical(SETTINGS)), 'source_sha256': fingerprints(),
            'runtime': runtime(), 'source_manifest': manifest,
            'dates_previously_studied': True, 'live_eligible': False}
    proof = {**body, 'protocol_sha256': hashlib.sha256(canonical(body)).hexdigest()}
    with (out / 'frozen_protocol.json').open('x') as f:
        f.write(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    return proof


def read_freeze(out, manifest):
    proof = json.loads((out / 'frozen_protocol.json').read_text())
    body = {k: v for k, v in proof.items() if k != 'protocol_sha256'}
    if (proof['protocol_sha256'] != hashlib.sha256(canonical(body)).hexdigest()
            or canonical(proof['settings']) != canonical(SETTINGS)
            or proof['source_sha256'] != fingerprints() or proof['runtime'] != runtime()
            or proof['source_manifest'] != manifest or proof['live_eligible'] is not False
            or proof['dates_previously_studied'] is not True):
        raise ValueError('Frozen protocol, source, runtime or input changed')
    return proof


def initialize(markets, target, covariance, masks, out):
    global _WORKER
    _WORKER = markets, target, covariance, masks, out


def replay(task):
    capital, scenario, period, policy = task
    markets, target, covariance, masks, out = _WORKER
    start, end = (pd.Timestamp(x, tz='UTC') for x in PERIODS[period])
    eq, trades, summary = run_intraday_spot(markets, target,
        start=start, end=end, initial_usdt=capital, daily_control=True,
        signal_delay_bars=int(scenario in ('delayed', 'joint_stress')),
        rebalance_destination='band_edge', risk=PortfolioRisk(volatility_target=.15),
        completed_covariances=covariance, completed_buy_mask=masks.get(policy),
        **(STRESS if scenario in ('cost_stress', 'joint_stress') else NOMINAL))
    summary['accounting'] = reconcile_spot_path(markets, eq, trades, capital)
    if capital == 10000 and period == 'full_continuous' and scenario in ('nominal', 'joint_stress'):
        stem = f'{capital}_{scenario}_{policy}'
        eq.to_csv(out / f'{stem}_equity.csv', index=False)
        trades.to_csv(out / f'{stem}_trades.csv', index=False)
    return task, summary


def evaluate(tasks, executor):
    result = {}
    iterator = executor.map(replay, tasks) if executor else map(replay, tasks)
    for task, summary in iterator:
        result[task] = summary
        print(f"{task}: {summary['total_return']:+.2%}; DD {summary['max_intraday_drawdown_bound']:.2%}", flush=True)
    return result


def audit_exports(markets, out):
    covariance = {lag: independent_covariances(markets, lag) for lag in (1, 2)}
    results = {}
    for scenario in ('nominal', 'joint_stress'):
        costs = STRESS if scenario == 'joint_stress' else NOMINAL
        for policy in POLICIES:
            stem = f'10000_{scenario}_{policy}'
            eq = pd.read_csv(out / f'{stem}_equity.csv')
            trades = pd.read_csv(out / f'{stem}_trades.csv')
            eq.timestamp = pd.to_datetime(eq.timestamp, utc=True)
            trades.timestamp = pd.to_datetime(trades.timestamp, utc=True)
            results[stem] = audit_ledger(markets, eq, trades, 10000, costs,
                                        covariance[2 if scenario == 'joint_stress' else 1])
    return results


def research(data, out, jobs=4):
    markets, manifest = load_spot_flow(data)
    proof = read_freeze(out, manifest)
    if (out / 'frozen_selection.json').exists():
        raise ValueError('Prior payoff run exists; preserve it instead of overwriting selection')
    archive_count = verify_archive_cache(data, manifest)
    archived = json.loads(gzip.decompress((ROOT / 'docs/evaluation/THETABOT_COST_RISK_2026-10-13.json.gz').read_bytes()))
    for symbol, entry in manifest['datasets'].items():
        old = archived['source_manifest']['datasets'][symbol]
        if [(a['month'],a['sha256'],a['rows']) for a in entry['archives']] != [
                (a['month'],a['sha256'],a['rows']) for a in old['archives']]:
            raise ValueError('Official archive history differs from the preceding study')
    predictions = {f: pd.DataFrame({s: rolling_forecast(m.close, f) for s, m in markets.items()})
                   for f in FAMILIES}
    masks = {p: buy_mask(predictions[f]) for p, f in POLICIES.items() if f is not None}
    equivalence = {}
    for left, right in (('theta_full', 'fourier_full'), ('mobius', 'fourier_squarefree')):
        a, b = predictions[left], predictions[right]
        error = float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy())))
        equal_masks = buy_mask(a).equals(buy_mask(b))
        if error > 1e-10 or not equal_masks:
            raise ValueError('Mathematical control predictions or trade masks disagree')
        equivalence[left + '_vs_' + right] = {'maximum_forecast_error': error, 'identical_buy_masks': equal_masks}
    for s, market in markets.items():
        table = pd.DataFrame({f: p[s] for f, p in predictions.items()})
        table['future_log_return'] = np.log(market.close.shift(-HORIZON) / market.close)
        table.index.name = 'bar_open'
        table.to_csv(out / f'predictions_{s}.csv')
    score_periods = {**PERIODS, '2025': ('2025-01-01', '2026-01-01'),
                     '2026': ('2026-01-01', '2026-10-01')}
    target = momentum_refinement_targets(markets)[0]['capped_allocation']
    covariance = completed_risk_covariances(markets, PortfolioRisk(volatility_target=.15))
    initialize(markets, target, covariance, masks, out)
    pool = ProcessPoolExecutor(max_workers=jobs, initializer=initialize,
        initargs=(markets, target, covariance, masks, out)) if jobs > 1 else None
    try:
        development = evaluate([(c, s, 'development', p) for c in (1000, 10000)
                                for s in SCENARIOS for p in POLICIES], pool)
        selection = {'protocol_sha256': proof['protocol_sha256'], 'candidate': 'mobius_gate',
            'frozen_at': datetime.now(timezone.utc).isoformat(),
            'criterion': 'Candidate fixed before all payoffs; no reselection',
            'development': {str(c): {s: {p: development[c, s, 'development', p] for p in POLICIES}
                           for s in SCENARIOS} for c in (1000, 10000)}}
        with (out / 'frozen_selection.json').open('x') as f:
            f.write(json.dumps(selection, indent=2, allow_nan=False) + '\n')
        remaining = evaluate([(c, s, period, p) for c in (1000, 10000) for s in SCENARIOS
                              for period in ('later_continuous', 'full_continuous') for p in POLICIES], pool)
    finally:
        if pool:
            pool.shutdown()
    paths = development | remaining
    predictive = {period: {family: {s: predictive_metrics(m.close, predictions[family][s],
                    pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC'))
                    for s, m in markets.items()} for family in FAMILIES}
                    for period, (start, end) in score_periods.items()}
    checked = 0
    for (c, s, period, policy), summary in paths.items():
        if policy == 'guarded_edge':
            compare_control(summary, archived['results'][str(c)][s][period]['guarded_edge'], f'{c}/{s}/{period}')
            checked += 1
    results = {str(c): {s: {period: {p: paths[c, s, period, p] for p in POLICIES}
               for period in PERIODS} for s in SCENARIOS} for c in (1000, 10000)}
    for scenarios in results.values():
        for periods in scenarios.values():
            for policies in periods.values():
                policies.update({alias: policies[source] for alias, source in ALIASES.items()})
    def mse(family):
        rows = predictive['later_continuous'][family].values()
        return sum(r['mse'] * r['samples'] for r in rows) / sum(r['samples'] for r in rows)
    gates = {'accounting': all(r['accounting']['reconciled'] for r in paths.values()),
        'lower_later_mse_than_theta_and_mean': all(mse('mobius') < mse(f) for f in ('theta_full', 'mean')),
        'candidate_drawdowns_within_budget': all(paths[c,s,p,'mobius_gate']['max_intraday_drawdown_bound'] >= -.20
             for c in (1000,10000) for s in SCENARIOS for p in PERIODS),
        'candidate_later_full_positive_all_stresses': all(paths[c,s,p,'mobius_gate']['net_pnl'] > 0
             for c in (1000,10000) for s in SCENARIOS for p in ('later_continuous','full_continuous')),
        'candidate_beats_control_and_theta_nominal_joint': all(paths[c,s,p,'mobius_gate']['net_pnl'] > paths[c,s,p,b]['net_pnl']
             for c in (1000,10000) for s in ('nominal','joint_stress')
             for p in ('later_continuous','full_continuous') for b in ('guarded_edge','theta_full_gate'))}
    report = {'frozen_protocol': proof, 'frozen_selection': selection,
        'verified_archives': archive_count, 'archived_control_checks': checked,
        'distinct_reconciled_paths': len(paths), 'economic_aliases': ALIASES,
        'prediction_equivalence': equivalence, 'predictive_metrics': predictive,
        'results': results, 'gates': gates, 'historical_improvement_screen_passed': bool(all(gates.values())),
        'independent_csv_audit': audit_exports(markets, out), 'live_eligible': False,
        'limitations': ['Known historical dates and surviving three-asset universe; no fresh holdout.',
            'Fixed extrapolated cosine regression, not a separately trained supervised predictor.',
            'Gate predicts 24h close returns; Monday entries and risk exits determine actual holding times.',
            'Fourier equivalence prevents a claim of special Möbius/theta predictive information.',
            'USDT research pairs do not establish executable EEA markets.',
            'Modeled opens, fees, spread and slippage; no order-book, tax, custody, operating-cost or FX model.',
            'Historical drawdown and volatility settings do not cap future loss.']}
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}
    (out / 'csv_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return report


def export_report(report, out, destination):
    destination.mkdir(parents=True, exist_ok=True)
    stem = 'THETABOT_MOBIUS_THETA_2026-10-22'
    packed = gzip.compress(canonical(report), compresslevel=9, mtime=0)
    (destination / (stem + '.json.gz')).write_bytes(packed)
    proof = report['frozen_protocol']
    (destination / 'THETABOT_MOBIUS_PROTOCOL_2026-10-22.json').write_text(json.dumps(proof, indent=2) + '\n')
    lines = ['# Theta/Möbius diagnostic — 2026-10-22', '',
        '**Previously studied history; no fresh holdout. Candidate fixed before payoffs.**', '',
        f"Historical improvement screen: **{report['historical_improvement_screen_passed']}**.", '',
        '## Funded economic results', '',
        'Each row uses one continuously funded January 2022–September 2026 account.',
        'Net includes modeled trading costs, before tax, hosting, labor and FX risk.', '']
    for capital, scenarios in report['results'].items():
        lines += [f'### Initial {capital} USDT', '',
            '| Policy | Full nominal net | Full joint net | Worst full DD | Later nominal net |',
            '|---|---:|---:|---:|---:|']
        for p in POLICIES:
            nominal = scenarios['nominal']['full_continuous'][p]['total_return']
            joint = scenarios['joint_stress']['full_continuous'][p]['total_return']
            dd = min(s['full_continuous'][p]['max_intraday_drawdown_bound'] for s in scenarios.values())
            later = scenarios['nominal']['later_continuous'][p]['total_return']
            lines += [f'| {p} | {nominal:+.2%} | {joint:+.2%} | {dd:.2%} | {later:+.2%} |']
        lines += ['']
    lines += ['## Predictive results: 2025–September 2026', '',
        'Nonoverlapping six-bar targets within each asset; assets remain dependent.', '',
        '| Family | Samples | MSE | Mean asset correlation |', '|---|---:|---:|---:|']
    for f, assets in report['predictive_metrics']['later_continuous'].items():
        count = sum(r['samples'] for r in assets.values())
        mse = sum(r['samples']*r['mse'] for r in assets.values())/count
        corr = np.mean([r['correlation'] for r in assets.values()])
        lines += [f'| {f} | {count} | {mse:.8f} | {corr:+.5f} |']
    lines += ['', '## Controls and evidence', '',
        'Full Fourier reuses theta economics; squarefree Fourier reuses Möbius economics only after',
        'independent forecasts and all buy masks agree. These are exact structural controls, not independent wins.', '',
        f"- {report['verified_archives']} official archive checksums rechecked.",
        f"- {report['archived_control_checks']} old guarded-edge control paths reproduce.",
        f"- {report['distinct_reconciled_paths']} distinct paths reconcile cash, inventory and NAV.",
        f"- {len(report['independent_csv_audit'])} exported full accounts independently audited from CSV.",
        f"- Protocol SHA-256: `{proof['protocol_sha256']}`.", '',
        '## Fixed gates', '', *[f'- {k}: **{v}**' for k,v in report['gates'].items()], '',
        '## Limitations', '', *['- '+s for s in report['limitations']], '',
        'Live eligibility remains false. [Frozen plan](MOBIUS_THETA_RESEARCH_PLAN.md).', '',
        '## Reproduce', '', '```bash', 'python -m scripts.download_spot_flow',
        'python -m scripts.research_mobius_theta freeze',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_mobius_theta run --jobs 4',
        '```', '']
    (destination / (stem + '.md')).write_text('\n'.join(lines))
    archive = out / 'evidence.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(out.iterdir()):
            if p.suffix in ('.csv', '.json'):
                z.write(p, p.name)
        z.write(PLAN, PLAN.name)
    payload = archive.read_bytes()
    name = stem + '_EVIDENCE.zip'
    pieces = []
    for i, start in enumerate(range(0,len(payload),5_000_000), 1):
        part = destination / f'{name}.part{i:03d}'
        part.write_bytes(payload[start:start+5_000_000])
        pieces.append({'file':part.name,'bytes':part.stat().st_size,
                       'sha256':hashlib.sha256(part.read_bytes()).hexdigest()})
    manifest = {'file':name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'parts':pieces,
                'reassemble':'Concatenate parts in listed order; verify the complete SHA-256 before unzipping.'}
    (destination / (name+'.manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze','run'))
    p.add_argument('--data',type=Path,default=Path('data/raw/spot_flow'))
    p.add_argument('--out',type=Path,default=Path('artifacts/mobius_theta_20261022'))
    p.add_argument('--report-out',type=Path,default=Path('docs/evaluation'))
    p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args()
    if args.jobs < 1:
        p.error('--jobs must be positive')
    if args.action=='freeze':
        _, manifest=load_spot_flow(args.data)
        proof=freeze(manifest,args.out)
        print(proof['protocol_sha256'])
    else:
        report=research(args.data,args.out,args.jobs)
        export_report(report,args.out,args.report_out)
        print(json.dumps({'gates':report['gates'],'live_eligible':False},indent=2))


if __name__=='__main__':
    main()
