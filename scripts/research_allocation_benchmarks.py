"""Frozen historical comparison of guarded momentum and simple allocations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
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
from scripts.research_cost_risk import compare_control, verify_archive_cache
from scripts.research_spot_flow import load_spot_flow, NOMINAL, STRESS
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.execution_risk import completed_risk_covariances
from spot_bot.portfolio.momentum_refinement import momentum_refinement_targets
from spot_bot.portfolio.risk import PortfolioRisk
from theta_bot_averaging.allocation_benchmarks import (
    REFERENCES, allocation_targets, realized_daily_volatility,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'docs/evaluation/ALLOCATION_BENCHMARK_RESEARCH_PLAN.md'
CANDIDATE = 'guarded_edge'
POLICIES = (CANDIDATE, *REFERENCES)
INVESTED = tuple(p for p in REFERENCES if p != 'cash')
CAPITALS = (1000, 10000)
PERIODS = {'development': ('2022-01-01', '2025-01-01'),
           'later_continuous': ('2025-01-01', '2026-10-01'),
           'full_continuous': ('2022-01-01', '2026-10-01')}
SCENARIOS = ('nominal', 'cost_stress', 'delayed', 'joint_stress')
RISK = PortfolioRisk(volatility_target=.15)
SETTINGS = {'candidate': CANDIDATE, 'policies': POLICIES, 'capitals': CAPITALS,
    'periods': PERIODS, 'scenarios': SCENARIOS, 'risk': asdict(RISK),
    'costs': {'nominal': NOMINAL, 'stress': STRESS}, 'rebalance_destination': 'band_edge',
    'daily_control': True, 'minimum_notional': 10, 'target_total_cap': .5, 'target_asset_cap': .25}
_WORKER = None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fingerprints():
    paths = sorted((ROOT / 'spot_bot').rglob('*.py'))
    paths += [ROOT / p for p in ('theta_bot_averaging/allocation_benchmarks.py',
        'scripts/research_allocation_benchmarks.py', 'scripts/research_spot_flow.py',
        'scripts/research_cost_risk.py', 'scripts/audit_cost_risk.py',
        'scripts/download_spot_flow.py', 'scripts/download_binance_archive.py',
        'requirements.txt', 'docs/evaluation/ALLOCATION_BENCHMARK_RESEARCH_PLAN.md')]
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


def initialize(markets, targets, covariance, out):
    global _WORKER
    _WORKER = markets, targets, covariance, out


def replay(task):
    capital, scenario, period, policy = task
    markets, targets, covariance, out = _WORKER
    start, end = (pd.Timestamp(x, tz='UTC') for x in PERIODS[period])
    equity, trades, summary = run_intraday_spot(markets, targets[policy],
        start=start, end=end, initial_usdt=capital, daily_control=True,
        signal_delay_bars=int(scenario in ('delayed', 'joint_stress')),
        rebalance_destination='band_edge', risk=RISK, completed_covariances=covariance,
        **(STRESS if scenario in ('cost_stress', 'joint_stress') else NOMINAL))
    summary['accounting'] = reconcile_spot_path(markets, equity, trades, capital)
    diagnostics = {'realized_daily_annual_volatility': realized_daily_volatility(equity, capital)}
    if capital == 10000 and period == 'full_continuous' and scenario in ('nominal', 'joint_stress'):
        stem = f'{capital}_{scenario}_{policy}'
        equity.to_csv(out / f'{stem}_equity.csv', index=False)
        trades.to_csv(out / f'{stem}_trades.csv', index=False)
    return task, summary, diagnostics


def evaluate(tasks, executor):
    results, diagnostics = {}, {}
    for task, row, extra in executor.map(replay, tasks) if executor else map(replay, tasks):
        results[task], diagnostics[task] = row, extra
        print(f"{task}: {row['total_return']:+.2%}; DD {row['max_intraday_drawdown_bound']:.2%}", flush=True)
    return results, diagnostics


def audit_exports(markets, out, paths):
    covariance = {lag: independent_covariances(markets, lag) for lag in (1, 2)}
    results = {}
    for scenario in ('nominal', 'joint_stress'):
        costs = STRESS if scenario == 'joint_stress' else NOMINAL
        for policy in POLICIES:
            stem = f'10000_{scenario}_{policy}'
            equity = pd.read_csv(out / f'{stem}_equity.csv')
            trades = pd.read_csv(out / f'{stem}_trades.csv')
            equity.timestamp = pd.to_datetime(equity.timestamp, utc=True)
            trades.timestamp = pd.to_datetime(trades.timestamp, utc=True)
            results[stem] = audit_ledger(markets, equity, trades, 10000, costs,
                covariance[2 if scenario == 'joint_stress' else 1])
            row = paths[10000, scenario, 'full_continuous', policy]
            pairs = ((equity.equity.iloc[-1], row['final_equity']),
                (equity.intraday_drawdown_bound.min(), row['max_intraday_drawdown_bound']),
                (trades.fee.sum(), row['fees_paid_total']),
                (trades.slippage.sum(), row['slippage_paid_total']))
            if len(trades) != row['trades_count'] or not all(
                    np.isclose(a, b, rtol=1e-12, atol=1e-8) for a, b in pairs):
                raise ValueError('Exported account differs from reported economics')
    return results


def nest(paths):
    return {str(c): {s: {period: {p: paths[c, s, period, p] for p in POLICIES}
            for period in PERIODS} for s in SCENARIOS} for c in CAPITALS}


def comparison_headroom(paths):
    result = {}
    for c in CAPITALS:
        result[str(c)] = {}
        for scenario in ('nominal', 'joint_stress'):
            periods = result[str(c)][scenario] = {}
            for period in ('later_continuous', 'full_continuous'):
                start, end = (pd.Timestamp(t, tz='UTC') for t in PERIODS[period])
                months = (end - start) / pd.Timedelta(days=30.4375)
                periods[period] = {}
                for reference in INVESTED:
                    surplus = (paths[c, scenario, period, CANDIDATE]['final_equity']
                               - paths[c, scenario, period, reference]['final_equity'])
                    periods[period][reference] = {'terminal_surplus_usdt': surplus,
                        'period_calendar_months': float(months),
                        'incremental_cost_headroom_usdt_per_month': surplus / months}
    return result


def research(data, out, jobs=4):
    markets, manifest = load_spot_flow(data)
    proof = read_freeze(out, manifest)
    if (out / 'frozen_selection.json').exists():
        raise ValueError('Prior payoff run exists; preserve it instead of overwriting selection')
    archive_count = verify_archive_cache(data, manifest)
    archived = json.loads(gzip.decompress((ROOT / 'docs/evaluation/THETABOT_COST_RISK_2026-10-13.json.gz').read_bytes()))
    for symbol, entry in manifest['datasets'].items():
        old = archived['source_manifest']['datasets'][symbol]
        if [(a['month'], a['sha256'], a['rows']) for a in entry['archives']] != [
                (a['month'], a['sha256'], a['rows']) for a in old['archives']]:
            raise ValueError('Official archive history differs from the preceding study')
    targets = {CANDIDATE: momentum_refinement_targets(markets)[0]['capped_allocation'],
               **allocation_targets(markets)}
    for policy, target in targets.items():
        target.to_csv(out / f'targets_{policy}.csv', index_label='bar_open')
    covariance = completed_risk_covariances(markets, RISK)
    initialize(markets, targets, covariance, out)
    pool = ProcessPoolExecutor(max_workers=jobs, initializer=initialize,
        initargs=(markets, targets, covariance, out)) if jobs > 1 else None
    try:
        development, dev_extra = evaluate([(c, s, 'development', p) for c in CAPITALS
            for s in SCENARIOS for p in POLICIES], pool)
        selection = {'protocol_sha256': proof['protocol_sha256'], 'candidate': CANDIDATE,
            'frozen_at': datetime.now(timezone.utc).isoformat(),
            'criterion': 'Existing candidate fixed before all payoffs; no reselection',
            'development': {str(c): {s: {p: development[c, s, 'development', p] for p in POLICIES}
                            for s in SCENARIOS} for c in CAPITALS}}
        with (out / 'frozen_selection.json').open('x') as f:
            f.write(json.dumps(selection, indent=2, allow_nan=False) + '\n')
        later, later_extra = evaluate([(c, s, period, p) for c in CAPITALS for s in SCENARIOS
            for period in ('later_continuous', 'full_continuous') for p in POLICIES], pool)
    finally:
        if pool:
            pool.shutdown()
    paths = development | later
    checked = 0
    for (c, s, period, policy), row in paths.items():
        if policy == CANDIDATE:
            compare_control(row, archived['results'][str(c)][s][period][CANDIDATE], f'{c}/{s}/{period}')
            checked += 1
    gates = {'accounting': all(r['accounting']['reconciled'] for r in paths.values()),
        'candidate_all_drawdowns_within_budget': all(paths[c,s,p,CANDIDATE]['max_intraday_drawdown_bound'] >= -.20
            for c in CAPITALS for s in SCENARIOS for p in PERIODS),
        'candidate_later_full_positive_all_stresses': all(paths[c,s,p,CANDIDATE]['net_pnl'] > 0
            for c in CAPITALS for s in SCENARIOS for p in ('later_continuous','full_continuous')),
        'candidate_beats_each_invested_reference_later_full_nominal_joint': all(
            paths[c,s,p,CANDIDATE]['net_pnl'] > paths[c,s,p,b]['net_pnl'] for c in CAPITALS
            for s in ('nominal','joint_stress') for p in ('later_continuous','full_continuous') for b in INVESTED)}
    report = {'frozen_protocol': proof, 'frozen_selection': selection,
        'verified_archives': archive_count, 'archived_control_checks': checked,
        'reconciled_paths': len(paths), 'results': nest(paths), 'diagnostics': nest(dev_extra | later_extra),
        'incremental_cost_headroom': comparison_headroom(paths), 'gates': gates,
        'historical_incremental_value_screen_passed': bool(all(gates.values())),
        'independent_csv_audit': audit_exports(markets, out, paths), 'live_eligible': False,
        'limitations': ['Previously studied dates and surviving three-asset universe; no fresh holdout.',
            'Shared guard settings do not exactly match realized volatility, exposure or drawdown.',
            'References rebalance weekly with risk sales; they are not literal buy-and-hold.',
            'Terminal surplus per month is descriptive incremental cost headroom, not cash withdrawals or forecast income.',
            'No order-book, executable paper fills, tax, operating-cost, custody or quote-FX model.',
            'USDT archives do not establish executable EEA/account markets.',
            'Terminal inventory is marked without forced liquidation; historical risk settings do not cap future loss.']}
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}
    (out / 'csv_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return report


def export_report(report, out, destination):
    destination.mkdir(parents=True, exist_ok=True)
    stem = 'THETABOT_ALLOCATION_BENCHMARKS_2026-10-23'
    (destination / (stem + '.json.gz')).write_bytes(gzip.compress(canonical(report), compresslevel=9, mtime=0))
    proof = report['frozen_protocol']
    (destination / 'THETABOT_ALLOCATION_PROTOCOL_2026-10-23.json').write_text(json.dumps(proof, indent=2) + '\n')
    lines = ['# Guarded momentum versus fixed allocations — 2026-10-23', '',
        '**Known history; fixed existing candidate, no fresh holdout or signal search.**', '',
        f"Historical incremental-value screen: **{report['historical_incremental_value_screen_passed']}**.", '',
        'Same execution and requested volatility guard; realized risk differs.',
        'Equal/inverse-volatility references rebalance weekly and sell for risk; they are not buy-and-hold.',
        'Full: January 2022–September 2026. Later: independently funded January 2025–September 2026.',
        'Net includes modeled fills, fees, spread and slippage, before tax and operating costs.', '']
    for capital, scenarios in report['results'].items():
        lines += [f'## Initial {capital} USDT', '',
            '| Policy | Full nominal net | Full joint net | Worst full DD | Nominal realized vol | Mean exposure | Fees + impact |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for policy in POLICIES:
            row = scenarios['nominal']['full_continuous'][policy]
            joint = scenarios['joint_stress']['full_continuous'][policy]['total_return']
            dd = min(s['full_continuous'][policy]['max_intraday_drawdown_bound'] for s in scenarios.values())
            vol = report['diagnostics'][capital]['nominal']['full_continuous'][policy]['realized_daily_annual_volatility']
            lines += [f"| {policy} | {row['total_return']:+.2%} | {joint:+.2%} | {dd:.2%} | {vol:.2%} | {row['mean_exposure']:.2%} | {row['fees_paid_total'] + row['slippage_paid_total']:.2f} |"]
        lines += ['', '| Policy | Later nominal net | Later joint net | Worst later DD | Later nominal realized vol |',
            '|---|---:|---:|---:|---:|']
        for policy in POLICIES:
            row = scenarios['nominal']['later_continuous'][policy]
            joint = scenarios['joint_stress']['later_continuous'][policy]['total_return']
            dd = min(s['later_continuous'][policy]['max_intraday_drawdown_bound'] for s in scenarios.values())
            vol = report['diagnostics'][capital]['nominal']['later_continuous'][policy]['realized_daily_annual_volatility']
            lines += [f"| {policy} | {row['total_return']:+.2%} | {joint:+.2%} | {dd:.2%} | {vol:.2%} |"]
        lines += ['', 'Realized vol is annualized population daily NAV-return standard deviation, including the initial funded return.', '']
    lines += ['## Descriptive incremental operating-cost headroom', '',
        'Candidate terminal NAV minus each reference NAV, divided by period calendar months.',
        '10,000-USDT accounts; USDT per month. Negative means the candidate already trails.',
        'Only additional candidate expenses relative to the reference belong here. No recurring withdrawals are replayed.', '',
        '| Reference | Full nominal | Full joint | Later nominal | Later joint |', '|---|---:|---:|---:|---:|']
    headroom = report['incremental_cost_headroom']['10000']
    for reference in INVESTED:
        values = [headroom[s][p][reference]['incremental_cost_headroom_usdt_per_month']
            for p in ('full_continuous','later_continuous') for s in ('nominal','joint_stress')]
        lines += ['| ' + reference + ' | ' + ' | '.join(f'{v:+.2f}' for v in values) + ' |']
    audit = report['independent_csv_audit']
    lines += ['', '## Evidence and fixed gates', '',
        f"- {report['verified_archives']} official monthly archives checked; {report['archived_control_checks']} archived candidate paths reproduce.",
        f"- {report['reconciled_paths']} full funded paths reconcile nonnegative cash, inventory and NAV.",
        f"- {len(audit)} CSV accounts independently audited: {sum(r['bars'] for r in audit.values()):,} bars and {sum(r['fills'] for r in audit.values()):,} fills.",
        f"- Maximum independent NAV error: {max(r['max_nav_error'] for r in audit.values()):.3g} USDT.",
        f"- Protocol SHA-256: `{proof['protocol_sha256']}`.", '',
        *[f'- {k}: **{v}**' for k,v in report['gates'].items()], '',
        '## Limits', '', *['- '+s for s in report['limitations']], '',
        'Live eligibility remains false. [Fixed plan](ALLOCATION_BENCHMARK_RESEARCH_PLAN.md).', '',
        '## Reproduce', '', '```bash', 'python -m scripts.download_spot_flow',
        'python -m scripts.research_allocation_benchmarks freeze',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_allocation_benchmarks run --jobs 4',
        '```', '']
    (destination / (stem + '.md')).write_text('\n'.join(lines))
    archive = out / 'evidence.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(out.iterdir()):
            if path.suffix in ('.csv', '.json'):
                z.write(path, path.name)
        z.write(PLAN, PLAN.name)
    payload = archive.read_bytes()
    name = stem + '_EVIDENCE.zip'
    parts = []
    for i, start in enumerate(range(0, len(payload), 5_000_000), 1):
        path = destination / f'{name}.part{i:03d}'
        path.write_bytes(payload[start:start + 5_000_000])
        parts.append({'file': path.name, 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {'file': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
        'parts': parts, 'reassemble': 'Concatenate parts in listed order; verify complete SHA-256 before unzipping.'}
    (destination / (name + '.manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','run'))
    parser.add_argument('--data', type=Path, default=Path('data/raw/spot_flow'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/allocation_benchmarks_20261023'))
    parser.add_argument('--report-out', type=Path, default=Path('docs/evaluation'))
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.action == 'freeze':
        _, manifest = load_spot_flow(args.data)
        print(freeze(manifest, args.out)['protocol_sha256'])
    else:
        report = research(args.data, args.out, args.jobs)
        export_report(report, args.out, args.report_out)
        print(json.dumps({'gates': report['gates'], 'live_eligible': False}, indent=2))


if __name__ == '__main__':
    main()
