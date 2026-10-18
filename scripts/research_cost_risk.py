"""Fixed offline turnover/risk study. Never connects to an exchange account."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts.download_spot_flow import BEGIN, END
from scripts.research_spot_flow import load_spot_flow, NOMINAL, STRESS, VALIDATION, TRANSFER
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.execution_risk import completed_risk_covariances
from spot_bot.portfolio.momentum_refinement import momentum_refinement_targets
from spot_bot.portfolio.risk import PortfolioRisk

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / 'docs/evaluation/COST_RISK_RESEARCH_PLAN.md'
POLICIES = ('spot_control', 'band_edge', 'vol_guard', 'guarded_edge')
SCENARIOS = ('nominal', 'cost_stress', 'delayed', 'joint_stress')
PERIODS = {'development': (BEGIN, VALIDATION), '2025': (VALIDATION, TRANSFER),
           '2026': (TRANSFER, END), 'later_continuous': (VALIDATION, END),
           'full_continuous': (BEGIN, END)}
CANDIDATE = 'guarded_edge'
RISK = PortfolioRisk(volatility_target=.15)
_WORKER = None


def initialize_worker(markets, target, covariance, out):
    global _WORKER
    _WORKER = markets, target, covariance, out


def replay(task):
    capital, scenario, period, policy = task
    markets, target, covariance, out = _WORKER
    guarded = policy in ('vol_guard', 'guarded_edge')
    start, end = PERIODS[period]
    eq, fills, summary = run_intraday_spot(markets, target, initial_usdt=capital,
        start=start, end=end, daily_control=True,
        signal_delay_bars=int(scenario in ('delayed', 'joint_stress')),
        rebalance_destination='band_edge' if policy in ('band_edge', 'guarded_edge') else 'target',
        risk=RISK if guarded else None, completed_covariances=covariance if guarded else None,
        **(STRESS if scenario in ('cost_stress', 'joint_stress') else NOMINAL))
    summary['accounting'] = reconcile_spot_path(markets, eq, fills, capital)
    if period == 'full_continuous':
        stem = f'{capital}_{scenario}_{policy}'
        eq.to_csv(out / f'{stem}_equity.csv', index=False)
        fills.to_csv(out / f'{stem}_trades.csv', index=False)
    return task, summary


def evaluate(tasks, executor):
    result = {}
    for task, summary in executor.map(replay, tasks) if executor else map(replay, tasks):
        result[task] = summary
        print(f"{task}: net={summary['total_return']:+.2%}; DD={summary['max_intraday_drawdown_bound']:.2%}", flush=True)
    return result


def compare_control(actual, old, context):
    if actual.keys() != old.keys():
        raise ValueError(f'Archived control fields differ: {context}')
    for key, value in old.items():
        new = actual[key]
        if isinstance(value, dict): compare_control(new, value, f'{context}/{key}')
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not np.isclose(new, value, rtol=1e-10, atol=1e-8):
                raise ValueError(f'Archived control changed: {context}/{key}: {new} vs {value}')
        elif new != value: raise ValueError(f'Archived control changed: {context}/{key}')


def verify_archive_cache(data, manifest):
    """Recheck downloaded official bytes without issuing a network request."""
    count = 0
    for dataset in manifest['datasets'].values():
        for archive in dataset['archives']:
            url = archive['url']
            payload = (data / 'cache' / hashlib.sha256(url.encode()).hexdigest()).read_bytes()
            checksum = (data / 'cache' / hashlib.sha256((url + '.CHECKSUM').encode()).hexdigest()).read_text().split()
            if (len(checksum) != 2 or checksum[1].lstrip('*') != url.rsplit('/', 1)[1]
                    or hashlib.sha256(payload).hexdigest() != archive['sha256']
                    or checksum[0].lower() != archive['sha256']):
                raise ValueError('Official archived bytes/checksum do not match source manifest')
            count += 1
    return count


def improvement_gates(by_scenario):
    candidate = {s: {p: rows[CANDIDATE] for p, rows in periods.items()}
                 for s, periods in by_scenario.items()}
    base = {s: {p: rows['spot_control'] for p, rows in periods.items()}
            for s, periods in by_scenario.items()}
    risk_gates = {
        'accounting': all(x['accounting']['reconciled'] for periods in candidate.values() for x in periods.values()),
        'all_historical_drawdowns_within_budget': all(x['max_intraday_drawdown_bound'] >= -.20
                for periods in candidate.values() for x in periods.values()),
        'later_and_full_stresses_profitable': all(candidate[s][p]['net_pnl'] > 0
                for s in SCENARIOS for p in ('2025', '2026', 'later_continuous', 'full_continuous')),
        'lower_worst_full_drawdown': min(candidate[s]['full_continuous']['max_intraday_drawdown_bound'] for s in SCENARIOS)
                > min(base[s]['full_continuous']['max_intraday_drawdown_bound'] for s in SCENARIOS),
    }
    profit_gates = {**risk_gates,
        **{f'beats_nominal_control_{p}': candidate['nominal'][p]['net_pnl'] > base['nominal'][p]['net_pnl']
           for p in ('2025', '2026', 'later_continuous', 'full_continuous')},
        'beats_full_joint_control': candidate['joint_stress']['full_continuous']['net_pnl']
                > base['joint_stress']['full_continuous']['net_pnl']}
    return {'risk_gates': risk_gates, 'profit_gates': profit_gates,
        'risk_improvement_screen_passed': bool(all(risk_gates.values())),
        'profit_improvement_screen_passed': bool(all(profit_gates.values()))}


def research(data, out, jobs=4):
    proof = json.loads((out / 'frozen_protocol.json').read_text())
    plan_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    if (proof['protocol_sha256'] != plan_hash or proof['frozen_before_variant_payoffs'] is not True
            or proof['fixed_candidate'] != CANDIDATE):
        raise ValueError('Protocol and fixed candidate must precede payoffs')
    markets, manifest = load_spot_flow(data)
    verified_archives = verify_archive_cache(data, manifest)
    target = momentum_refinement_targets(markets)[0]['capped_allocation']
    covariance = completed_risk_covariances(markets, RISK)
    initialize_worker(markets, target, covariance, out)
    pool = ProcessPoolExecutor(max_workers=jobs, initializer=initialize_worker,
        initargs=(markets, target, covariance, out)) if jobs > 1 else None
    try:
        development = evaluate([(c, s, 'development', p) for c in (1000, 10000)
                                for s in SCENARIOS for p in POLICIES], pool)
        from datetime import datetime, timezone
        selection = {'protocol_sha256': plan_hash, 'fixed_candidate': CANDIDATE,
            'frozen_at': datetime.now(timezone.utc).isoformat(), 'frozen_before_later_payoffs': True,
            'criterion': 'candidate fixed before any payoff; no later or ablation reselection',
            'development': {str(c): {s: {p: development[c, s, 'development', p] for p in POLICIES}
                                    for s in SCENARIOS} for c in (1000, 10000)}}
        (out / 'frozen_selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False) + '\n')
        later = evaluate([(c, s, period, p) for c in (1000, 10000) for s in SCENARIOS
                          for period in PERIODS if period != 'development' for p in POLICIES], pool)
    finally:
        if pool: pool.shutdown()
    paths = development | later
    archived = json.loads(gzip.decompress((REPO / 'docs/evaluation/THETABOT_BAYES_MONTE_CARLO_2026-10-12.json.gz').read_bytes()))
    archived_checks = 0
    for (capital, scenario, period, policy), actual in paths.items():
        if policy == 'spot_control':
            account = archived['results'][str(capital)]
            old = (account['selection']['development'][scenario]['spot_control'] if period == 'development'
                   else account['scenarios'][scenario][period]['spot_control'])
            compare_control(actual, old, f'{capital}/{scenario}/{period}')
            archived_checks += 1
    results = {str(c): {s: {period: {p: paths[c, s, period, p] for p in POLICIES}
                           for period in PERIODS} for s in SCENARIOS} for c in (1000, 10000)}
    report = {'protocol_sha256': plan_hash, 'frozen_protocol': proof, 'frozen_selection': selection,
        'risk_config': asdict(RISK), 'policies': POLICIES, 'scenarios': SCENARIOS,
        'costs': {'nominal': NOMINAL, 'stress': STRESS}, 'source_manifest': manifest,
        'runtime_versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy')},
        'results': results, 'gates': {c: improvement_gates(v) for c, v in results.items()},
        'archived_control_checks': archived_checks, 'verified_archives': verified_archives,
        'reconciled_paths': len(paths),
        'maximum_nav_error': max(x['accounting']['max_equity_error'] for x in paths.values()),
        'dates_were_previously_studied': True, 'options_enabled': False, 'live_eligible': False,
        'limitations': ['Known dates and surviving three-asset universe, not independent alpha evidence.',
            'Risk uses lagged completed daily covariance and can only sell on modeled open fills.',
            'Signal/covariance lag stress is not a model of delayed emergency-order execution.',
            'Minimum notionals and fill costs can leave actual volatility above its requested target.',
            '20% historical drawdown and 15% estimated volatility are not future loss guarantees.',
            'No tick path, historical order-book depth, taxes, custody/default risk or USDT FX risk.',
            'Terminal inventory is marked, not forcibly liquidated; costs use fixed hypothetical rates.']}
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.csv'))}
    (out / 'csv_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return report


def write_report(report, out, destination):
    destination.mkdir(parents=True, exist_ok=True)
    stem = 'THETABOT_COST_RISK_2026-10-13'
    compressed = gzip.compress((out / 'summary.json').read_bytes(), mtime=0)
    (destination / f'{stem}.json.gz').write_bytes(compressed)
    with zipfile.ZipFile(destination / f'{stem}_EVIDENCE.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out.iterdir()):
            if path.suffix in ('.csv', '.json'): bundle.write(path, path.name)
        bundle.write(PLAN, PLAN.name)
    lines = ['# ThetaBot: turnover and actual-holdings risk — 2026-10-13', '',
        '**The policy `guarded_edge` was fixed before development payoffs.**',
        'All dates were already studied. These are modeled net returns, not promised income.', '',
        'Ordinary trades can stop at the nearest 1pp band edge. Optional risk checks reduce',
        'actual inventory at every 4h open using the prior completed-day 60-day covariance',
        'with 25% diagonal shrinkage and a fixed 15% annualized volatility ceiling.',
        'Target caps remain 50% total/25% per asset; the guard never adds leverage or cash.', '',
        f"Protocol SHA-256: `{report['protocol_sha256']}`. [Fixed plan](COST_RISK_RESEARCH_PLAN.md).", '']
    for capital, scenarios in report['results'].items():
        gates = report['gates'][capital]
        lines += [f'## Initial account {capital} USDT', '',
            f"Profit-improvement screen: **{gates['profit_improvement_screen_passed']}**; risk-improvement screen: **{gates['risk_improvement_screen_passed']}**.", '',
            '| Policy | Nominal full net | Full joint stress net | Worst full DD | Full CAGR | Full fills | Fees + impact |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for policy in POLICIES:
            row = scenarios['nominal']['full_continuous'][policy]
            dd = min(scenarios[s]['full_continuous'][policy]['max_intraday_drawdown_bound'] for s in SCENARIOS)
            lines.append(f"| `{policy}` | {row['total_return']:+.2%} | {scenarios['joint_stress']['full_continuous'][policy]['total_return']:+.2%} | {dd:.2%} | {row['cagr']:+.2%} | {row['trades_count']} | {row['fees_paid_total'] + row['slippage_paid_total']:.2f} |")
        lines += ['', '| Policy | 2025 nominal | 2026 nominal | Later continuous nominal |',
                  '|---|---:|---:|---:|']
        for policy in POLICIES:
            lines.append('| `' + policy + '` | ' + ' | '.join(f"{scenarios['nominal'][period][policy]['total_return']:+.2%}"
                        for period in ('2025', '2026', 'later_continuous')) + ' |')
        lines += ['', 'Independent yearly accounts reset funding; continuous columns carry all holdings/losses.',
            'Percentages across separately funded years must not be added.', '', '| Fixed-candidate gate | Passed |', '|---|:---:|']
        lines += [f'| {name} | {value} |' for name, value in gates['profit_gates'].items()]
        lines += ['', '| Full scenario | Guarded-edge net | DD bound | Risk reduction fills | Residual bars |',
                  '|---|---:|---:|---:|---:|']
        for scenario in SCENARIOS:
            row = scenarios[scenario]['full_continuous'][CANDIDATE]
            lines.append(f"| {scenario} | {row['total_return']:+.2%} | {row['max_intraday_drawdown_bound']:.2%} | {row['risk_reduction_fills']} | {row['risk_target_residual_bars']} |")
        lines += ['', 'Residual bars disclose the actual holdings left above the requested constrained target,',
            'including the 10-USDT minimum and costs. A requested cap is not an enforced future risk guarantee.', '']
    lines += ['## Independent evidence and limits', '',
        f"All {report['verified_archives']} official archive bytes/checksums were rechecked without network access.",
        f"All {report['reconciled_paths']} paths reconcile cash, nonnegative owned inventory and NAV.",
        f"All {report['archived_control_checks']} archived control paths reproduce; maximum NAV error {report['maximum_nav_error']:.3g} USDT.",
        'The evidence ZIP contains 32 complete continuous accounts (64 CSV files), source manifest,',
        'protocol/selection timestamps, runtime versions and SHA-256 hashes for every CSV.',
        'Ablations diagnose the intervention and do not replace the fixed candidate after results.', '']
    if 'independent_csv_audit' in report:
        audit = report['independent_csv_audit']
        lines += [f"A separate CSV audit checks {audit['audited_accounts']} accounts, {audit['audited_bars']:,} bars and {audit['audited_fills']:,} fills.",
            f"It recomputes source-open fill costs, NAV and conservative DD; direct NumPy return-window covariance verifies guard mathematics.",
            f"Maximum independent NAV residual: {audit['maximum_nav_error']:.3g} USDT; volatility residual: {audit['maximum_risk_volatility_error']:.3g}.", '']
    lines += [
        *['- ' + item for item in report['limitations']], '',
        'Live eligibility is false. No exchange connection, live/default selection, deployment or merge is enabled.', '',
        'Repeated selection on known data can overfit: [Bailey et al., original paper](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).', '',
        '## Reproduce', '', '```bash',
        'python -m scripts.download_spot_flow',
        'python -m scripts.research_cost_risk --freeze-only',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_cost_risk --jobs 4 --report-out docs/evaluation',
        '```', '']
    (destination / f'{stem}.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw/spot_flow'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/cost_risk_20261013'))
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument('--report-out', type=Path)
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.freeze_only:
        from datetime import datetime, timezone
        path = args.out / 'frozen_protocol.json'
        if path.exists(): raise ValueError('Refuse to overwrite prior protocol evidence; use a new out directory')
        path.write_text(json.dumps({'protocol_sha256': hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            'fixed_candidate': CANDIDATE, 'frozen_before_variant_payoffs': True,
            'frozen_at': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
        return
    report = research(args.data, args.out, args.jobs)
    from scripts.audit_cost_risk import audit
    report['independent_csv_audit'] = audit(args.data, args.out)
    (args.out / 'independent_csv_audit.json').write_text(json.dumps(report['independent_csv_audit'], indent=2, allow_nan=False) + '\n')
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.report_out: write_report(report, args.out, args.report_out)
    print(json.dumps(report['gates'], indent=2))


if __name__ == '__main__': main()
