"""Frozen offline Bayesian/conditional-Monte-Carlo overlays of the spot signal."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from scripts.download_spot_flow import BEGIN, END
from scripts.research_option_budgets import compare_economic_fields
from scripts.research_spot_flow import load_spot_flow, NOMINAL, STRESS, VALIDATION, TRANSFER
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.core.cost_model import compute_round_trip_cost
from spot_bot.portfolio.forecast import price_features
from spot_bot.portfolio.momentum_refinement import momentum_refinement_targets
from spot_bot.portfolio.probabilistic import (FORECAST_FEATURES, SIGNAL_POLICIES,
    ProbabilisticConfig, probabilistic_signal_policies, walk_forward_probabilistic)

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / 'docs/evaluation/BAYES_MONTE_CARLO_RESEARCH_PLAN.md'
SEEDS = (20261012, 20261013, 20261014)
SCENARIOS = ('nominal', 'cost_stress', 'delayed', 'joint_stress')
PERIODS = {'development': (BEGIN, VALIDATION), '2025': (VALIDATION, TRANSFER),
           '2026': (TRANSFER, END), 'later_continuous': (VALIDATION, END),
           'full_continuous': (BEGIN, END)}
_WORKER = None


def initialize_worker(markets, policies, out):
    global _WORKER
    _WORKER = markets, policies, out


def replay_task(task):
    capital, seed, scenario, period, policy = task
    markets, policies, out = _WORKER
    costly = scenario in ('cost_stress', 'joint_stress')
    delay = int(scenario in ('delayed', 'joint_stress'))
    if policy == 'cash':
        target = next(iter(policies.values()))['spot_control'][0] * 0.
        buy_mask = None
    else:
        target, buy_mask = policies[seed, costly][policy]
    start, end = PERIODS[period]
    eq, fills, summary = run_intraday_spot(markets, target, initial_usdt=capital,
        start=start, end=end, daily_control=True, signal_delay_bars=delay,
        completed_buy_mask=buy_mask, **(STRESS if costly else NOMINAL))
    summary['accounting'] = reconcile_spot_path(markets, eq, fills, capital)
    if period == 'full_continuous' and scenario == 'nominal' and seed == SEEDS[0]:
        eq.to_csv(out / f'{capital}_{policy}_equity.csv', index=False)
        fills.to_csv(out / f'{capital}_{policy}_trades.csv', index=False)
    return task, summary


def evaluate(tasks, executor):
    result = {}
    for task, summary in executor.map(replay_task, tasks) if executor else map(replay_task, tasks):
        result[task] = summary
        capital, seed, scenario, period, policy = task
        print(f'{capital} {period} {scenario} seed={seed} {policy}: '
              f"net {summary['total_return']:+.2%}; DD {summary['max_intraday_drawdown_bound']:.2%}", flush=True)
    return result


def select_development(by_scenario):
    eligible = [p for p in SIGNAL_POLICIES if all(
        by_scenario[s][p]['net_pnl'] > 0 and by_scenario[s][p]['max_intraday_drawdown_bound'] >= -.20
        for s in SCENARIOS)]
    ordered = sorted(eligible, key=lambda p: (-min(by_scenario[s][p]['total_return'] for s in SCENARIOS),
                                            -by_scenario['nominal'][p]['total_return'], p))
    return ordered[0] if ordered else 'cash'


def forecast_skill(forecasts, daily, hurdles):
    result = {}
    for period in ('development', '2025', '2026', 'later_continuous', 'full_continuous'):
        start, end = PERIODS[period]
        result[period] = {}
        for symbol, frame in forecasts.items():
            outcome = np.log(daily[symbol].close.shift(-5) / daily[symbol].close)
            outcome_at = frame.index + pd.Timedelta('6D')
            keep = frame.ready & outcome.notna() & (frame.available_at >= start) & (outcome_at <= end)
            rows = frame.loc[keep]
            actual = outcome.loc[keep].to_numpy()
            metrics = {'observations': len(rows), 'zero_log_return_mse': float(np.square(actual).mean())}
            for expert in ('bayes', 'mc', 'fused', 'unconditional'):
                error = rows[expert + '_mean'].to_numpy() - actual
                metrics[expert] = {'log_return_mse': float(np.square(error).mean()),
                    'mean_bias': float(error.mean()),
                    'gain_brier_loss': {name: float(np.square(rows[f'{expert}_prob_{name}'].to_numpy()
                                                - (actual > hurdle)).mean()) for name, hurdle in hurdles.items()}}
            result[period][symbol] = metrics
    return result


def decision_audit(forecasts, base_target, hurdles):
    """One source decision per completed daily close, before any execution lag."""
    records = []
    for symbol, frame in forecasts.items():
        original = base_target[symbol].reindex(frame.index + pd.Timedelta('20h')).to_numpy()
        for name, hurdle in hurdles.items():
            for expert in ('bayes', 'mc', 'fused'):
                means, probabilities = frame[expert + '_mean'], frame[f'{expert}_prob_{name}']
                gate = frame.ready & means.gt(hurdle) & probabilities.ge(.55)
                conviction = ((probabilities - .5) / .1).clip(0., 1.).where(means.gt(hurdle), 0.).fillna(0.)
                rows = pd.DataFrame({'day': frame.index, 'available_at': frame.available_at,
                    'symbol': symbol, 'hurdle_scenario': name, 'expert': expert,
                    'ready': frame.ready, 'mean_log_return': means, 'gain_probability': probabilities,
                    'log_cost_hurdle': hurdle, 'buy_allowed': gate, 'conviction': conviction,
                    'original_target': original,
                    'blended_target': original * (.5 + .5 * conviction) * frame.ready,
                    'training_count': frame.training_count,
                    'last_training_outcome_at': frame.last_training_outcome_at,
                    'last_fusion_outcome_at': frame.last_fusion_outcome_at})
                records.append(rows)
    return pd.concat(records, ignore_index=True)


def validate_forecasts(forecasts, hurdles):
    ready_count, max_numeric_z, max_probability_error = 0, 0., 0.
    from scipy.stats import norm, t as student_t
    for frame in forecasts.values():
        f = frame.loc[frame.ready]
        ready_count += len(f)
        for key in ('last_training_outcome_at', 'last_analogue_outcome_at', 'last_fusion_outcome_at'):
            if not f[key].dropna().le(f.loc[f[key].notna(), 'available_at']).all():
                raise ValueError('Forecast used an unmatured outcome')
        if ((f.training_count < 252).any() or (f.bayes_observations > np.ceil(f.training_count / 5)).any()
                or not np.allclose(f.fused_weight_bayes + f.fused_weight_mc, 1.)
                or not f.fused_weight_bayes.between(0, 1).all()
                or not (f.fused_variance >= f[['bayes_variance', 'mc_variance']].min(axis=1)).all()):
            raise ValueError('Forecast history or uncertainty invariant failed')
        difference = np.abs(f.mc_mean - f.mc_exact_analogue_mean)
        denominator = f.mc_numerical_mean_se.clip(lower=1e-15)
        max_numeric_z = max(max_numeric_z, float((difference / denominator).max()))
        for name, hurdle in hurdles.items():
            bayes = student_t.sf((hurdle - f.bayes_mean) / f.bayes_scale, f.bayes_df)
            fused = norm.sf((hurdle - f.fused_mean) / np.sqrt(f.fused_variance))
            max_probability_error = max(max_probability_error,
                float(np.max(np.abs(bayes - f[f'bayes_prob_{name}']))),
                float(np.max(np.abs(fused - f[f'fused_prob_{name}']))))
            for expert in ('bayes', 'mc', 'fused', 'unconditional'):
                if not f[f'{expert}_prob_{name}'].between(0, 1).all():
                    raise ValueError('Invalid model probability')
    return {'ready_forecasts': ready_count, 'maximum_numerical_mc_mean_error_in_se': max_numeric_z,
            'maximum_probability_recalculation_error': max_probability_error}


def research(spot_folder, out, jobs=4):
    proof = json.loads((out / 'frozen_protocol.json').read_text())
    plan_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    if proof['protocol_sha256'] != plan_hash or proof['frozen_before_variant_payoffs'] is not True:
        raise ValueError('Protocol must be frozen before candidate payoffs')
    markets, manifest = load_spot_flow(spot_folder)
    targets, daily = momentum_refinement_targets(markets)
    base_target = targets['capped_allocation']
    features = price_features(daily)
    hurdles = {name: float(np.log1p(compute_round_trip_cost(**costs)))
               for name, costs in (('nominal', NOMINAL), ('cost_stress', STRESS))}
    policies, predictions, checks = {}, {}, {}
    for seed in SEEDS:
        config = replace(ProbabilisticConfig(), seed=seed)
        predictions[seed] = {s: walk_forward_probabilistic(features[s].loc[:, FORECAST_FEATURES],
            daily[s].close, symbol=s, hurdles=hurdles, config=config) for s in markets}
        checks[str(seed)] = validate_forecasts(predictions[seed], hurdles)
        for name, costly in (('nominal', False), ('cost_stress', True)):
            policies[seed, costly] = probabilistic_signal_policies(base_target, predictions[seed],
                hurdle_name=name, hurdle=hurdles[name], enabled=True)
        print(f'Built causal forecasts for seed {seed}: {checks[str(seed)]}', flush=True)
    pd.concat([f.assign(symbol=s, day=f.index) for s, f in predictions[SEEDS[0]].items()],
              ignore_index=True).to_csv(out / 'forecasts.csv', index=False)
    decision_audit(predictions[SEEDS[0]], base_target, hurdles).to_csv(out / 'signal_decisions.csv', index=False)
    initialize_worker(markets, policies, out)
    pool = ProcessPoolExecutor(max_workers=jobs, initializer=initialize_worker,
                               initargs=(markets, policies, out)) if jobs > 1 else None
    try:
        development_tasks = [(capital, SEEDS[0], scenario, 'development', policy)
                             for capital in (1000, 10000) for scenario in SCENARIOS for policy in SIGNAL_POLICIES]
        dev = evaluate(development_tasks, pool)
        selection = {}
        for capital in (1000, 10000):
            by_scenario = {s: {p: dev[capital, SEEDS[0], s, 'development', p] for p in SIGNAL_POLICIES}
                           for s in SCENARIOS}
            selection[str(capital)] = {'frozen_candidate': select_development(by_scenario),
                'development': by_scenario, 'criterion': 'max worst-scenario positive return; DD <=20% in all four'}
        (out / 'frozen_selection.json').write_text(json.dumps({'protocol_sha256': plan_hash,
            'selection': selection}, indent=2, allow_nan=False) + '\n')
        print('Frozen before later payoffs: ' + str({k: v['frozen_candidate'] for k, v in selection.items()}), flush=True)
        later_tasks = [(capital, SEEDS[0], scenario, period, policy)
            for capital in (1000, 10000) for scenario in SCENARIOS
            for period in ('2025', '2026', 'later_continuous', 'full_continuous')
            for policy in (*SIGNAL_POLICIES, *(('cash',) if selection[str(capital)]['frozen_candidate'] == 'cash' else ()))]
        later = evaluate(later_tasks, pool)
        seed_tasks = [(capital, seed, 'nominal', period, policy)
            for capital in (1000, 10000) for seed in SEEDS[1:]
            for period in ('later_continuous', 'full_continuous') for policy in SIGNAL_POLICIES[2:]]
        sensitivities = evaluate(seed_tasks, pool)
    finally:
        if pool:
            pool.shutdown()
    skills = forecast_skill(predictions[SEEDS[0]], daily, hurdles)
    archived = json.loads((REPO / 'docs/evaluation/THETABOT_OPTION_BUDGETS_2026-10-12.json').read_text())
    old_checks = 0
    excluded = {'option_net_pnl', 'spot_net_pnl', 'options_enabled', 'drawdown_uses_synthetic_option_marks'}
    for (capital, seed, scenario, period, policy), actual in later.items():
        if policy != 'spot_control':
            continue
        old = archived['results'][str(capital)]['scenarios'][scenario][period]['options_off']
        if set(old) - set(actual) != excluded:
            raise ValueError('Unrecognized archived spot control field')
        compare_economic_fields(actual, {k: v for k, v in old.items() if k not in excluded})
        old_checks += 1
    results, gates = {}, {}
    for capital in (1000, 10000):
        key, frozen = str(capital), selection[str(capital)]['frozen_candidate']
        results[key] = {'selection': selection[key], 'scenarios': {s: {period: {
            p: value for (c, seed, scenario, label, p), value in later.items()
            if (c, seed, scenario, label) == (capital, SEEDS[0], s, period)}
            for period in ('2025', '2026', 'later_continuous', 'full_continuous')} for s in SCENARIOS},
            'seed_sensitivity': {str(seed): {period: {
                p: value for (c, sd, scenario, label, p), value in sensitivities.items()
                if (c, sd, label) == (capital, seed, period)}
                for period in ('later_continuous', 'full_continuous')} for seed in SEEDS[1:]}}
        model_selected = frozen in SIGNAL_POLICIES[2:]
        primary = results[key]['scenarios']
        full = [primary[s]['full_continuous'][frozen] for s in SCENARIOS]
        seed_paths = [results[key]['seed_sensitivity'][str(seed)][period][frozen]
                      for seed in SEEDS[1:] for period in ('later_continuous', 'full_continuous')] if model_selected else []
        risk = all(x['max_intraday_drawdown_bound'] >= -.20 for x in [*full, *seed_paths])
        beats_controls = all(primary[s]['later_continuous'][frozen]['total_return'] >
            max(primary[s]['later_continuous'][p]['total_return'] for p in ('spot_control', 'ready_control'))
            for s in ('nominal', 'joint_stress'))
        positive = all(primary[s][period][frozen]['net_pnl'] > 0 for s in SCENARIOS
                       for period in ('later_continuous', 'full_continuous')) and all(x['net_pnl'] > 0 for x in seed_paths)
        gates[key] = {'model_selected': model_selected, 'beats_both_later_controls': beats_controls,
                      'full_and_seed_risk_passed': risk, 'all_later_full_positive': positive,
                      'historical_economic_improvement': model_selected and beats_controls and risk and positive}
    accounts = [value['accounting'] for value in [*dev.values(), *later.values(), *sensitivities.values()]]
    summary = {'protocol_sha256': plan_hash, 'frozen_protocol': proof,
        'model_config': asdict(ProbabilisticConfig()), 'features': list(FORECAST_FEATURES),
        'scenarios': list(SCENARIOS), 'seeds': list(SEEDS), 'log_cost_hurdles': hurdles,
        'spot_dataset_hashes': {s: x['sha256'] for s, x in manifest['datasets'].items()},
        'runtime_versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'scikit-learn')},
        'results': results, 'historical_gates': gates, 'forecast_skill': skills,
        'forecast_audit': checks, 'archived_control_checks': old_checks,
        'reconciled_paths': len(accounts), 'accounting_passed': all(a['reconciled'] for a in accounts),
        'maximum_accounting_error': max(max(a[k] for k in ('max_cash_error','max_inventory_error','max_equity_error')) for a in accounts),
        'probabilistic_enabled': True, 'options_enabled': False, 'live_eligible': False,
        'dates_were_previously_studied': True, 'limitations': [
            'These experts reuse price/theta information; Monte Carlo sampling does not add independent market data.',
            'Bayesian linear/noise assumptions, nearest-neighbor states and Gaussian fusion probabilities are uncalibrated.',
            'Analogues and evaluated five-day horizons may overlap; no independent significance claim is made.',
            'Monte Carlo standard errors measure numerical integration only, not epistemic uncertainty.',
            'The archived three-asset universe and all dates have already been studied.',
            'Fixed fee and impact scenarios are not observed live fee tiers, asks, liquidity or delays.',
            'Target caps and historical drawdown screens do not guarantee future losses or eliminate venue/quote risk.']}
    (out / 'summary.json').write_text(json.dumps(summary, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'gates': gates, 'paths': len(accounts), 'archived_controls': old_checks,
                      'maximum_accounting_error': summary['maximum_accounting_error']}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spot-data', type=Path, default=Path('data/raw/spot_flow'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/bayes_monte_carlo_20261012'))
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--probabilistic-enabled', action='store_true')
    args = parser.parse_args()
    if not args.probabilistic_enabled:
        parser.error('Offline probabilistic research requires explicit --probabilistic-enabled opt-in')
    if not 1 <= args.jobs <= 8:
        parser.error('--jobs must be between 1 and 8')
    args.out.mkdir(parents=True, exist_ok=True)
    proof = args.out / 'frozen_protocol.json'
    if not proof.exists():
        proof.write_text(json.dumps({'protocol_sha256': hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            'frozen_before_variant_payoffs': True, 'created_at': datetime.now(timezone.utc).isoformat(),
            'parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()},
            indent=2) + '\n')
    research(args.spot_data, args.out, args.jobs)


if __name__ == '__main__':
    main()
