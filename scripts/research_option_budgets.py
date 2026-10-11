"""Fixed follow-up: bounded paid-option debits and joint/missed-fill stresses."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.download_spot_flow import BEGIN, END
from scripts.research_spot_flow import load_spot_flow, NOMINAL, STRESS, VALIDATION, TRANSFER
from scripts.research_long_options import verify_source_evidence
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.momentum_refinement import momentum_refinement_targets
from spot_bot.research.long_options import (LongOptionsConfig, replay_option_sleeve,
    combine_paths, owned_spot_open_equity)

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / 'docs/evaluation/OPTION_BUDGET_RESEARCH_PLAN.md'
ORIGINAL_PLAN = REPO / 'docs/evaluation/LONG_OPTIONS_RESEARCH_PLAN.md'
SCENARIOS = ('nominal', 'cost_stress', 'delayed', 'joint_stress')


def fixed_policies():
    policies = {'options_off': None, 'reserved_cash': None,
        'legacy_call_otm': LongOptionsConfig(options_enabled=True, policy='call_otm'),
        'legacy_direction_otm': LongOptionsConfig(options_enabled=True, policy='direction_otm')}
    for signal in ('call', 'direction'):
        for name, fraction, basis in [('current2', .02, 'current'), ('current1', .01, 'current'),
                                      ('initial2', .02, 'initial')]:
            policies[f'{signal}_{name}'] = LongOptionsConfig(options_enabled=True,
                policy=signal + '_otm', monthly_account_budget=fraction, account_budget_basis=basis)
    return policies


def compare_economic_fields(actual, archived, path=''):
    """Every archived field is retained; new budget diagnostics may be added."""
    if isinstance(archived, dict):
        for key, value in archived.items():
            compare_economic_fields(actual[key], value, path + '/' + key)
    elif isinstance(archived, (float, int)) and not isinstance(archived, bool):
        if not np.isclose(actual, archived, rtol=1e-10, atol=1e-8):
            raise ValueError(f'Archived control changed: {path}: {actual} vs {archived}')
    elif actual != archived:
        raise ValueError(f'Archived control changed: {path}')


def read_study(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_text())


def research(spot_folder, option_path, out, *, plan_path=PLAN, policy_factory=None,
             entry_filter_factory=None, forecast_configs=None, scenarios=SCENARIOS,
             reference_path=None, reference_names=None):
    out.mkdir(parents=True, exist_ok=True)
    protocol_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if json.loads((out / 'frozen_protocol.json').read_text())['protocol_sha256'] != protocol_hash:
        raise ValueError('Follow-up protocol was not frozen before payoffs')
    markets, manifest = load_spot_flow(spot_folder)
    source = json.loads(option_path.read_text())
    if source['protocol_sha256'] != hashlib.sha256(ORIGINAL_PLAN.read_bytes()).hexdigest():
        raise ValueError('Original option source protocol changed')
    if source['spot_dataset_hashes'] != {s: x['sha256'] for s, x in manifest['datasets'].items()}:
        raise ValueError('Option and verified spot source hashes differ')
    source_checks = verify_source_evidence(source, option_path.parent)
    targets, _ = momentum_refinement_targets(markets)
    target = targets['capped_allocation']
    archived = read_study(reference_path or REPO / 'docs/evaluation/THETABOT_LONG_OPTIONS_2026-10-10.json')
    policies = (policy_factory or fixed_policies)()
    filters = {s: entry_filter_factory(markets, s) if entry_filter_factory else {} for s in scenarios}
    old_names = reference_names or {'options_off': 'options_off', 'reserved_cash': 'reserved_cash',
        'legacy_call_otm': 'inverse_call_otm', 'legacy_direction_otm': 'inverse_direction_otm'}
    paths, checks, summaries, closed, budget_rows, nominal_closures = {}, [], {}, [], [], {}
    selection_rows = []
    control_checks = 0

    def replay(capital, policy, start, end, scenario, label, *, iv_factor=1., unfilled=frozenset()):
        nonlocal control_checks
        delay = int(scenario in ('delayed', 'joint_stress'))
        costly = scenario in ('cost_stress', 'joint_stress')
        fraction = 1. if policy == 'options_off' else .9
        key = capital, fraction, str(start), str(end), scenario
        if key not in paths:
            spot, fills, stats = run_intraday_spot(markets, target, start=start, end=end,
                initial_usdt=capital * fraction, daily_control=True, signal_delay_bars=delay,
                **(STRESS if costly else NOMINAL))
            stats['accounting'] = reconcile_spot_path(markets, spot, fills, capital * fraction)
            opening = owned_spot_open_equity(markets, spot, capital * fraction)
            paths[key] = spot, fills, stats, opening
            checks.append(stats['accounting'])
        spot, fills, spot_stats, opening = paths[key]
        option_trades = pd.DataFrame()
        if policy == 'options_off':
            eq, result = spot, dict(spot_stats)
            result.update(option_net_pnl=0., spot_net_pnl=result['net_pnl'], options_enabled=False,
                          drawdown_uses_synthetic_option_marks=False)
        else:
            config = policies[policy]
            if config is None:
                option = pd.DataFrame({'timestamp': spot.timestamp, 'equity': capital * .1,
                    'cash': capital * .1, 'open_equity': capital * .1,
                    'high_bound': capital * .1, 'low_bound': capital * .1})
                option_stats = {'net_pnl': 0., 'buys': 0, 'minimum_lot_skips': 0,
                    'no_recent_trade': 0, 'no_listed_contract': 0, 'accounting': {'reconciled': True}}
            else:
                config = replace(config, delay_bars=delay, iv_mark_multiplier=iv_factor)
                if costly:
                    config = replace(config, premium_markup=.2, option_fee_rate=.0006,
                        delivery_fee_rate=.0003, conversion_fee=.002, conversion_impact=.0012)
                option, option_trades, option_stats = replay_option_sleeve(markets, source['records'], config,
                    start=start, end=end, initial_cash=capital * .1, initial_account_equity=capital,
                    spot_open_values=opening, unfilled_entries=unfilled,
                    entry_filter=filters.get(scenario, {}).get(policy))
                checks.append(option_stats['accounting'])
                if label == 'full_continuous' and scenario == 'nominal' and iv_factor == 1.:
                    nominal_closures[capital, policy] = option_stats['closed_options']
                    closed.extend({'capital': capital, 'policy': policy, **x} for x in option_stats['closed_options'])
                    budget_rows.extend({'capital': capital, 'policy': policy, **x}
                        for x in option_stats.get('purchase_budgets', []))
                    selection_rows.extend({'capital': capital, 'policy': policy, **x}
                        for x in option_stats.get('entry_selection', []))
                    option.to_csv(out / f'{capital}_{policy}_option_equity.csv', index=False)
                    option_trades.to_csv(out / f'{capital}_{policy}_option_ledger.csv', index=False)
                option_stats = {k: v for k, v in option_stats.items() if k not in ('closed_options', 'purchase_budgets', 'entry_selection')}
            eq, result = combine_paths(spot, option, capital, markets, capital * fraction)
            result.update(spot_net_pnl=spot_stats['net_pnl'], option_net_pnl=option_stats['net_pnl'],
                options=option_stats, spot_fees=spot_stats['fees_paid_total'],
                spot_impact=spot_stats['slippage_paid_total'], options_enabled=config is not None, live_eligible=False)
        if label == 'full_continuous' and scenario == 'nominal' and iv_factor == 1.:
            eq.to_csv(out / f'{capital}_{policy}_combined_equity.csv', index=False)
        if config_for_stats := policies[policy]:
            buys = option_trades.loc[option_trades.event == 'buy'] if not option_trades.empty else option_trades
            monthly_fractions = []
            if not buys.empty:
                for when, events in buys.groupby('timestamp', sort=True):
                    cash_before = float(events.cash_after.iloc[0] - events.cash_flow.iloc[0])
                    monthly_fractions.append(float(-events.cash_flow.sum()) / (float(opening.loc[when]) + cash_before))
            result['maximum_monthly_debit_account_fraction'] = max(monthly_fractions, default=0.)
            result['account_purchase_cap'] = config_for_stats.monthly_account_budget
            result['account_purchase_cap_basis'] = config_for_stats.account_budget_basis
        old_name = old_names.get(policy)
        reference_scenario = 'nominal' if scenario in ('forecast_low', 'forecast_high') else scenario
        if old_name and reference_scenario in archived['results'][str(capital)]['scenarios'] and label in ('2025', '2026', 'later_continuous', 'full_continuous'):
            reference = archived['results'][str(capital)]['scenarios'][reference_scenario][label][old_name]
            compare_economic_fields(result, {k: v for k, v in reference.items() if k != 'top_three_option_pnl_fraction_of_net'})
            control_checks += 1
        if not np.isclose(result['spot_net_pnl'] + result['option_net_pnl'], result['net_pnl'], rtol=1e-10, atol=1e-8):
            raise ValueError('Funded account contributions do not reconcile')
        return result

    selection = {}
    for capital in (1000, 10000):
        dev = {s: {p: replay(capital, p, BEGIN, VALIDATION, s, 'development') for p in policies} for s in scenarios}
        eligible = [p for p in policies if all(dev[s][p]['net_pnl'] > 0
                    and dev[s][p]['max_intraday_drawdown_bound'] >= -.2 for s in scenarios)]
        scores = {p: min(dev[s][p]['net_pnl'] for s in scenarios) for p in eligible}
        frozen = max(eligible, key=lambda p: (scores[p], dev['nominal'][p]['net_pnl'], p)) if eligible else 'cash'
        selection[str(capital)] = {'development': dev, 'eligible': eligible,
                                  'minimum_scenario_net_pnl': scores, 'frozen_candidate': frozen}
        print(f'Frozen {len(scenarios)}-scenario development {capital}: {frozen}', flush=True)
    (out / 'frozen_selection.json').write_text(json.dumps({'protocol_sha256': protocol_hash,
        **selection}, separators=(',', ':'), allow_nan=False) + '\n')
    periods = {'2025': (VALIDATION, TRANSFER), '2026': (TRANSFER, END),
               'later_continuous': (VALIDATION, END), 'full_continuous': (BEGIN, END)}
    for capital in (1000, 10000):
        record = summaries[str(capital)] = {'selection': selection[str(capital)], 'scenarios': {}}
        for scenario in scenarios:
            record['scenarios'][scenario] = {}
            for label, (start, end) in periods.items():
                result = {p: replay(capital, p, start, end, scenario, label) for p in policies}
                record['scenarios'][scenario][label] = result
                if label == 'full_continuous':
                    print(f'{capital} {scenario}: ' + '; '.join(f'{p} {x["total_return"]:+.2%} DD '
                        f'{x["max_intraday_drawdown_bound"]:.2%}' for p, x in result.items()), flush=True)
        record['missed_top_three'] = {}
        record['iv_mark_sensitivity'] = {'0.5': {}, '2.0': {}}
        for policy, config in policies.items():
            if config is None:
                continue
            closures = nominal_closures[capital, policy]
            top = sorted((x for x in closures if x['net_pnl'] > 0),
                key=lambda x: (x['net_pnl'], x['instrument'], x['entry']), reverse=True)[:3]
            omitted = {(x['instrument'], x['entry']) for x in top}
            result = replay(capital, policy, BEGIN, END, 'nominal', 'missed_top_three', unfilled=omitted)
            record['missed_top_three'][policy] = {'omitted_entries': [
                {'instrument': x['instrument'], 'entry': x['entry'], 'original_net_pnl': x['net_pnl']} for x in top],
                'result': result, 'is_ex_post_diagnostic': True}
            nominal = record['scenarios']['nominal']['full_continuous'][policy]
            nominal['top_three_option_pnl_fraction_of_net'] = sum(x['net_pnl'] for x in top) / nominal['option_net_pnl'] if nominal['option_net_pnl'] > 0 else None
            if old_names.get(policy):
                prior = archived['results'][str(capital)]['scenarios']['nominal']['full_continuous'][old_names[policy]]
                if 'top_three_option_pnl_fraction_of_net' in prior:
                    compare_economic_fields(nominal['top_three_option_pnl_fraction_of_net'], prior['top_three_option_pnl_fraction_of_net'])
            for factor in (.5, 2.):
                s = replay(capital, policy, BEGIN, END, 'nominal', 'marking', iv_factor=factor)
                if not np.isclose(s['net_pnl'], nominal['net_pnl'], rtol=1e-10, atol=1e-8):
                    raise ValueError('Synthetic option marks changed terminal profit')
                record['iv_mark_sensitivity'][str(factor)][policy] = {
                    'net_pnl': s['net_pnl'], 'model_dd': s['max_intraday_drawdown_bound']}
        frozen = record['selection']['frozen_candidate']
        record['frozen_full_risk_screen_passed'] = frozen != 'cash' and all(
            record['scenarios'][s]['full_continuous'][frozen]['net_pnl'] > 0
            and record['scenarios'][s]['full_continuous'][frozen]['max_intraday_drawdown_bound'] >= -.2 for s in scenarios)
    pd.DataFrame(closed).to_csv(out / 'closed_options.csv', index=False)
    pd.DataFrame(budget_rows).to_csv(out / 'purchase_budgets.csv', index=False)
    if forecast_configs is not None:
        pd.DataFrame(selection_rows).to_csv(out / 'entry_selection.csv', index=False)
    result = {'protocol_sha256': protocol_hash, 'original_observation_sha256': hashlib.sha256(option_path.read_bytes()).hexdigest(),
        'source_checks': source_checks, 'spot_dataset_hashes': source['spot_dataset_hashes'],
        'policies': {p: asdict(c) if c else None for p, c in policies.items()}, 'results': summaries,
        'archived_control_checks': control_checks, 'reconciled_paths': len(checks),
        'accounting_passed': all(c['reconciled'] for c in checks),
        'maximum_accounting_error': max(c.get('max_equity_error', 0.) for c in checks),
        'options_enabled': True, 'live_eligible': False, 'execution_prices_are_trade_proxies': True,
        'held_option_marks_are_synthetic': True, 'dates_were_previously_studied': True,
        'limitations': archived['limitations'] + [
            'Entry debit limits do not guarantee a cap on held option market values or future drawdown',
            'Missed top-three fills are selected ex post as a fragility diagnostic, not an entry predictor']}
    if forecast_configs is not None:
        result['forecast_configs'] = forecast_configs
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != protocol_hash:
        raise ValueError('Frozen follow-up protocol changed during replay')
    (out / 'summary.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spot-data', type=Path, default=Path('data/raw/spot_flow'))
    parser.add_argument('--option-data', type=Path, default=Path('data/raw/long_options/entry_observations.json'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/option_budgets_20261012'))
    parser.add_argument('--options-enabled', action='store_true')
    args = parser.parse_args()
    if not args.options_enabled:
        parser.error('Bounded option research requires explicit offline --options-enabled opt-in')
    args.out.mkdir(parents=True, exist_ok=True)
    proof = args.out / 'frozen_protocol.json'
    if not proof.exists():
        proof.write_text(json.dumps({'protocol_sha256': hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                                    'frozen_before_variant_payoffs': True}) + '\n')
    research(args.spot_data, args.option_data, args.out)


if __name__ == '__main__':
    main()
