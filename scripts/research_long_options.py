"""Fixed exploratory paid-option study; real trade proxies, modeled held marks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.download_long_options import SIGNALS
from scripts.download_spot_flow import BEGIN, END
from scripts.research_spot_flow import load_spot_flow, NOMINAL, STRESS, VALIDATION, TRANSFER
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.momentum_refinement import momentum_refinement_targets
from spot_bot.research.long_options import LongOptionsConfig, replay_option_sleeve, combine_paths

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / 'docs/evaluation/LONG_OPTIONS_RESEARCH_PLAN.md'


def verify_source_evidence(source, folder):
    """Match every recorded trade and delivery to hashed raw official responses."""
    verified, delivery = set(), {'BTC': {}, 'ETH': {}}
    for proof in source['source_files']:
        path = folder / proof['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != proof['sha256']:
            raise ValueError('Option metadata/delivery source checksum changed')
        for asset in delivery:
            if proof['file'].startswith(asset + '_delivery_'):
                rows = json.loads(path.read_text())['result']['data']
                delivery[asset].update({r['date']: float(r['delivery_price']) for r in rows})
    for row in source['records']:
        proof = row.get('trade_source')
        if proof is None:
            continue
        path = folder / 'trade_cache' / proof['file']
        if proof['file'] not in verified:
            if hashlib.sha256(path.read_bytes()).hexdigest() != proof['sha256']:
                raise ValueError('Historical option trade source checksum changed')
            verified.add(proof['file'])
        trades = json.loads(path.read_text())['result']['trades']
        if row['status'] == 'observed_trade':
            if len(trades) != 1 or trades[0] != row['trade']:
                raise ValueError('Recorded option premium does not match source trade')
            if row['trade']['instrument_name'] != row['contract']['instrument_name']:
                raise ValueError('Trade and selected option instrument do not match')
            expiry = pd.Timestamp(row['expiry']).strftime('%Y-%m-%d')
            if row['official_delivery_price'] != delivery[row['asset']][expiry]:
                raise ValueError('Recorded delivery price does not match source index')
        elif row['status'] == 'no_recent_trade' and trades:
            raise ValueError('An available historical trade was mislabeled absent')
    return {'verified_raw_source_files': len(source['source_files']),
            'verified_trade_windows': len(verified), 'trade_and_delivery_fields_match': True}


def research(spot_folder, option_path, out, *, options_enabled=False):
    markets, manifest = load_spot_flow(spot_folder)
    targets, _ = momentum_refinement_targets(markets)
    target = targets['capped_allocation']
    out.mkdir(parents=True, exist_ok=True)
    protocol_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    source = json.loads(option_path.read_text()) if options_enabled else None
    if source is not None:
        if source['protocol_sha256'] != protocol_hash:
            raise ValueError('Option observations are not from the frozen protocol')
        if source['spot_dataset_hashes'] != {s: r['sha256'] for s, r in manifest['datasets'].items()}:
            raise ValueError('Option signal source hashes do not match verified spot data')
    source_checks = verify_source_evidence(source, option_path.parent) if source else None
    policies = {'options_off': None}
    if options_enabled:
        policies['reserved_cash'] = None
        policies.update({f'{family}_{signal}': LongOptionsConfig(options_enabled=True, family=family, policy=signal)
                         for family in ('inverse', 'linear') for signal in SIGNALS})
    paths, checks, summaries, closed = {}, [], {}, []
    old = json.loads((REPO / 'docs/evaluation/THETABOT_MULTISCALE_SPOT_2026-10-09.json').read_text())

    def replay(capital, policy, start, end, scenario, label, iv_factor=1.):
        delay = int(scenario == 'delayed')
        costs = STRESS if scenario == 'cost_stress' else NOMINAL
        reserved = policy != 'options_off'
        initial_spot = capital * (.9 if reserved else 1.)
        key = capital, initial_spot, str(start), str(end), scenario
        if key not in paths:
            eq, fills, s = run_intraday_spot(markets, target, start=start, end=end,
                initial_usdt=initial_spot, daily_control=True, signal_delay_bars=delay, **costs)
            s['accounting'] = reconcile_spot_path(markets, eq, fills, initial_spot)
            paths[key] = eq, fills, s
            checks.append(s['accounting'])
        spot, fills, spot_summary = paths[key]
        if not reserved:
            result, eq = dict(spot_summary), spot
            result.update(drawdown_uses_synthetic_option_marks=False, options_enabled=False,
                          option_net_pnl=0., spot_net_pnl=result['net_pnl'])
            if capital == 1000 and label in old['later']['nominal'] and scenario in ('nominal', 'cost_stress'):
                reference = old['later']['nominal' if scenario == 'nominal' else 'doubled'][label]['capped_control']
                for field in ('total_return', 'max_intraday_drawdown_bound', 'fees_paid_total', 'slippage_paid_total', 'trades_count'):
                    if not np.isclose(result[field], reference[field], rtol=1e-10, atol=1e-8):
                        raise ValueError(f'Existing spot control changed: {label} {field}')
        else:
            config = policies[policy]
            if config is None:
                dates = spot.timestamp
                option = pd.DataFrame({'timestamp': dates, 'equity': capital * .1, 'cash': capital * .1,
                    'open_equity': capital * .1, 'high_bound': capital * .1, 'low_bound': capital * .1})
                option_summary = {'net_pnl': 0., 'buys': 0, 'minimum_lot_skips': 0, 'no_recent_trade': 0,
                                  'no_listed_contract': 0, 'accounting': {'reconciled': True}}
            else:
                config = replace(config, delay_bars=delay, iv_mark_multiplier=iv_factor)
                if scenario == 'cost_stress':
                    config = replace(config, premium_markup=.2, option_fee_rate=.0006, delivery_fee_rate=.0003,
                        conversion_fee=.002, conversion_impact=.0012)
                option, option_trades, option_summary = replay_option_sleeve(markets, source['records'], config,
                    start=start, end=end, initial_cash=capital * .1)
                checks.append(option_summary['accounting'])
                if label == 'full_continuous' and scenario == 'nominal' and iv_factor == 1:
                    option.to_csv(out / f'{capital}_{policy}_option_equity.csv', index=False)
                    option_trades.to_csv(out / f'{capital}_{policy}_option_ledger.csv', index=False)
                    closed.extend({'capital': capital, 'policy': policy, **r} for r in option_summary['closed_options'])
                option_summary = {k: v for k, v in option_summary.items() if k != 'closed_options'}
            eq, result = combine_paths(spot, option, capital, markets, initial_spot)
            result.update(spot_net_pnl=spot_summary['net_pnl'], option_net_pnl=option_summary['net_pnl'],
                options=option_summary, spot_fees=spot_summary['fees_paid_total'], spot_impact=spot_summary['slippage_paid_total'],
                options_enabled=config is not None, live_eligible=False)
        if label == 'full_continuous' and scenario == 'nominal' and iv_factor == 1:
            eq.to_csv(out / f'{capital}_{policy}_combined_equity.csv', index=False)
        return result

    selection = {}
    for capital in (1000, 10000):
        dev = {p: replay(capital, p, BEGIN, VALIDATION, 'nominal', 'development') for p in policies}
        eligible = [p for p, s in dev.items() if s['net_pnl'] > 0 and s['max_intraday_drawdown_bound'] >= -.2]
        frozen = max(eligible, key=lambda p: dev[p]['net_pnl']) if eligible else 'cash'
        selection[str(capital)] = {'development': dev, 'frozen_candidate': frozen}
        print(f'Frozen development {capital}: {frozen}', flush=True)
    (out / 'frozen_selection.json').write_text(json.dumps({'protocol_sha256': protocol_hash, **selection}, indent=2, allow_nan=False) + '\n')
    periods = {'2025': (VALIDATION, TRANSFER), '2026': (TRANSFER, END),
               'later_continuous': (VALIDATION, END), 'full_continuous': (BEGIN, END)}
    for capital in (1000, 10000):
        record = summaries[str(capital)] = {'selection': selection[str(capital)], 'scenarios': {}}
        for scenario in ('nominal', 'cost_stress', 'delayed'):
            record['scenarios'][scenario] = {}
            for label, (start, end) in periods.items():
                result = {p: replay(capital, p, start, end, scenario, label) for p in policies}
                record['scenarios'][scenario][label] = result
                if label == 'full_continuous':
                    print(f'{capital} {scenario}: ' + '; '.join(f'{p} {r["total_return"]:+.2%} DD {r["max_intraday_drawdown_bound"]:.2%}' for p, r in result.items()), flush=True)
        marking = record['iv_mark_sensitivity'] = {}
        for factor in (.5, 2.):
            marking[str(factor)] = {}
            for policy in policies:
                if policies[policy] is None:
                    continue
                s = replay(capital, policy, BEGIN, END, 'nominal', 'marking', iv_factor=factor)
                main = record['scenarios']['nominal']['full_continuous'][policy]
                if not np.isclose(s['net_pnl'], main['net_pnl'], rtol=1e-10, atol=1e-8):
                    raise ValueError('Synthetic marks changed the realized expiry payoff profit')
                marking[str(factor)][policy] = {'net_pnl': s['net_pnl'], 'model_dd': s['max_intraday_drawdown_bound']}
    pd.DataFrame(closed).to_csv(out / 'closed_options.csv', index=False)
    result = {'protocol': str(PLAN.relative_to(REPO)), 'protocol_sha256': protocol_hash,
        'options_enabled': options_enabled, 'live_eligible': False, 'execution_prices_are_trade_proxies': True,
        'held_option_marks_are_synthetic': True, 'validated_executable_option_profit': False,
        'source_checks': source_checks,
        'source_manifest': {k: v for k, v in source.items() if k != 'records'} if source else None,
        'option_observation_sha256': hashlib.sha256(option_path.read_bytes()).hexdigest() if source else None,
        'results': summaries, 'reconciled_paths': len(checks),
        'accounting_passed': all(c['reconciled'] for c in checks),
        'maximum_accounting_error': max((c.get('max_equity_error', 0.) for c in checks), default=0.),
        'policies': {p: asdict(c) if c else None for p, c in policies.items()},
        'limitations': ['Trade prices are not historical executable bid/ask quotes or market depth',
            'Constant-entry-IV Black-Scholes held values are synthetic, including model drawdowns',
            'Uniform current fee scenarios and conservative lot floors are not dated rule histories',
            'Binance spot converts inverse premiums/payoffs; index/venue basis remains',
            'USD/USDT/USDC parity, stablecoin and exchange failure risk are not simulated',
            'These spot dates and momentum signals were studied before; not an unseen holdout',
            'A fixed 10% initial option sleeve is not replenished; wins can change its later weight',
            'A 20% historical model drawdown is not a future loss guarantee']}
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != protocol_hash:
        raise ValueError('Frozen protocol changed during profitability replay')
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spot-data', type=Path, default=Path('data/raw/spot_flow'))
    parser.add_argument('--option-data', type=Path, default=Path('data/raw/long_options/entry_observations.json'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/long_options_20261010'))
    parser.add_argument('--options-enabled', action='store_true', help='Enable ONLY the offline fixed paid-option study')
    args = parser.parse_args()
    research(args.spot_data, args.option_data, args.out, options_enabled=args.options_enabled)


if __name__ == '__main__':
    main()
