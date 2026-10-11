"""Fixed causal payoff filters, warmup controls and forecast-uncertainty scenarios."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from scripts.research_option_budgets import research, REPO
from spot_bot.research.long_options import LongOptionsConfig
from spot_bot.research.option_selection import OptionForecastConfig, CausalOptionEntryFilter

PLAN = REPO / 'docs/evaluation/OPTION_FORECAST_RESEARCH_PLAN.md'
SCENARIOS = ('nominal', 'cost_stress', 'delayed', 'joint_stress', 'forecast_low', 'forecast_high')


def forecast_configs():
    configs = {}
    for signal in ('call', 'direction'):
        configs[signal + '_warmup'] = OptionForecastConfig(method='warmup')
        for suffix, method, edge in [('empirical10', 'empirical', .1), ('empirical25', 'empirical', .25),
                                     ('rv10', 'gaussian_rv', .1)]:
            configs[signal + '_' + suffix] = OptionForecastConfig(method=method, minimum_edge=edge)
    return configs


def policy_configs():
    policies = {'options_off': None, 'reserved_cash': None,
        'call_current1': LongOptionsConfig(options_enabled=True, policy='call_otm', monthly_account_budget=.01),
        'call_current2': LongOptionsConfig(options_enabled=True, policy='call_otm', monthly_account_budget=.02),
        'direction_current2': LongOptionsConfig(options_enabled=True, policy='direction_otm', monthly_account_budget=.02)}
    for name in forecast_configs():
        policies[name] = LongOptionsConfig(options_enabled=True,
            policy=name.split('_')[0] + '_otm', monthly_account_budget=.02)
    return policies


def entry_filters(markets, scenario):
    factor = .75 if scenario == 'forecast_low' else 1.25 if scenario == 'forecast_high' else 1.
    return {p: CausalOptionEntryFilter(markets, replace(c, volatility_scale=factor))
            for p, c in forecast_configs().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spot-data', type=Path, default=Path('data/raw/spot_flow'))
    parser.add_argument('--option-data', type=Path, default=Path('data/raw/long_options/entry_observations.json'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/option_selection_20261012'))
    parser.add_argument('--options-enabled', action='store_true')
    args = parser.parse_args()
    if not args.options_enabled:
        parser.error('Payoff-forecast research requires explicit offline --options-enabled opt-in')
    args.out.mkdir(parents=True, exist_ok=True)
    proof = args.out / 'frozen_protocol.json'
    if not proof.exists():
        proof.write_text(json.dumps({'protocol_sha256': hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            'frozen_before_variant_payoffs': True}) + '\n')
    r = research(args.spot_data, args.option_data, args.out, plan_path=PLAN,
        policy_factory=policy_configs, entry_filter_factory=entry_filters,
        forecast_configs={p: asdict(c) for p, c in forecast_configs().items()}, scenarios=SCENARIOS,
        reference_path=REPO / 'docs/evaluation/THETABOT_OPTION_BUDGETS_2026-10-12.json',
        reference_names={p: p for p in ('options_off', 'reserved_cash', 'call_current1', 'call_current2', 'direction_current2')})
    r['limitations'].extend(['Forecast models assume conditional delivery-index/spot parity and uncalibrated return distributions',
        'Overlapping historical return windows are not independent samples',
        'Forecast volatility scaling changes entry decisions; it is separate from synthetic held-mark IV sensitivity'])
    (args.out / 'summary.json').write_text(json.dumps(r, separators=(',', ':'), allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
