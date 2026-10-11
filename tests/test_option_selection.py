"""Causality, cost-aware selection, forecast uncertainty and integration checks."""
from dataclasses import replace
import copy

import numpy as np
import pandas as pd
import pytest

from spot_bot.research.long_options import LongOptionsConfig, option_value, run_optional_long_options
from spot_bot.research.option_selection import OptionForecastConfig, CausalOptionEntryFilter


def histories(days=425, growth=.002, oscillation=.03):
    index = pd.date_range('2023-01-01', periods=days * 6, freq='4h', tz='UTC').as_unit('ns')
    day = np.arange(len(index)) // 6
    prices = 100 * np.exp(growth * day + oscillation * np.sin(day))
    return {s: pd.DataFrame({'open': prices, 'high': prices * 1.01, 'low': prices * .99,
        'close': prices, 'volume': 100.}, index=index) for s in ('BTCUSDT', 'ETHUSDT', 'BNBUSDT')}


def row(family='inverse'):
    when = pd.Timestamp('2024-01-01', tz='UTC')
    prefix = 'BTC' if family == 'inverse' else 'BTC_USDC'
    return {'entry': str(when), 'execution_time': str(when), 'expiry': '2024-01-26 08:00:00+00:00',
        'family': family, 'policy': 'call_atm', 'asset': 'BTC', 'status': 'observed_trade', 'delay_bars': 0,
        'trade_age_seconds': 3600., 'official_delivery_price': 100.,
        'contract': {'instrument_name': prefix + '-26JAN24-100-C', 'strike': 100., 'option_type': 'call',
            'creation_timestamp': (when - pd.Timedelta(days=30)).value // 1_000_000,
            'tick_size': .0001 if family == 'inverse' else .01, 'min_trade_amount': .01, 'contract_size': 1.},
        'trade': {'price': .01 if family == 'inverse' else 1., 'index_price': 100., 'iv': 50.,
            'timestamp': (when - pd.Timedelta(hours=1)).value // 1_000_000}}


def free_costs(family='inverse'):
    return LongOptionsConfig(options_enabled=True, family=family, premium_markup=0., option_fee_rate=0.,
        delivery_fee_rate=0., conversion_fee=0., conversion_impact=0.)


@pytest.mark.parametrize('method', ['empirical', 'gaussian_rv', 'warmup'])
def test_forecast_ignores_current_day_closes_future_prices_and_actual_delivery(method):
    markets = histories()
    item = row()
    config = OptionForecastConfig(method=method)
    a = CausalOptionEntryFilter(markets, config)(item, 100., free_costs(), item['entry'])
    changed = {s: f.copy() for s, f in markets.items()}
    for frame in changed.values():
        frame.loc[frame.index >= pd.Timestamp(item['entry']), 'close'] *= 1000.
    future = copy.deepcopy(item)
    future['official_delivery_price'] = 999999.
    b = CausalOptionEntryFilter(changed, config)(future, 100., free_costs(), item['entry'])
    assert a == b
    assert pd.Timestamp(a['last_completed_daily_close']) + pd.Timedelta(days=1) <= pd.Timestamp(item['entry'])


def test_empirical_forecast_matches_a_known_constant_growth_payoff():
    markets = histories(15, growth=np.log(1.01), oscillation=0.)
    item = row();item['expiry'] = '2023-01-14 00:00:00+00:00'
    config = OptionForecastConfig(minimum_daily_closes=8, maximum_daily_closes=12, volatility_days=3)
    result = CausalOptionEntryFilter(markets, config)(item, 100., free_costs(), '2023-01-11T00:00:00Z')
    assert result['forecast_horizon_days'] == 3
    assert result['expected_net_receipt_per_unit'] == pytest.approx(100 * 1.01**3 - 100)
    assert result['accepted']


@pytest.mark.parametrize('side', ['call', 'put'])
def test_gaussian_rv_quadrature_agrees_with_independent_option_formula(side):
    markets = histories(growth=0.)
    item = row();item['contract']['option_type'] = side;item['contract']['strike'] = 105.
    forecast = CausalOptionEntryFilter(markets, OptionForecastConfig(method='gaussian_rv'))
    result = forecast(item, 100., free_costs(), item['entry'])
    value = option_value(100., 105., result['forecast_horizon_days'] / 365.25,
                         result['historical_annualized_volatility'], side)
    assert result['expected_net_receipt_per_unit'] == pytest.approx(float(value), rel=.02, abs=.03)
    assert 0 <= result['estimated_win_probability'] <= 1


def test_warmup_control_accepts_expensive_option_while_forecast_rejects_it():
    markets = histories(growth=0., oscillation=0.)
    item = row()
    assert CausalOptionEntryFilter(markets, OptionForecastConfig(method='warmup'))(item, 100., free_costs(), item['entry'])['accepted']
    rejected = CausalOptionEntryFilter(markets)(item, 100., free_costs(), item['entry'])
    assert not rejected['accepted'] and rejected['expected_edge'] == -1.


def test_higher_purchase_cost_can_turn_a_positive_estimated_edge_into_rejection():
    markets = histories(15, growth=np.log(1.01), oscillation=0.)
    item = row();item['expiry'] = '2023-01-14 00:00:00+00:00';item['trade']['price'] = .02
    config = OptionForecastConfig(minimum_daily_closes=8, maximum_daily_closes=12, volatility_days=3, minimum_edge=.25)
    forecast = CausalOptionEntryFilter(markets, config)
    assert forecast(item, 100., free_costs(), '2023-01-11T00:00:00Z')['accepted']
    stressed = forecast(item, 100., replace(free_costs(), premium_markup=.8), '2023-01-11T00:00:00Z')
    assert not stressed['accepted'] and stressed['buy_debit_per_unit'] > 2.


def test_incomplete_or_missing_daily_history_is_rejected_without_gap_compression():
    markets = histories()
    item = row()
    early = CausalOptionEntryFilter(markets)(item, 100., free_costs(), '2023-05-01T00:00:00Z')
    assert not early['accepted'] and early['reason'] == 'insufficient_history'
    markets['BTCUSDT'].loc['2023-07-01', 'close'] = np.nan
    bad = CausalOptionEntryFilter(markets)(item, 100., free_costs(), item['entry'])
    assert not bad['accepted'] and bad['reason'] == 'invalid_history'


def test_forecast_volatility_changes_estimates_but_held_mark_iv_does_not():
    markets, item = histories(growth=0.), row()
    low = CausalOptionEntryFilter(markets, OptionForecastConfig(method='gaussian_rv', volatility_scale=.75))
    high = CausalOptionEntryFilter(markets, OptionForecastConfig(method='gaussian_rv', volatility_scale=1.25))
    a = low(item, 100., free_costs(), item['entry'])
    b = high(item, 100., free_costs(), item['entry'])
    assert b['expected_net_receipt_per_unit'] > a['expected_net_receipt_per_unit']
    assert low(item, 100., replace(free_costs(), iv_mark_multiplier=2.), item['entry']) == a


def test_rejected_purchase_keeps_funded_cash_and_auditable_reason():
    markets, item = histories(growth=0., oscillation=0.), row()
    target = pd.DataFrame(.15, index=markets['BTCUSDT'].index, columns=list(markets))
    forecast = CausalOptionEntryFilter(markets)
    _, fills, stats = run_optional_long_options(markets, target, [item],
        config=LongOptionsConfig(options_enabled=True, monthly_account_budget=.02),
        start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_usdt=1000., entry_filter=forecast)
    assert fills['options'].empty and stats['options']['final_cash'] == 100.
    assert stats['options']['forecast_rejections'] == 1
    assert stats['options']['entry_selection'][0]['reason'] == 'insufficient_edge'
    assert stats['options']['accounting']['reconciled']


def test_default_off_switch_does_not_call_or_read_the_forecast_filter():
    markets = histories()
    target = pd.DataFrame(.15, index=markets['BTCUSDT'].index, columns=list(markets))
    def fail(*args):raise AssertionError('Disabled filter was used')
    a = run_optional_long_options(markets, target, [{'unread': True}], start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z')
    b = run_optional_long_options(markets, target, [{'unread': True}], start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', entry_filter=fail)
    pd.testing.assert_frame_equal(a[0], b[0]);pd.testing.assert_frame_equal(a[1], b[1]);assert a[2] == b[2]


@pytest.mark.parametrize('kwargs', [{'method': 'oracle'}, {'minimum_edge': -.1}, {'minimum_edge': np.nan},
    {'volatility_scale': 0.}, {'volatility_scale': np.inf}, {'minimum_daily_closes': True}, {'volatility_days': 252}])
def test_invalid_forecast_configuration_fails_closed(kwargs):
    with pytest.raises(ValueError): OptionForecastConfig(**kwargs)
