"""Meaningful paid-option risk, expiry cash flow, causality and off-switch checks."""
from dataclasses import replace
from contextlib import nullcontext
import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts.download_long_options import select_contract
from spot_bot.backtest.intraday import run_intraday_spot
from spot_bot.research.long_options import (LongOptionsConfig, option_value, entry_cost,
    expiry_receipt, replay_option_sleeve, run_optional_long_options, reconcile_option_sleeve,
    owned_spot_open_equity, reconcile_purchase_budgets)


def market(days=60):
    dates = pd.date_range('2024-01-01', periods=days * 6, freq='4h', tz='UTC').as_unit('ns')
    markets = {s: pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 100.}, index=dates)
               for s in ('BTCUSDT', 'ETHUSDT', 'BNBUSDT')}
    target = pd.DataFrame(.15, index=dates, columns=list(markets))
    return markets, target


def observation(family='inverse', side='call', entry='2024-01-01', expiry='2024-01-26 08:00', delivery=100.):
    when = pd.Timestamp(entry, tz='UTC')
    prefix = 'BTC' if family == 'inverse' else 'BTC_USDC'
    contract = {'instrument_name': prefix + '-26JAN24-100-' + ('C' if side == 'call' else 'P'),
        'strike': 100., 'option_type': side, 'creation_timestamp': (when - pd.Timedelta(days=20)).value // 1_000_000,
        'tick_size': .0001 if family == 'inverse' else .01, 'min_trade_amount': .01, 'contract_size': 1.}
    return {'entry': str(when), 'execution_time': str(when), 'expiry': str(pd.Timestamp(expiry, tz='UTC')),
        'family': family, 'policy': 'call_atm', 'asset': 'BTC', 'status': 'observed_trade', 'delay_bars': 0,
        'contract': contract, 'official_delivery_price': delivery, 'trade_age_seconds': 3600.,
        'trade': {'price': .01 if family == 'inverse' else 1., 'index_price': 100., 'iv': 50.,
                  'timestamp': (when - pd.Timedelta(hours=1)).value // 1_000_000}}


def test_option_pricing_intrinsic_limits_and_put_call_parity():
    spots = np.array([50., 100., 200.])
    call, put = option_value(spots, 100., .1, .6, 'call'), option_value(spots, 100., .1, .6, 'put')
    np.testing.assert_allclose(call - put, spots - 100., atol=1e-12)
    assert (call >= 0).all() and (call <= spots).all()
    assert (put >= 0).all() and (put <= 100).all()
    np.testing.assert_array_equal(option_value(spots, 100., 0., .6, 'call'), [0., 0., 100.])
    np.testing.assert_array_equal(option_value(spots, 100., 0., .6, 'put'), [50., 0., 0.])


def test_disabled_options_are_exactly_the_existing_spot_account():
    markets, target = market()
    a = run_intraday_spot(markets, target, daily_control=True)
    b = run_optional_long_options(markets, target, [{'intentionally': 'unread'}])
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
    assert a[2] == b[2]


@pytest.mark.parametrize('family', ['inverse', 'linear'])
def test_complete_premium_loss_cannot_debit_other_funds_or_exceed_the_budget(family):
    markets, _ = market(31)
    config = LongOptionsConfig(options_enabled=True, family=family)
    eq, trades, s = replay_option_sleeve(markets, [observation(family)], config,
        start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_cash=100.)
    paid = -trades.loc[trades.event == 'buy', 'cash_flow'].sum()
    assert 0 < paid <= 20. + 1e-10
    assert s['final_cash'] == pytest.approx(100 - paid)
    assert s['settlements'] == 1 and s['winning_options'] == 0
    assert s['minimum_cash'] >= 80. - 1e-10
    assert s['accounting']['reconciled']
    assert eq.held_contracts.iloc[-1] == 0.
    assert trades.loc[trades.event == 'settle', 'option_fee'].sum() == 0.


def test_real_delivery_index_and_coin_conversion_determine_inverse_payout():
    row = observation(delivery=120.)
    config = LongOptionsConfig(options_enabled=True, premium_markup=0., option_fee_rate=0.,
        delivery_fee_rate=0., conversion_fee=0., conversion_impact=0.)
    receipt, costs = expiry_receipt(row, 90., config)
    assert receipt == pytest.approx((120 - 100) / 120 * 90)
    assert costs['gross_payoff'] == receipt
    assert expiry_receipt(observation(delivery=80.), 90., config)[0] == 0.


def test_buy_fee_tick_markup_and_currency_costs_are_added_once():
    row, config = observation(), LongOptionsConfig(options_enabled=True)
    unit, lot, parts = entry_cost(row, 100., config)
    assert parts['raw_premium'] == 1.
    assert parts['premium_markup_cost'] == pytest.approx(.05)
    assert parts['option_fee'] == pytest.approx(.03)
    assert parts['conversion_impact'] == pytest.approx(1.08 * .0006)
    assert parts['conversion_fee'] == pytest.approx(1.08 * 1.0006 * .001)
    assert unit == sum(parts.values()) and lot == .1


def test_unaffordable_minimum_lot_is_skipped_without_fractional_fake_fill():
    markets, _ = market(31)
    row = observation()
    row['trade']['price'] = 10.
    eq, trades, s = replay_option_sleeve(markets, [row], LongOptionsConfig(options_enabled=True),
        start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_cash=100.)
    assert trades.empty and s['minimum_lot_skips'] == 1 and s['final_cash'] == 100.
    assert eq.equity.eq(100).all()


def test_future_payoff_cannot_change_entry_cash_quantity_or_pre_expiry_equity():
    markets, _ = market(31)
    row = observation(delivery=100.)
    future = copy.deepcopy(row)
    future['official_delivery_price'] = 150.
    config = LongOptionsConfig(options_enabled=True)
    args = dict(start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_cash=100.)
    a, fa, _ = replay_option_sleeve(markets, [row], config, **args)
    b, fb, _ = replay_option_sleeve(markets, [future], config, **args)
    cutoff = pd.Timestamp(row['expiry'])
    pd.testing.assert_frame_equal(a[a.timestamp < cutoff], b[b.timestamp < cutoff])
    pd.testing.assert_frame_equal(fa[fa.event == 'buy'], fb[fb.event == 'buy'])
    assert a.equity.iloc[-1] < b.equity.iloc[-1]


def test_entry_iv_marking_changes_drawdown_but_not_closed_cash_profit():
    markets, _ = market(31)
    row, config = observation(delivery=115.), LongOptionsConfig(options_enabled=True)
    args = dict(start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_cash=100.)
    a, _, sa = replay_option_sleeve(markets, [row], config, **args)
    b, _, sb = replay_option_sleeve(markets, [row], replace(config, iv_mark_multiplier=2.), **args)
    assert sa['final_cash'] == sb['final_cash']
    assert not np.allclose(a.equity, b.equity)


def test_losses_reduce_next_months_cash_budget_without_martingale():
    markets, _ = market(60)
    first = observation()
    second = observation(entry='2024-02-05', expiry='2024-02-23 08:00')
    second['contract']['instrument_name'] = 'BTC-23FEB24-100-C'
    _, trades, s = replay_option_sleeve(markets, [first, second], LongOptionsConfig(options_enabled=True),
        start='2024-01-01T00:00:00Z', end='2024-03-01T00:00:00Z', initial_cash=100.)
    buys = trades[trades.event == 'buy']
    assert len(buys) == 2
    assert buys.qty.iloc[1] < buys.qty.iloc[0]
    assert -buys.cash_flow.iloc[1] <= .2 * (100 + buys.cash_flow.iloc[0]) + 1e-10
    assert s['maximum_monthly_budget_fraction'] <= .2 + 1e-10


def test_enabled_account_reserves_option_capital_instead_of_adding_free_money():
    markets, target = market(31)
    copy_mode = pd.option_context('mode.copy_on_write', True) if int(pd.__version__.split('.')[0]) < 3 else nullcontext()
    with copy_mode:
        eq, _, s = run_optional_long_options(markets, target, [observation()],
            config=LongOptionsConfig(options_enabled=True), start='2024-01-01T00:00:00Z',
            end='2024-02-01T00:00:00Z', initial_usdt=1000.)
    spot, _, _ = run_intraday_spot(markets, target, daily_control=True,
        start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_usdt=900.)
    pd.testing.assert_series_equal(eq.spot_equity, spot.equity, check_names=False)
    assert s['net_pnl'] == pytest.approx(s['spot']['net_pnl'] + s['options']['net_pnl'])
    assert s['options']['initial_cash'] == 100.
    assert s['options']['final_cash'] < 100.
    assert not s['live_eligible']
    json.dumps(s, allow_nan=False)


def test_low_level_option_replay_also_requires_explicit_offline_opt_in():
    markets, _ = market(31)
    with pytest.raises(ValueError, match='explicit offline opt-in'):
        replay_option_sleeve(markets, [observation()], LongOptionsConfig(),
            start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_cash=100.)


def test_independent_ledger_detects_changed_contract_quantity():
    markets, _ = market(31)
    row, config = observation(), LongOptionsConfig(options_enabled=True)
    eq, trades, _ = replay_option_sleeve(markets, [row], config,
        start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_cash=100.)
    pos = {'row': row, 'instrument': row['contract']['instrument_name'], 'entry': pd.Timestamp(row['entry']),
           'expiry': pd.Timestamp(row['expiry']), 'qty': trades.qty.iloc[0]}
    trades.loc[trades.event == 'buy', 'qty'] += .1
    with pytest.raises(ValueError, match='quantities do not reconcile'):
        reconcile_option_sleeve(markets, eq, trades, [pos], config, 100.)


def test_contract_selection_excludes_future_listing_and_does_not_use_payoffs():
    row = observation()
    entry, expiry = pd.Timestamp(row['entry']), pd.Timestamp(row['expiry'])
    a = {**row['contract'], 'expiration_timestamp': expiry.value // 1_000_000, 'strike': 99.}
    b = {**a, 'strike': 101.}
    future = {**a, 'strike': 100., 'creation_timestamp': entry.value // 1_000_000 + 1}
    assert select_contract([future, b, a], 'BTC', 'inverse', 'call', 100., entry, expiry) == a


@pytest.mark.parametrize('kind', ['future_trade', 'stale_trade', 'future_contract'])
def test_unverified_or_noncausal_entry_is_rejected(kind):
    markets, _ = market(31)
    row = observation()
    now = pd.Timestamp(row['entry']).value // 1_000_000
    if kind == 'future_trade': row['trade']['timestamp'] = now
    elif kind == 'stale_trade': row['trade']['timestamp'] = now - 4 * 3600_000 - 1
    else: row['contract']['creation_timestamp'] = now + 1
    with pytest.raises(ValueError, match='Future or stale'):
        replay_option_sleeve(markets, [row], LongOptionsConfig(options_enabled=True),
            start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_cash=100.)


@pytest.mark.parametrize('kwargs', [{'monthly_cash_budget': 1.1}, {'sleeve_fraction': np.nan},
    {'options_enabled': 1}, {'family': 'future'}, {'policy': 'written_call'}, {'delay_bars': True},
    {'option_fee_rate': -.1}, {'conversion_fee': -.1}, {'iv_mark_multiplier': 0.}])
def test_invalid_optional_risk_or_product_config_fails_closed(kwargs):
    with pytest.raises(ValueError): LongOptionsConfig(**kwargs)


@pytest.mark.parametrize('family', ['inverse', 'linear'])
def test_account_purchase_limit_includes_all_costs_and_respects_owned_cash(family):
    markets, target = market(31)
    config = LongOptionsConfig(options_enabled=True, family=family, monthly_account_budget=.01)
    _, fills, summary = run_optional_long_options(markets, target, [observation(family)],
        config=config, start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_usdt=1000.)
    paid = -fills['options'].loc[fills['options'].event == 'buy', 'cash_flow'].sum()
    assert 0 < paid <= 10. + 1e-10
    assert summary['options']['final_cash'] == pytest.approx(100 - paid)
    assert summary['options']['maximum_monthly_debit_account_fraction'] <= .01
    assert summary['options']['accounting']['monthly_budgets_reconciled']


@pytest.mark.parametrize('basis', ['current', 'initial'])
def test_account_budget_cannot_expand_with_only_the_option_sleeves_wins(basis):
    markets, _ = market(60)
    first = observation(delivery=500.)
    second = observation(entry='2024-02-05', expiry='2024-02-23 08:00')
    second['contract']['instrument_name'] = 'BTC-23FEB24-100-C'
    config = LongOptionsConfig(options_enabled=True, monthly_account_budget=.02, account_budget_basis=basis)
    opening = pd.Series(900., index=markets['BTCUSDT'].index)
    _, trades, stats = replay_option_sleeve(markets, [first, second], config,
        start='2024-01-01T00:00:00Z', end='2024-03-01T00:00:00Z', initial_cash=100.,
        initial_account_equity=1000., spot_open_values=opening)
    first_budget, later = stats['purchase_budgets']
    assert first_budget['allowed_debit'] == 20.
    assert later['cash_before'] > 100.
    expected = .02 * (later['cash_before'] + 900. if basis == 'current' else 1000.)
    assert later['allowed_debit'] == pytest.approx(expected)
    assert later['spent'] <= expected + 1e-10
    assert later['spent'] < .2 * later['cash_before']
    assert trades.qty.gt(0).all()


@pytest.mark.parametrize('problem', ['missing', 'wrong_capital', 'wrong_funding', 'negative', 'shifted'])
def test_account_cap_rejects_inconsistent_or_unfunded_context(problem):
    markets, _ = market(31)
    opening = pd.Series(900., index=markets['BTCUSDT'].index)
    capital = 1000.
    if problem == 'missing': opening = None
    elif problem == 'wrong_capital': capital = 2000.
    elif problem == 'wrong_funding': opening.iloc[0] = 1900.
    elif problem == 'negative': opening.iloc[-1] = -1.
    elif problem == 'shifted': opening.index += pd.Timedelta(hours=4)
    with pytest.raises(ValueError, match='funded'):
        replay_option_sleeve(markets, [observation()], LongOptionsConfig(options_enabled=True, monthly_account_budget=.02),
            start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_cash=100.,
            initial_account_equity=capital, spot_open_values=opening)


def test_opening_budget_uses_previous_owned_inventory_and_current_open_only():
    markets, _ = market(1)
    dates = markets['BTCUSDT'].index
    spot = pd.DataFrame({'timestamp': dates, 'usdt': 500.,
        'base_BTCUSDT': 1., 'base_ETHUSDT': 0., 'base_BNBUSDT': 0.})
    a = owned_spot_open_equity(markets, spot, 1000.)
    changed = spot.copy()
    changed.loc[2:, ['usdt', 'base_BTCUSDT']] = 999999.
    markets['BTCUSDT'].loc[dates[2:], 'close'] = 999999.
    b = owned_spot_open_equity(markets, changed, 1000.)
    np.testing.assert_array_equal(a.iloc[:3], b.iloc[:3])
    assert a.iloc[0] == 1000. and a.iloc[1] == 600.


def test_independent_budget_audit_detects_added_cash_debit():
    markets, _ = market(31)
    config = LongOptionsConfig(options_enabled=True, monthly_account_budget=.01)
    _, trades, stats = replay_option_sleeve(markets, [observation()], config,
        start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_cash=100.,
        initial_account_equity=1000., spot_open_values=pd.Series(900., index=markets['BTCUSDT'].index))
    trades.loc[trades.event == 'buy', 'cash_flow'] -= 1.
    with pytest.raises(ValueError, match='Monthly purchase'):
        reconcile_purchase_budgets(trades, stats['purchase_budgets'], config)


def test_counterfactual_unfilled_entry_is_separate_from_missing_source_quotes():
    markets, _ = market(31)
    row = observation()
    _, trades, stats = replay_option_sleeve(markets, [row], LongOptionsConfig(options_enabled=True),
        start='2024-01-01T00:00:00Z', end='2024-02-01T00:00:00Z', initial_cash=100.,
        unfilled_entries={(row['contract']['instrument_name'], row['execution_time'])})
    assert trades.empty and stats['counterfactual_unfilled'] == 1
    assert stats['no_recent_trade'] == 0 and stats['final_cash'] == 100.


@pytest.mark.parametrize('kwargs', [{'monthly_account_budget': -.01}, {'monthly_account_budget': np.nan},
    {'monthly_account_budget': .3}, {'account_budget_basis': 'future'}])
def test_invalid_account_purchase_limits_are_rejected(kwargs):
    with pytest.raises(ValueError): LongOptionsConfig(**kwargs)
