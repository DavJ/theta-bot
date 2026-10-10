"""Fully paid long-option sleeve; historical trade proxies, synthetic held marks.

No short options, borrowing, order submission or replenishment from spot. The
default off switch delegates directly to the existing spot replay unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.special import ndtr

from spot_bot.backtest.intraday import run_intraday_spot, rolling_gains


@dataclass(frozen=True)
class LongOptionsConfig:
    options_enabled: bool = False
    family: str = 'inverse'
    policy: str = 'call_atm'
    sleeve_fraction: float = .1
    monthly_cash_budget: float = .2
    premium_markup: float = .05
    option_fee_rate: float = .0003
    delivery_fee_rate: float = .00015
    fee_cap: float = .125
    conversion_fee: float = .001
    conversion_impact: float = .0006
    delay_bars: int = 0
    iv_mark_multiplier: float = 1.

    def __post_init__(self):
        if (type(self.options_enabled) is not bool or self.family not in ('inverse', 'linear')
                or self.policy not in ('call_atm', 'call_otm', 'direction_atm', 'direction_otm')
                or type(self.delay_bars) is not int or self.delay_bars not in (0, 1)):
            raise ValueError('Invalid long-option research switch/policy/delay')
        values = [self.sleeve_fraction, self.monthly_cash_budget, self.premium_markup,
                  self.option_fee_rate, self.delivery_fee_rate, self.fee_cap,
                  self.conversion_fee, self.conversion_impact, self.iv_mark_multiplier]
        if (not np.isfinite(values).all() or not 0 < self.sleeve_fraction < 1
                or not 0 < self.monthly_cash_budget <= 1 or not 0 <= self.premium_markup <= 1
                or not 0 <= self.option_fee_rate < .1 or not 0 <= self.delivery_fee_rate < .1
                or not 0 <= self.fee_cap <= 1 or not 0 <= self.conversion_fee < .1
                or not 0 <= self.conversion_impact < .1 or self.iv_mark_multiplier <= 0):
            raise ValueError('Invalid owned-cash option budget/cost/valuation inputs')


def option_value(spot, strike, years, iv, side):
    """Zero-rate European premium per underlying unit; not an observed mark."""
    s, t = np.asarray(spot, dtype=float), np.asarray(years, dtype=float)
    if (side not in ('call', 'put') or not np.isfinite(s).all() or (s <= 0).any()
            or not np.isfinite(t).all() or (t < 0).any()
            or not np.isfinite([strike, iv]).all() or strike <= 0 or iv <= 0):
        raise ValueError('Invalid option valuation inputs')
    root = iv * np.sqrt(np.maximum(t, 1e-16))
    d1 = (np.log(s / strike) + .5 * iv * iv * t) / root
    d2 = d1 - root
    modeled = s * ndtr(d1) - strike * ndtr(d2) if side == 'call' else strike * ndtr(-d2) - s * ndtr(-d1)
    intrinsic = np.maximum(s - strike, 0) if side == 'call' else np.maximum(strike - s, 0)
    return np.maximum(np.where(t == 0, intrinsic, modeled), 0.)


def rounded_premium(price, contract, markup):
    raw = price * (1 + markup)
    tick = float(contract['tick_size'])
    for step in contract.get('tick_size_steps', []):
        if raw > step['above_price']:
            tick = max(tick, float(step['tick_size']))
    if not np.isfinite([raw, tick]).all() or raw <= 0 or tick <= 0:
        raise ValueError('Invalid quoted premium or tick')
    return math.ceil(raw / tick - 1e-12) * tick


def entry_cost(row, spot, config):
    """Per-unit owned-cash debit and additive cost breakdown; fees not doubled."""
    contract, trade = row['contract'], row['trade']
    price = rounded_premium(float(trade['price']), contract, config.premium_markup)
    if config.family == 'inverse':
        base = float(trade['price']) * spot
        premium = price * spot
        option_fee = min(config.option_fee_rate, config.fee_cap * price) * spot
        impact = (premium + option_fee) * config.conversion_impact
        conversion_fee = (premium + option_fee + impact) * config.conversion_fee
    else:
        base, premium = float(trade['price']), price
        option_fee = min(config.option_fee_rate * float(trade['index_price']), config.fee_cap * price)
        impact = 0.
        conversion_fee = (premium + option_fee) * config.conversion_fee
    parts = {'raw_premium': base, 'premium_markup_cost': premium - base,
             'option_fee': option_fee, 'conversion_impact': impact, 'conversion_fee': conversion_fee}
    lot_floor = (.1 if row['asset'] == 'BTC' else 1.) if config.family == 'inverse' else (.01 if row['asset'] == 'BTC' else .1)
    lot = max(lot_floor, float(contract['min_trade_amount']))
    return sum(parts.values()), lot, parts


def expiry_receipt(row, spot, config):
    delivery, strike = float(row['official_delivery_price']), float(row['contract']['strike'])
    side = row['contract']['option_type']
    intrinsic = max(delivery - strike, 0.) if side == 'call' else max(strike - delivery, 0.)
    if config.family == 'inverse':
        gross = intrinsic / delivery * spot
        option_fee = min(config.delivery_fee_rate, config.fee_cap * intrinsic / delivery) * spot
        impact = (gross - option_fee) * config.conversion_impact
        conversion_fee = (gross - option_fee - impact) * config.conversion_fee
    else:
        gross = intrinsic
        option_fee = min(config.delivery_fee_rate * delivery, config.fee_cap * intrinsic)
        impact = 0.
        conversion_fee = (gross - option_fee) * config.conversion_fee
    return gross - option_fee - impact - conversion_fee, {
        'gross_payoff': gross, 'option_fee': option_fee, 'conversion_impact': impact,
        'conversion_fee': conversion_fee}


def replay_option_sleeve(markets, observations, config, *, start, end, initial_cash):
    """Hold paid options to official delivery; mark held positions with entry IV."""
    if not config.options_enabled:
        raise ValueError('Paid-option replay requires explicit offline opt-in')
    full = next(iter(markets.values())).index.tz_convert('UTC').as_unit('ns')
    begin, finish = pd.Timestamp(start), pd.Timestamp(end)
    index = full[(full >= begin) & (full < finish)]
    if index.empty or not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError('Invalid option cash or evaluation dates')
    schedule = {}
    for r in observations:
        if r['family'] != config.family or r['policy'] != config.policy or r['status'] == 'inactive_signal':
            continue
        if 'delay_bars' in r and r['delay_bars'] != config.delay_bars:
            continue
        when = pd.Timestamp(r.get('execution_time', r['entry']))
        if 'execution_time' not in r:
            when += pd.Timedelta(hours=4 * config.delay_bars)
        if begin <= when < finish:
            schedule.setdefault(when, []).append(r)
    cash, peak = float(initial_cash), float(initial_cash)
    arrays = {s: f.reindex(index)[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
              for s, f in markets.items()}
    positions, ledger, values = [], [], []
    diagnostics = dict(eligible_signals=0, no_listed_contract=0, no_recent_trade=0,
        invalid_trade_model_inputs=0, minimum_lot_skips=0, buys=0, settlements=0,
        maximum_monthly_budget_fraction=0., maximum_trade_age_seconds=0.)
    buy_positions = []
    for i, ts in enumerate(index):
        open_nav = cash
        for pos in positions:
            r = pos['row']
            offset = i - pos['offset']
            if offset < len(pos['pricing']['open']):
                open_nav += pos['qty'] * pos['pricing']['open'][offset]
            else:
                s, k = arrays[r['asset'] + 'USDT'][i, 0], r['contract']['strike']
                open_nav += pos['qty'] * max(s - k if r['contract']['option_type'] == 'call' else k - s, 0.)
        retained = []
        for pos in positions:
            if pos['expiry'] > ts:
                retained.append(pos)
                continue
            r, q = pos['row'], pos['qty']
            receipt, costs = expiry_receipt(r, arrays[r['asset'] + 'USDT'][i, 0], config)
            change = q * receipt
            cash += change
            ledger.append({'timestamp': ts, 'event': 'settle', 'instrument': pos['instrument'], 'qty': q,
                'cash_flow': change, 'cash_after': cash, **{k: q * v for k, v in costs.items()}})
            diagnostics['settlements'] += 1
        positions = retained
        entries = schedule.get(ts, [])
        if entries:
            if positions:
                raise ValueError('Monthly expiry schedule unexpectedly overlaps')
            budget = cash * config.monthly_cash_budget
            allocation, spent = budget / len(entries), 0.
            for r in entries:
                diagnostics['eligible_signals'] += 1
                if r['status'] != 'observed_trade':
                    diagnostics[r['status']] += 1
                    continue
                trade = r['trade']
                if (trade['timestamp'] >= ts.value // 1_000_000
                        or trade['timestamp'] < (ts - pd.Timedelta(hours=4)).value // 1_000_000
                        or r['contract']['creation_timestamp'] > pd.Timestamp(r['entry']).value // 1_000_000):
                    raise ValueError('Future or stale option trade/contract in entry')
                spot = arrays[r['asset'] + 'USDT'][i, 0]
                unit, lot, costs = entry_cost(r, spot, config)
                qty = math.floor(allocation / (unit * lot) + 1e-12) * lot
                if qty <= 0:
                    diagnostics['minimum_lot_skips'] += 1
                    continue
                debit = unit * qty
                if debit > allocation + 1e-8 or debit > cash + 1e-8:
                    raise ValueError('Option premium and fees exceed owned cash budget')
                cash -= debit
                spent += debit
                expiry = pd.Timestamp(r['expiry'])
                if expiry >= finish:
                    raise ValueError('Study ends with an unsettled monthly option')
                pos = {'row': r, 'qty': qty, 'expiry': expiry, 'entry': ts,
                       'instrument': r['contract']['instrument_name'], 'debit': debit, 'offset': i}
                stop = index.searchsorted(expiry)
                bars = arrays[r['asset'] + 'USDT'][i:stop]
                t0 = np.maximum((expiry.value - index[i:stop].asi8) / 1e9, 0.) / (365.25 * 86400)
                t1 = np.maximum(t0 - 4 / (365.25 * 24), 0.)
                iv = r['trade']['iv'] / 100 * config.iv_mark_multiplier
                strike, side = r['contract']['strike'], r['contract']['option_type']
                pos['pricing'] = {'open': option_value(bars[:, 0], strike, t0, iv, side),
                    'close': option_value(bars[:, 3], strike, t1, iv, side),
                    'high': option_value(bars[:, 1 if side == 'call' else 2], strike, t0, iv, side),
                    'low': option_value(bars[:, 2 if side == 'call' else 1], strike, t1, iv, side)}
                positions.append(pos)
                buy_positions.append(pos)
                ledger.append({'timestamp': ts, 'event': 'buy', 'instrument': pos['instrument'], 'qty': qty,
                    'cash_flow': -debit, 'cash_after': cash, **{k: qty * v for k, v in costs.items()}})
                diagnostics['buys'] += 1
                diagnostics['maximum_trade_age_seconds'] = max(diagnostics['maximum_trade_age_seconds'], r['trade_age_seconds'])
            if budget > 0:
                diagnostics['maximum_monthly_budget_fraction'] = max(diagnostics['maximum_monthly_budget_fraction'], spent / (budget / config.monthly_cash_budget))
        mark = high = low = 0.
        for pos in positions:
            qty, offset = pos['qty'], i - pos['offset']
            mark += qty * pos['pricing']['close'][offset]
            high += qty * pos['pricing']['high'][offset]
            low += qty * pos['pricing']['low'][offset]
        nav = cash + mark
        if cash < -1e-8 or not np.isfinite([cash, mark, high, low]).all():
            raise ValueError('Option owned-cash invariant failed')
        peak = max(peak, nav)
        values.append({'timestamp': ts, 'equity': nav, 'cash': cash, 'option_value': mark, 'open_equity': open_nav,
                       'high_bound': cash + high, 'low_bound': cash + low,
                       'drawdown': nav / peak - 1, 'held_contracts': sum(p['qty'] for p in positions)})
    if positions:
        raise ValueError('Terminal open option position')
    eq = pd.DataFrame(values)
    trades = pd.DataFrame(ledger)
    accounting = reconcile_option_sleeve(markets, eq, trades, buy_positions, config, initial_cash)
    summaries = []
    for pos in buy_positions:
        r = pos['row']
        receipt, settlement_costs = expiry_receipt(r, float(markets[r['asset'] + 'USDT'].loc[pos['expiry'], 'open']), config)
        _, _, purchase_costs = entry_cost(r, float(markets[r['asset'] + 'USDT'].loc[pos['entry'], 'open']), config)
        total = pos['qty'] * receipt
        summaries.append({'instrument': pos['instrument'], 'entry': str(pos['entry']), 'expiry': str(pos['expiry']),
            'qty': pos['qty'], 'paid_including_costs': pos['debit'], 'net_receipt': total,
            'net_pnl': total - pos['debit'], 'payoff_multiple': total / pos['debit'],
            'price_proxy_trade_id': r['trade'].get('trade_id'),
            'price_proxy_timestamp_ms': r['trade']['timestamp'],
            'raw_premium_per_unit': r['trade']['price'],
            'price_proxy_index': r['trade']['index_price'],
            'price_proxy_iv_percent': r['trade']['iv'],
            'strike': r['contract']['strike'],
            'official_delivery_price': r['official_delivery_price'],
            'premium_currency': r['asset'] if config.family == 'inverse' else 'USDC',
            'trade_source_sha256': r.get('trade_source', {}).get('sha256'),
            **{'buy_' + k: pos['qty'] * v for k, v in purchase_costs.items()},
            **{'settlement_' + k: pos['qty'] * v for k, v in settlement_costs.items()}})
    pnl = [r['net_pnl'] for r in summaries]
    stats = {**diagnostics, 'initial_cash': initial_cash, 'final_cash': cash,
        'net_pnl': cash - initial_cash, 'total_return': cash / initial_cash - 1,
        'minimum_cash': float(eq.cash.min()), 'max_model_close_drawdown': float(eq.drawdown.min()),
        'winning_options': int(sum(p > 0 for p in pnl)), 'win_rate': float(sum(p > 0 for p in pnl) / len(pnl)) if pnl else None,
        'maximum_payoff_multiple': max((r['payoff_multiple'] for r in summaries), default=None),
        'closed_options': summaries, 'accounting': accounting,
        'costs': {c: float(trades[c].fillna(0).sum()) if c in trades else 0.
            for c in ('option_fee', 'conversion_fee', 'conversion_impact', 'premium_markup_cost')},
        'fees_are_modeled': True, 'held_marks_are_synthetic': True}
    return eq, trades, stats


def reconcile_option_sleeve(markets, equity, trades, positions, config, initial_cash):
    """Independently rebuild cash/contract quantities and marks at every 4h close."""
    index = pd.DatetimeIndex(equity.timestamp).as_unit('ns')
    flows = pd.Series(0., index=index)
    if not trades.empty:
        for r in trades.itertuples(index=False):
            flows.loc[r.timestamp] += r.cash_flow
    cash = initial_cash + flows.cumsum()
    expected, held = cash.to_numpy().copy(), np.zeros(len(index))
    flow_error = 0.
    for pos in positions:
        r = pos['row']
        legs = trades[trades.instrument == pos['instrument']]
        buys, settlements = legs[legs.event == 'buy'], legs[legs.event == 'settle']
        if (len(buys) != 1 or len(settlements) != 1 or buys.timestamp.iloc[0] != pos['entry']
                or settlements.timestamp.iloc[0] != pos['expiry']):
            raise ValueError('Option ledger contract lifecycle does not reconcile')
        q = float(buys.qty.iloc[0])
        if q != float(settlements.qty.iloc[0]) or q != pos['qty']:
            raise ValueError('Option ledger contract quantities do not reconcile')
        debit, _, _ = entry_cost(r, float(markets[r['asset'] + 'USDT'].loc[pos['entry'], 'open']), config)
        receipt, _ = expiry_receipt(r, float(markets[r['asset'] + 'USDT'].loc[pos['expiry'], 'open']), config)
        flow_error = max(flow_error, abs(float(buys.cash_flow.iloc[0]) + q * debit),
                         abs(float(settlements.cash_flow.iloc[0]) - q * receipt))
        active = (index >= pos['entry']) & (index < pos['expiry'])
        dates = index[active]
        spot = markets[r['asset'] + 'USDT'].close.reindex(dates).to_numpy()
        years = np.maximum((pos['expiry'].value - dates.asi8) / 1e9 - 4 * 3600, 0.) / (365.25 * 86400)
        expected[active] += q * option_value(spot, r['contract']['strike'], years,
            r['trade']['iv'] / 100 * config.iv_mark_multiplier, r['contract']['option_type'])
        held[active] += q
        if q <= 0:
            raise ValueError('Short option detected by independent ledger')
    cash_error = float(np.max(np.abs(cash.to_numpy() - equity.cash.to_numpy())))
    nav_error = float(np.max(np.abs(expected - equity.equity.to_numpy())))
    qty_error = float(np.max(np.abs(held - equity.held_contracts.to_numpy())))
    if max(cash_error, nav_error, qty_error, flow_error) > 1e-7 or cash.min() < -1e-7:
        raise ValueError('Option ledger does not reconcile')
    return {'reconciled': True, 'max_cash_error': cash_error, 'max_equity_error': nav_error,
            'max_contract_error': qty_error, 'max_contract_cash_flow_error': flow_error}


def combine_paths(spot, option, initial, markets=None, initial_spot=None):
    if not spot.timestamp.equals(option.timestamp):
        raise ValueError('Spot/option funding periods do not match')
    eq = spot[['timestamp']].copy()
    eq['spot_equity'] = spot.equity
    eq['option_equity'] = option.equity
    eq['equity'] = spot.equity + option.equity
    eq['high_bound'] = spot.high_bound + option.high_bound
    eq['low_bound'] = spot.low_bound + option.low_bound
    eq['cash'] = spot.usdt + option.cash
    eq['drawdown'] = eq.equity / np.maximum(initial, eq.equity.cummax()) - 1
    opening = eq.equity.copy()
    if markets is not None:
        spot_open = spot.usdt.shift(1, fill_value=initial_spot).to_numpy()
        dates = pd.DatetimeIndex(spot.timestamp)
        for symbol in markets:
            spot_open += spot[f'base_{symbol}'].shift(1, fill_value=0.).to_numpy() * markets[symbol].open.reindex(dates).to_numpy()
        opening = spot_open + option.open_equity.to_numpy()
    highest = np.maximum(eq.high_bound.to_numpy(), np.asarray(opening))
    peaks = np.maximum(initial, np.maximum.accumulate(highest))
    eq['model_intraday_drawdown_bound'] = np.minimum.reduce([
        eq.equity.to_numpy(), eq.low_bound.to_numpy(), np.asarray(opening)]) / peaks - 1
    daily = eq.set_index('timestamp').equity.resample('1D').last()
    final = float(eq.equity.iloc[-1])
    years = len(eq) / (6 * 365.25)
    summary = {'initial_equity': initial, 'final_equity': final, 'net_pnl': final - initial,
        'total_return': final / initial - 1, 'cagr': (final / initial) ** (1 / years) - 1,
        'maxDD': float(eq.drawdown.min()),
        'max_intraday_drawdown_bound': float(eq.model_intraday_drawdown_bound.min()),
        'minimum_cash': float(eq.cash.min()), 'rolling_gains': rolling_gains(daily, initial),
        'drawdown_uses_synthetic_option_marks': True}
    return eq, summary


def run_optional_long_options(markets, completed_targets, observations, *, config=LongOptionsConfig(),
                              start=None, end=None, initial_usdt=1000., spot_costs=None):
    """Research on/off switch. Disabled mode preserves the existing spot engine."""
    costs = {} if spot_costs is None else spot_costs
    if not config.options_enabled:
        return run_intraday_spot(markets, completed_targets, start=start, end=end,
            initial_usdt=initial_usdt, daily_control=True, signal_delay_bars=config.delay_bars, **costs)
    index = next(iter(markets.values())).index
    start = index[0] if start is None else pd.Timestamp(start)
    end = index[-1] + pd.Timedelta(hours=4) if end is None else pd.Timestamp(end)
    spot, fills, spot_summary = run_intraday_spot(markets, completed_targets, start=start, end=end,
        initial_usdt=initial_usdt * (1 - config.sleeve_fraction), daily_control=True,
        signal_delay_bars=config.delay_bars, **costs)
    option, trades, option_summary = replay_option_sleeve(markets, observations, config,
        start=start, end=end, initial_cash=initial_usdt * config.sleeve_fraction)
    eq, summary = combine_paths(spot, option, initial_usdt, markets, initial_usdt * (1 - config.sleeve_fraction))
    summary.update(spot=spot_summary, options=option_summary, live_eligible=False,
                   options_enabled=True, options_execution_is_modeled=True)
    return eq, {'spot': fills, 'options': trades}, summary
