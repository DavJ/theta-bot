"""Conditional owned-cash execution estimates from public spot depth.

No exchange adapter, credentials or strategy selection. A REST snapshot is not
a reserved executable quote; every result remains a conditional simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from math import lcm


ZERO = Decimal('0')
ONE = Decimal('1')
BPS = Decimal('10000')
SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'BTCUSDC', 'ETHUSDC', 'BNBUSDC')


def number(value, *, positive=False):
    if isinstance(value, (bool, float)):
        raise ValueError('Exact decimal strings or integers required')
    try:
        result = Decimal(value)
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise ValueError('Invalid decimal value') from exc
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise ValueError('Finite nonnegative decimal required')
    return result


def floor_step(value, step):
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step


def common_step(steps):
    steps = [s for s in steps if s > 0]
    if not steps:
        raise ValueError('A positive quantity step is required')
    scale = Decimal(10) ** max(0, -min(s.as_tuple().exponent for s in steps))
    return Decimal(lcm(*(int(s * scale) for s in steps))) / scale


@dataclass(frozen=True)
class Rules:
    symbol: str
    base: str
    quote: str
    step: Decimal
    minimum_qty: Decimal
    maximum_qty: Decimal
    minimum_notional: Decimal
    maximum_notional: Decimal | None
    price_tick: Decimal
    fee_quantum: Decimal

    @classmethod
    def from_info(cls, payload, symbol):
        rows = payload.get('symbols', [])
        if (symbol not in SYMBOLS or not isinstance(rows,list) or len(rows) != 1
                or not isinstance(rows[0],dict) or rows[0].get('symbol') != symbol):
            raise ValueError('Exactly the requested frozen spot symbol is required')
        info = rows[0]
        base, quote = info.get('baseAsset'), info.get('quoteAsset')
        if (base not in ('BTC','ETH','BNB') or quote not in ('USDT','USDC')
                or base + quote != symbol or info.get('status') != 'TRADING'
                or info.get('isSpotTradingAllowed') is not True or 'MARKET' not in info.get('orderTypes', [])):
            raise ValueError('Verified public trading spot market required')
        filters = info.get('filters', [])
        if not isinstance(filters,list) or any(not isinstance(f,dict) or 'filterType' not in f for f in filters):
            raise ValueError('Explicit public filters required')
        by_type = {f['filterType']: f for f in filters}
        if len(by_type) != len(filters) or not {'LOT_SIZE','MARKET_LOT_SIZE','PRICE_FILTER'} <= by_type.keys():
            raise ValueError('Unique public quantity and price filters required')
        lots = [by_type[t] for t in ('LOT_SIZE','MARKET_LOT_SIZE')]
        steps = [number(f['stepSize']) for f in lots]
        minima = [number(f['minQty']) for f in lots]
        maxima = [number(f['maxQty']) for f in lots]
        maxima = [v for v in maxima if v > 0]
        if not maxima:
            raise ValueError('Finite positive market quantity cap required')
        minimum_notional, maximum_notional = Decimal('10'), None
        seen_notional = False
        for kind in ('MIN_NOTIONAL','NOTIONAL'):
            if kind not in by_type:
                continue
            seen_notional = True
            f = by_type[kind]
            apply_min = f.get('applyToMarket') if kind == 'MIN_NOTIONAL' else f.get('applyMinToMarket')
            if not isinstance(apply_min, bool):
                raise ValueError('Explicit market notional flags required')
            if apply_min:
                minimum_notional = max(minimum_notional, number(f['minNotional']))
            if kind == 'NOTIONAL':
                if not isinstance(f.get('applyMaxToMarket'), bool):
                    raise ValueError('Explicit market maximum-notional flag required')
                if f['applyMaxToMarket']:
                    maximum_notional = number(f['maxNotional'], positive=True)
        precision = info.get('quoteCommissionPrecision')
        if (not seen_notional or type(precision) is not int or not 0 <= precision <= 18
                or max(minima) > min(maxima)):
            raise ValueError('Consistent notional, quantity and quote-fee precision required')
        return cls(symbol, base, quote, common_step(steps), max(minima), min(maxima),
            minimum_notional, maximum_notional,
            number(by_type['PRICE_FILTER']['tickSize'], positive=True), Decimal(10) ** -precision)


@dataclass(frozen=True)
class Book:
    symbol: str
    update_id: int
    bids: tuple
    asks: tuple

    @classmethod
    def from_depth(cls, payload, rules):
        update = payload.get('lastUpdateId')
        if type(update) is not int or update < 0:
            raise ValueError('Integer source update ID required')
        sides = []
        for name in ('bids','asks'):
            rows = payload.get(name)
            if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
                raise ValueError('Nonempty bounded public depth required')
            levels = []
            for row in rows:
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError('Exact price/quantity depth pairs required')
                price, qty = (number(v, positive=True) for v in row)
                if price % rules.price_tick != 0:
                    raise ValueError('Book price violates the public tick size')
                levels.append((price, qty))
            prices = [p for p,q in levels]
            if any((a <= b if name == 'bids' else a >= b) for a,b in zip(prices, prices[1:])):
                raise ValueError('Strictly ordered unique price levels required')
            sides.append(tuple(levels))
        if sides[0][0][0] >= sides[1][0][0]:
            raise ValueError('Crossed or locked book rejected')
        return cls(rules.symbol, update, *sides)

    @property
    def mid(self):
        return (self.bids[0][0] + self.asks[0][0]) / 2

    @property
    def spread_bps(self):
        return (self.asks[0][0] - self.bids[0][0]) / self.mid * BPS


def walk(book, side, quantity):
    """Quote the full quantity or reject; never invent unseen liquidity."""
    quantity = number(quantity, positive=True)
    if side not in ('buy','sell'):
        raise ValueError('Owned spot buy or sell required')
    remaining, notional, used = quantity, ZERO, 0
    for price, available in book.asks if side == 'buy' else book.bids:
        take = min(remaining, available)
        notional += take * price
        remaining -= take
        used += 1
        if remaining == 0:
            return notional, used
    raise ValueError('insufficient_visible_depth')


def fee(notional, rate, quantum):
    return (notional * rate).quantize(quantum, rounding=ROUND_CEILING)


def simulate_fill(book, rules, side, quantity, cash, inventory, fee_rate, reserve_bps='0'):
    """FOK-style conditional fill; balances are simulated, fees charged in quote.

    Quantity filters and modeled VWAP notionals are checked. Exchange reference
    prices, execution rules and account-specific filters remain unverified.
    """
    quantity = number(quantity, positive=True)
    cash, inventory, fee_rate, reserve = map(number, (cash, inventory, fee_rate, reserve_bps))
    if rules.symbol != book.symbol or fee_rate >= Decimal('.5') or reserve >= BPS:
        raise ValueError('Invalid market, hypothetical fee or adverse reserve')
    if quantity % rules.step or not rules.minimum_qty <= quantity <= rules.maximum_qty:
        raise ValueError('quantity_filter')
    if side == 'sell' and quantity > inventory:
        raise ValueError('insufficient_owned_inventory')
    raw, levels = walk(book, side, quantity)
    multiplier = ONE + reserve / BPS if side == 'buy' else ONE - reserve / BPS
    notional = raw * multiplier
    if (notional < rules.minimum_notional
            or rules.maximum_notional is not None and notional > rules.maximum_notional):
        raise ValueError('modeled_notional_filter')
    commission = fee(notional, fee_rate, rules.fee_quantum)
    if side == 'buy':
        if notional + commission > cash:
            raise ValueError('insufficient_owned_cash')
        after_cash, after_base = cash - notional - commission, inventory + quantity
    else:
        after_cash, after_base = cash + notional - commission, inventory - quantity
    if min(after_cash, after_base) < 0:
        raise ValueError('Negative owned balance')
    return {'side': side, 'quantity': quantity, 'notional': notional, 'fee_quote': commission,
        'raw_book_notional': raw, 'average_price': notional / quantity, 'levels_used': levels,
        'cash_after': after_cash, 'inventory_after': after_base,
        'exchange_filters_fully_verified': False, 'observed_exchange_fill': False}


def round_trip_probe(book, rules, budget, fee_rate, reserve_bps='0', capital='10000'):
    """Independently funded buy then sell on one static book, an execution-cost probe."""
    with localcontext() as ctx:
        ctx.prec = 40
        budget, rate, reserve, capital = map(number, (budget, fee_rate, reserve_bps, capital))
        if budget <= 0 or budget > capital * Decimal('.25') or rate >= Decimal('.5') or reserve >= BPS:
            raise ValueError('Probe budget must be positive and at most 25% of owned cash')
        # Binary search whole quantity steps with the fee rounded conservatively.
        ceiling = budget / (book.asks[0][0] * (ONE + reserve/BPS) * (ONE + rate))
        lo, hi = 0, int(floor_step(ceiling, rules.step) / rules.step)
        if hi == 0 or Decimal(hi) * rules.step > rules.maximum_qty:
            raise ValueError('quantity_filter')
        # Refuse a budget whose initial top-ask-sized order exceeds known depth;
        # silently shrinking to the available book would conceal missing liquidity.
        walk(book, 'buy', Decimal(hi) * rules.step)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            raw, _ = walk(book, 'buy', Decimal(mid) * rules.step)
            amount = raw * (ONE + reserve/BPS)
            if amount + fee(amount, rate, rules.fee_quantum) <= budget:
                lo = mid
            else:
                hi = mid - 1
        qty = Decimal(lo) * rules.step
        buy = simulate_fill(book, rules, 'buy', qty, capital, ZERO, rate, reserve)
        sell = simulate_fill(book, rules, 'sell', qty, buy['cash_after'], buy['inventory_after'], rate, reserve)
        spent, received = buy['notional'] + buy['fee_quote'], sell['notional'] - sell['fee_quote']
        loss = spent - received
        assert spent <= budget and sell['inventory_after'] == 0 and loss >= 0
        return {'budget_including_fee': budget, 'funded_quote': capital, 'base_quantity': qty,
            'buy': buy, 'sell': sell, 'quote_spent': spent, 'quote_received': received,
            'round_trip_cost_quote': loss, 'round_trip_cost_bps': loss / spent * BPS,
            'cash_after_roundtrip': sell['cash_after'], 'spread_bps': book.spread_bps,
            'buy_depth_impact_bps': (buy['raw_book_notional']/qty/book.asks[0][0]-ONE)*BPS,
            'sell_depth_impact_bps': (ONE-sell['raw_book_notional']/qty/book.bids[0][0])*BPS,
            'all_balances_nonnegative': True, 'probe_kind': 'conditional_same_book_round_trip',
            'exchange_filters_fully_verified': False, 'observed_exchange_fill': False}


def json_values(value):
    if isinstance(value, Decimal):
        return format(value, 'f')
    if isinstance(value, dict):
        return {k:json_values(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [json_values(v) for v in value]
    return value
