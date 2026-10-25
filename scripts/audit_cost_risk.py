"""Independent NumPy ledger/covariance audit of the cost/risk CSV evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def independent_covariances(markets, lag):
    """Use direct return-window covariance; do not call the risk implementation."""
    index = next(iter(markets.values())).index
    close = pd.DataFrame({s: f.close for s, f in markets.items()})
    count = close.resample('1D').size()
    daily = close.resample('1D').last().loc[count.eq(6)]
    prices = daily.to_numpy()
    returns = prices[1:] / prices[:-1] - 1.
    matrices = np.full((len(daily), len(markets), len(markets)), np.nan)
    for i in range(60, len(daily)):
        sample = returns[i - 60:i]
        centered = sample - sample.mean(axis=0)
        cov = centered.T @ centered / 60
        matrices[i] = .75 * cov + .25 * np.diag(np.diag(cov))
    published = np.full((len(index), len(markets), len(markets)), np.nan)
    if len(daily):
        positions = (daily.index + pd.Timedelta('20h')).searchsorted(index, side='right') - 1
        ready = positions >= 0
        published[ready] = matrices[positions[ready]]
    result = np.full_like(published, np.nan)
    result[lag:] = published[:-lag]
    return result


def audit_ledger(markets, equity, trades, capital, costs, covariance=None):
    """Reconstruct every fill, open/close NAV and conservative drawdown bound."""
    symbols = list(markets)
    index = pd.DatetimeIndex(equity.timestamp)
    if not index.is_unique or not index.is_monotonic_increasing or index.empty:
        raise ValueError('Invalid equity clock')
    cash_changes = np.zeros(len(index))
    changes = np.zeros((len(index), len(symbols)))
    cash, inventory = float(capital), np.zeros(len(symbols))
    fee_error = impact_error = fill_state_error = 0.
    if not trades.timestamp.is_monotonic_increasing:
        raise ValueError('Trades are not chronological')
    for row in trades.itertuples(index=False):
        if row.side not in ('buy', 'sell') or row.qty <= 0 or row.symbol not in symbols:
            raise ValueError('Invalid owned-spot fill')
        i, j = index.get_loc(row.timestamp), symbols.index(row.symbol)
        mid = float(markets[row.symbol].loc[row.timestamp, 'open'])
        sign = 1 if row.side == 'buy' else -1
        price = mid * (1 + sign * (costs['slippage_bps'] + costs['spread_bps'] / 2) / 10000)
        fee = row.qty * price * costs['fee_rate']
        impact = row.qty * abs(price - mid)
        if (not np.isclose(row.price, price, rtol=1e-12, atol=1e-8)
                or not np.isclose(row.mid_price, mid, rtol=1e-12, atol=1e-8)
                or not np.isclose(row.notional, row.qty * price, rtol=1e-12, atol=1e-8)):
            raise ValueError('Fill price/notional disagrees with source open and fixed costs')
        fee_error = max(fee_error, abs(row.fee - fee))
        impact_error = max(impact_error, abs(row.slippage - impact))
        cash -= sign * row.qty * price + fee
        inventory[j] += sign * row.qty
        if cash < -1e-7 or inventory.min() < -1e-10:
            raise ValueError('Independent ledger detected borrowed cash or short inventory')
        fill_state_error = max(fill_state_error, abs(row.cash_after - cash), abs(row.base_after - inventory[j]))
        cash_changes[i] -= sign * row.qty * price + fee
        changes[i, j] += sign * row.qty
        if covariance is not None:
            if row.side == 'buy' and (row.risk_reduction or not row.scheduled_decision):
                raise ValueError('Risk reduction or unscheduled decision bought inventory')
    cash_path = capital + cash_changes.cumsum()
    quantity = changes.cumsum(axis=0)
    prices = {name: np.column_stack([f.loc[index, name] for f in markets.values()])
              for name in ('open', 'high', 'low', 'close')}
    nav = cash_path + np.einsum('ij,ij->i', quantity, prices['close'])
    open_post = cash_path + np.einsum('ij,ij->i', quantity, prices['open'])
    previous_cash = np.r_[capital, cash_path[:-1]]
    previous_qty = np.vstack([np.zeros(len(symbols)), quantity[:-1]])
    open_pre = previous_cash + np.einsum('ij,ij->i', previous_qty, prices['open'])
    high = cash_path + np.einsum('ij,ij->i', quantity, prices['high'])
    low = cash_path + np.einsum('ij,ij->i', quantity, prices['low'])
    decision_peak = np.maximum.accumulate(np.maximum(open_pre, np.r_[capital, high[:-1]]))
    bounds = np.minimum.reduce([open_pre / decision_peak - 1., open_post / decision_peak - 1.,
                                low / np.maximum(decision_peak, high) - 1.])
    errors = {'max_nav_error': float(np.max(np.abs(nav - equity.equity))),
        'max_cash_error': float(np.max(np.abs(cash_path - equity.usdt))),
        'max_inventory_error': float(np.max(np.abs(quantity - equity[[f'base_{s}' for s in symbols]].to_numpy()))),
        'max_price_bound_error': float(max(np.max(np.abs(high - equity.high_bound)), np.max(np.abs(low - equity.low_bound)))),
        'max_drawdown_error': float(np.max(np.abs(bounds - equity.intraday_drawdown_bound))),
        'max_fill_state_error': fill_state_error, 'max_fee_error': fee_error, 'max_impact_error': impact_error}
    if covariance is not None:
        w = quantity * prices['open'] / open_post[:, None]
        known = np.isfinite(covariance).all(axis=(1, 2))
        after_vol = np.sqrt(np.maximum(0., np.einsum('ij,ijk,ik->i', w[known], covariance[known], w[known]) * 365))
        errors['max_risk_volatility_error'] = float(np.max(np.abs(after_vol - equity.loc[known, 'estimated_annual_volatility_after_fills'])))
        if equity.loc[~known, 'estimated_annual_volatility_after_fills'].notna().any():
            raise ValueError('A volatility estimate appeared without completed history')
        before_w = previous_qty * prices['open'] / open_pre[:, None]
        vol = np.zeros(len(index))
        vol[known] = np.sqrt(np.maximum(0., np.einsum('ij,ijk,ik->i', before_w[known], covariance[known], before_w[known]) * 365))
        expected_scale = np.zeros(len(index))
        expected_scale[known] = np.minimum(1., .15 / np.maximum(vol[known], 1e-100))
        unscheduled = ~equity.scheduled_decision.to_numpy(dtype=bool)
        expected_exposure = before_w.sum(axis=1) * expected_scale
        errors['max_unscheduled_risk_scale_error'] = float(np.max(np.abs(expected_scale[unscheduled] - equity.loc[unscheduled, 'risk_scale'])))
        errors['max_unscheduled_planned_exposure_error'] = float(np.max(np.abs(expected_exposure[unscheduled] - equity.loc[unscheduled, 'planned_exposure'])))
        errors['max_decision_nav_error'] = float(np.max(np.abs(open_pre - equity.decision_nav)))
        errors['max_decision_peak_error'] = float(np.max(np.abs(decision_peak - equity.decision_peak_bound)))
    if max(errors.values()) > 1e-7:
        raise ValueError(f'Independent CSV audit residual exceeds tolerance: {errors}')
    return {'bars': len(index), 'fills': len(trades), 'reconciled': True, **errors}


def audit(data, out):
    report = json.loads((out / 'summary.json').read_text())
    hashes = json.loads((out / 'csv_hashes.json').read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
            raise ValueError('CSV evidence checksum mismatch')
    manifest = report['source_manifest']
    markets = {}
    for symbol, entry in manifest['datasets'].items():
        path = data / entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Source spot CSV changed before independent audit')
        frame = pd.read_csv(path)
        frame.timestamp = pd.to_datetime(frame.timestamp, utc=True)
        markets[symbol] = frame.set_index('timestamp')
    # Preserve the same explicit source-column order as the study, not sorted JSON order.
    markets = {s: markets[s] for s in ('BTCUSDT', 'ETHUSDT', 'BNBUSDT')}
    covariance = {lag: independent_covariances(markets, lag) for lag in (1, 2)}
    results = {}
    for capital in ('1000', '10000'):
        for scenario in report['scenarios']:
            costs = report['costs']['stress' if scenario in ('cost_stress', 'joint_stress') else 'nominal']
            lag = 2 if scenario in ('delayed', 'joint_stress') else 1
            for policy in report['policies']:
                stem = f'{capital}_{scenario}_{policy}'
                eq = pd.read_csv(out / f'{stem}_equity.csv')
                trades = pd.read_csv(out / f'{stem}_trades.csv')
                eq.timestamp = pd.to_datetime(eq.timestamp, utc=True)
                trades.timestamp = pd.to_datetime(trades.timestamp, utc=True)
                result = audit_ledger(markets, eq, trades, float(capital), costs,
                    covariance[lag] if policy in ('vol_guard', 'guarded_edge') else None)
                summary = report['results'][capital][scenario]['full_continuous'][policy]
                if (not np.isclose(eq.equity.iloc[-1], summary['final_equity'], atol=1e-7, rtol=1e-12)
                        or not np.isclose(eq.intraday_drawdown_bound.min(), summary['max_intraday_drawdown_bound'], atol=1e-12)
                        or not np.isclose(trades.fee.sum(), summary['fees_paid_total'], atol=1e-7, rtol=1e-12)
                        or not np.isclose(trades.slippage.sum(), summary['slippage_paid_total'], atol=1e-7, rtol=1e-12)):
                    raise ValueError('Exported accounts disagree with reported economics')
                results[stem] = result
    return {'audited_accounts': len(results), 'audited_bars': sum(x['bars'] for x in results.values()),
        'audited_fills': sum(x['fills'] for x in results.values()), 'csv_files_checked': len(hashes),
        'maximum_nav_error': max(x['max_nav_error'] for x in results.values()),
        'maximum_risk_volatility_error': max(x.get('max_risk_volatility_error', 0.) for x in results.values()),
        'accounts': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw/spot_flow'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/cost_risk_20261013'))
    args = parser.parse_args()
    result = audit(args.data, args.out)
    (args.out / 'independent_csv_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'accounts'}, indent=2))


if __name__ == '__main__': main()
