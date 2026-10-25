"""Calendar diagnostics, owned-spot P&L attribution and fixed asset omissions."""
from __future__ import annotations

import numpy as np
import pandas as pd

SYMBOLS = ('BTCUSDT','ETHUSDT','BNBUSDT')
UNIVERSES = ('all_three',*(f'without_{s}' for s in SYMBOLS))


def omission_targets(original,universe):
    """At most two eligible names exhaust their unchanged 25% asset caps."""
    if (universe not in UNIVERSES or list(original)!=list(SYMBOLS)
            or not np.isfinite(original.to_numpy()).all() or (original<0).any().any()
            or (original>.25+1e-12).any().any() or (original.sum(axis=1)>.5+1e-12).any()):
        raise ValueError('Fixed universe and valid original capped targets required')
    if universe=='all_three':
        return original.copy()
    excluded=universe.removeprefix('without_')
    result=original.gt(0).astype(float)*.25
    result[excluded]=0.
    return result


def account_index(equity,capital):
    index=pd.DatetimeIndex(equity.timestamp)
    if (not np.isfinite(capital) or capital<=0 or index.empty or index.tz is None
            or not index.is_unique or not index.is_monotonic_increasing
            or not index.to_series().diff().dropna().eq(pd.Timedelta('4h')).all()
            or not np.isfinite(equity.equity.to_numpy()).all() or (equity.equity<=0).any()):
        raise ValueError('Positive NAV/capital and a complete aware 4h account required')
    index=index.tz_convert('UTC')
    if ((index.hour%4!=0).any() or (index.minute!=0).any() or (index.second!=0).any()
            or (index.microsecond!=0).any() or (index.nanosecond!=0).any()):
        raise ValueError('UTC four-hour labels required')
    return index


def calendar_diagnostics(equity,capital):
    index=account_index(equity,capital)
    finish=index[-1]+pd.Timedelta('4h')
    if (index[0].day!=1 or index[0].hour!=0 or finish.day!=1 or finish.hour!=0):
        raise ValueError('Complete calendar months required; partial months are not returns')
    nav=pd.Series(equity.equity.to_numpy(),index=index).resample('ME').last()
    previous=np.r_[capital,nav.to_numpy()[:-1]]
    returns=nav.to_numpy()/previous-1.
    months=[{'month':t.strftime('%Y-%m'),'starting_nav':float(start),'ending_nav':float(end),
             'return':float(value)} for t,start,end,value in zip(nav.index,previous,nav,returns)]
    years=[]
    for year in sorted(set(nav.index.year)):
        positions=np.flatnonzero(nav.index.year==year)
        years.append({'year':int(year),'first_month':months[positions[0]]['month'],
            'last_month':months[positions[-1]]['month'],'months':len(positions),
            'partial_year':bool(len(positions)!=12),'return':float(np.expm1(np.log1p(returns[positions]).sum()))})
    rolling={}
    for window in (12,24):
        values=[]
        for end in range(window,len(months)+1):
            value=float(nav.iloc[end-1]/previous[end-window]-1.)
            values.append({'first_month':months[end-window]['month'],'last_month':months[end-1]['month'],'return':value})
        numbers=np.array([v['return'] for v in values])
        rolling[str(window)]={'windows':values,'count':len(values),
            'negative_count':int((numbers<0).sum()),
            'negative_fraction':float((numbers<0).mean()) if len(numbers) else None,
            'worst_return':float(numbers.min()) if len(numbers) else None,
            'best_return':float(numbers.max()) if len(numbers) else None}
    logs=np.log1p(returns)
    order=np.argsort(-logs,kind='stable')[:3]
    positive=np.maximum(logs,0.)
    positive_total=float(positive.sum())
    peaks=np.maximum.accumulate(np.r_[capital,nav.to_numpy()])[1:]
    longest=current=0
    for underwater in nav.to_numpy()<peaks-1e-9:
        current=current+1 if underwater else 0
        longest=max(longest,current)
    return {'months':months,'calendar_years':years,'rolling_months':rolling,
        'best_month':months[int(np.argmax(returns))],'worst_month':months[int(np.argmin(returns))],
        'best_three_months':[months[int(i)] for i in order],
        'best_three_share_of_positive_log_return':float(positive[order].sum()/positive_total) if positive_total>0 else None,
        'mathematical_return_excluding_best_three_months':float(np.expm1(logs.sum()-logs[order].sum())),
        'longest_month_end_underwater_run':longest,'total_return':float(nav.iloc[-1]/capital-1.)}


def attribute_spot_pnl(markets,equity,trades,capital):
    """Additive USDT attribution; no standalone-asset or omitted-strategy claim."""
    index=account_index(equity,capital)
    if list(markets)!=list(SYMBOLS) or any(not f.index.equals(index) for f in markets.values()):
        raise ValueError('Full aligned fixed source markets required')
    opens=np.column_stack([f.open.to_numpy() for f in markets.values()])
    closes=np.column_stack([f.close.to_numpy() for f in markets.values()])
    qty=equity[[f'base_{s}' for s in SYMBOLS]].to_numpy(dtype=float)
    if (not np.isfinite(opens).all() or not np.isfinite(closes).all() or (opens<=0).any()
            or (closes<=0).any() or not np.isfinite(qty).all() or (qty<0).any()):
        raise ValueError('Positive prices and owned nonnegative inventory required')
    old_qty=np.vstack([np.zeros(len(SYMBOLS)),qty[:-1]])
    old_close=np.vstack([opens[0],closes[:-1]])
    gross=old_qty*(opens-old_close)+qty*(closes-opens)
    fees,impact=np.zeros_like(gross),np.zeros_like(gross)
    for row in trades.itertuples(index=False):
        if row.symbol not in SYMBOLS or not np.isfinite([row.fee,row.slippage]).all() or min(row.fee,row.slippage)<0:
            raise ValueError('Finite nonnegative fixed-asset costs required')
        i,j=index.get_loc(row.timestamp),SYMBOLS.index(row.symbol)
        fees[i,j]+=row.fee
        impact[i,j]+=row.slippage
    net=gross-fees-impact
    changes=np.diff(np.r_[capital,equity.equity.to_numpy()])
    error=float(np.max(np.abs(net.sum(axis=1)-changes)))
    terminal_error=float(abs(net.sum()-(equity.equity.iloc[-1]-capital)))
    if error>1e-7 or terminal_error>1e-7:
        raise ValueError(f'Asset attribution does not reconcile account NAV changes: {error}')
    rows={s:{'gross_holding_pnl':float(gross[:,j].sum()),'fees':float(fees[:,j].sum()),
        'impact':float(impact[:,j].sum()),'net_pnl_contribution':float(net[:,j].sum())}
        for j,s in enumerate(SYMBOLS)}
    return {'assets':rows,'total_net_pnl':float(net.sum()),'max_bar_residual':error,
            'terminal_residual':terminal_error,
            'units':'USDT; additive contribution to this shared-cash account'}
