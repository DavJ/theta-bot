"""Causal publication, fixed allocations and evidence integrity checks."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.research_allocation_benchmarks import freeze, read_freeze
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.execution_risk import completed_risk_covariances
from spot_bot.portfolio.risk import PortfolioRisk
from theta_bot_averaging.allocation_benchmarks import (
    REFERENCES, allocation_targets, realized_daily_volatility,
)


def markets(days=94):
    index = pd.date_range('2022-01-01', periods=days*6, freq='4h', tz='UTC')
    rng = np.random.default_rng(8921)
    result = {}
    for i, symbol in enumerate(('BTCUSDT','ETHUSDT','BNBUSDT')):
        close = 100 * np.exp(np.cumsum(rng.normal(.0001,.003*(i+1),len(index))))
        result[symbol] = pd.DataFrame({'open': close, 'high': close*1.01,
            'low': close*.99, 'close': close, 'volume': 100.}, index=index)
    return result


def test_fixed_allocations_caps_and_inverse_volatility_match_direct_window():
    source = markets()
    target = allocation_targets(source)
    assert tuple(target) == REFERENCES
    assert target['cash'].eq(0).all().all()
    for name, budget in (('equal_25',.25),('equal_50',.5)):
        assert target[name].iloc[:5].eq(0).all().all()
        np.testing.assert_allclose(target[name].iloc[5:], budget/3)
    for frame in target.values():
        assert frame.ge(0).all().all()
        assert frame.le(.25+1e-12).all().all()
        assert frame.sum(axis=1).le(.5+1e-12).all()
    inverse = target['inverse_vol_50']
    assert inverse.iloc[:60*6+5].eq(0).all().all()
    # Independent capped water filling, with no momentum or engine helper call.
    close = pd.DataFrame({s: f.close for s,f in source.items()}).resample('1D').last()
    r = close.to_numpy()[-61:]
    sigma = np.maximum((r[1:]/r[:-1]-1).std(axis=0,ddof=0),.0001)
    lo, hi = 0., 1.
    for _ in range(100):
        mid=(lo+hi)/2
        if np.minimum(.25,mid/sigma).sum() > .5: hi=mid
        else: lo=mid
    np.testing.assert_allclose(inverse.iloc[-1],np.minimum(.25,lo/sigma),atol=1e-12)


@pytest.mark.parametrize('cut',[1,5,6,60*6+4,60*6+6,87*6+3])
def test_prefix_targets_ignore_future_and_incomplete_day(cut):
    source=markets()
    prefix={s:f.iloc[:cut] for s,f in source.items()}
    full=allocation_targets(source)
    short=allocation_targets(prefix)
    changed={s:f.copy() for s,f in source.items()}
    for f in changed.values(): f.iloc[cut:,f.columns.get_loc('close')]*=5
    alternate=allocation_targets(changed)
    for name in REFERENCES:
        pd.testing.assert_frame_equal(short[name],full[name].iloc[:cut])
        pd.testing.assert_frame_equal(short[name],alternate[name].iloc[:cut])


def test_incomplete_first_day_does_not_publish():
    source={s:f.iloc[1:] for s,f in markets().items()}
    t=allocation_targets(source)
    assert t['equal_50'].iloc[:10].eq(0).all().all()
    np.testing.assert_allclose(t['equal_50'].iloc[10],.5/3)


@pytest.mark.parametrize('bad',['nan','negative','missing','unaligned','naive','off_clock','wrong_assets'])
def test_invalid_source_fails(bad):
    source=markets()
    first=next(iter(source))
    if bad=='nan': source[first].iloc[10,3]=np.nan
    elif bad=='negative': source[first].iloc[10,3]=-1
    elif bad=='missing': source={s:f.drop(f.index[10]) for s,f in source.items()}
    elif bad=='unaligned': source[first]=source[first].iloc[:-1]
    elif bad=='naive': source={s:f.tz_localize(None) for s,f in source.items()}
    elif bad=='off_clock':
        for f in source.values(): f.index=f.index+pd.Timedelta('1h')
    else: source.pop(first)
    with pytest.raises(ValueError): allocation_targets(source)


@pytest.mark.parametrize('policy',REFERENCES)
def test_guarded_references_preserve_owned_funding_and_warmup(policy):
    source=markets()
    risk=PortfolioRisk(volatility_target=.15)
    equity,trades,row=run_intraday_spot(source,allocation_targets(source)[policy],
        daily_control=True,rebalance_destination='band_edge',risk=risk,
        completed_covariances=completed_risk_covariances(source,risk))
    assert reconcile_spot_path(source,equity,trades,1000)['reconciled']
    assert equity.usdt.min()>=0
    assert equity.filter(like='base_').min().min()>=0
    if policy=='cash':
        assert trades.empty and row['net_pnl']==0
        assert realized_daily_volatility(equity,1000)==0
    else:
        assert not trades.empty
        assert trades.timestamp.min()>=pd.Timestamp('2022-03-03',tz='UTC')
        buy=trades.loc[trades.side=='buy']
        assert buy.scheduled_decision.all()
        assert buy.timestamp.dt.day_of_week.eq(0).all()


def test_realized_volatility_includes_the_initial_funded_return():
    equity=pd.DataFrame({'timestamp':pd.date_range('2022-01-01',periods=3,freq='1D',tz='UTC'),
                         'equity':[90.,99.,99.]})
    expected=np.std([-.1,.1,0.],ddof=0)*np.sqrt(365)
    assert realized_daily_volatility(equity,100)==pytest.approx(expected)


@pytest.mark.parametrize('field',['settings','runtime','live_eligible','source_sha256'])
def test_freeze_rejects_tampering_even_with_rehashed_body(tmp_path,field):
    import hashlib
    from scripts.research_allocation_benchmarks import canonical
    manifest={'datasets':{},'source':'fixture'}
    proof=freeze(manifest,tmp_path)
    assert read_freeze(tmp_path,manifest)==proof
    with pytest.raises(FileExistsError): freeze(manifest,tmp_path)
    with pytest.raises(ValueError): read_freeze(tmp_path,{'source':'changed'})
    proof[field]=True if field=='live_eligible' else {}
    proof['protocol_sha256']=hashlib.sha256(canonical({k:v for k,v in proof.items()
        if k!='protocol_sha256'})).hexdigest()
    (tmp_path/'frozen_protocol.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError): read_freeze(tmp_path,manifest)
