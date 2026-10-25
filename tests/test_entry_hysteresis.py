"""Entry state, completed clocks, unchanged controls and frozen-source integrity."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from theta_bot_averaging.entry_hysteresis import (
    ENTRY_HURDLE,VARIANTS,eligibility_states,completed_entry_targets,
)
from scripts.research_entry_hysteresis import freeze,read_freeze,canonical
from spot_bot.portfolio.momentum_refinement import momentum_refinement_targets
from spot_bot.portfolio.execution_risk import completed_risk_covariances
from spot_bot.portfolio.risk import PortfolioRisk
from spot_bot.backtest.intraday import run_intraday_spot,reconcile_spot_path


def markets(days=134):
    rng=np.random.default_rng(8461)
    index=pd.date_range('2022-01-01',periods=days*6,freq='4h',tz='UTC')
    result={}
    for i,symbol in enumerate(('BTCUSDT','ETHUSDT','BNBUSDT')):
        close=100*np.exp(np.cumsum(rng.normal(.00015,.004*(i+1),len(index))))
        result[symbol]=pd.DataFrame({'open':close,'high':close*1.01,'low':close*.99,
            'close':close,'volume':100.},index=index)
    return result


@pytest.mark.parametrize('variant,expected',[
    ('guarded_edge',[1,1,1,1,0,1,1,1,0,1,1,1]),
    ('entry_hysteresis',[0,1,1,1,0,0,1,1,0,1,1,1]),
    ('symmetric_hurdle',[0,1,1,0,0,0,1,0,0,1,1,0]),
    ('two_day_confirmation',[0,1,1,1,0,0,1,1,0,0,1,1]),
    ('hysteresis_confirmed',[0,0,1,1,0,0,0,0,0,0,1,1])])
def test_explicit_entry_memory_and_unchanged_nonpositive_exit(variant,expected):
    momentum=pd.DataFrame({'s':[.001,.007,.008,.003,0,.005,.009,.002,-.01,.02,.021,.001]})
    ready=pd.DataFrame(True,index=momentum.index,columns=momentum.columns)
    assert eligibility_states(momentum,ready,variant).s.astype(int).tolist()==expected


def test_hurdle_is_strict_and_has_the_fixed_nominal_cost_interpretation():
    from scripts.research_spot_flow import NOMINAL
    leg=NOMINAL['fee_rate']+(NOMINAL['slippage_bps']+NOMINAL['spread_bps']/2)/10000
    assert ENTRY_HURDLE==pytest.approx(4*leg,abs=1e-15)
    m=pd.DataFrame({'s':[ENTRY_HURDLE,ENTRY_HURDLE+1e-7,ENTRY_HURDLE,0.]})
    ready=pd.DataFrame(True,index=m.index,columns=m.columns)
    assert eligibility_states(m,ready,'entry_hysteresis').s.tolist()==[False,True,True,False]


def test_pre_warmup_strength_does_not_become_an_unfunded_entry():
    m=pd.DataFrame({'s':[.01,.02,.03,.001,.002,.009]})
    ready=pd.DataFrame({'s':[False,False,False,True,True,True]})
    assert eligibility_states(m,ready,'entry_hysteresis').s.tolist()==[False,False,False,False,False,True]
    assert not eligibility_states(m,ready,'hysteresis_confirmed').s.any()


@pytest.mark.parametrize('variant',VARIANTS)
def test_targets_ignore_future_prices_and_partial_last_days(variant):
    source=markets()
    full=completed_entry_targets(source)[variant]
    for cut in (5,61*6+2,101*6+4):
        prefix={s:f.iloc[:cut] for s,f in source.items()}
        short=completed_entry_targets(prefix)[variant]
        changed={s:f.copy() for s,f in source.items()}
        for f in changed.values():
            f.iloc[cut:,f.columns.get_indexer(['open','high','low','close'])]*=7
        alternate=completed_entry_targets(changed)[variant]
        pd.testing.assert_frame_equal(short,full.iloc[:cut])
        pd.testing.assert_frame_equal(short,alternate.iloc[:cut])


def test_reconstructed_control_matches_the_existing_frozen_rule():
    source=markets()
    actual=completed_entry_targets(source)['guarded_edge']
    expected=momentum_refinement_targets(source)[0]['capped_allocation']
    pd.testing.assert_frame_equal(actual,expected,rtol=1e-13,atol=1e-14)


def test_all_targets_keep_spot_caps_and_full_volatility_warmup():
    for target in completed_entry_targets(markets()).values():
        assert target.iloc[:60*6+5].eq(0).all().all()
        assert target.ge(0).all().all() and target.le(.25+1e-12).all().all()
        assert target.sum(axis=1).le(.5+1e-12).all()


@pytest.mark.parametrize('bad',['nan','negative','gap','unaligned','naive','off_clock'])
def test_invalid_source_clocks_and_prices_reject(bad):
    source=markets()
    if bad=='nan': source['BTCUSDT'].iloc[10,3]=np.nan
    elif bad=='negative': source['BTCUSDT'].iloc[10,3]=-1
    elif bad=='gap': source={s:f.drop(f.index[10]) for s,f in source.items()}
    elif bad=='unaligned': source['BTCUSDT']=source['BTCUSDT'].iloc[:-1]
    elif bad=='naive': source={s:f.tz_localize(None) for s,f in source.items()}
    else:
        for f in source.values(): f.index+=pd.Timedelta('1h')
    with pytest.raises(ValueError): completed_entry_targets(source)


@pytest.mark.parametrize('variant',VARIANTS)
def test_candidate_targets_execute_with_owned_cash_and_lagged_monday_buys(variant):
    source=markets()
    target=completed_entry_targets(source)[variant]
    risk=PortfolioRisk(volatility_target=.15)
    equity,trades,_=run_intraday_spot(source,target,daily_control=True,
        rebalance_destination='band_edge',risk=risk,
        completed_covariances=completed_risk_covariances(source,risk))
    assert reconcile_spot_path(source,equity,trades,1000)['reconciled']
    buys=trades.loc[trades.side=='buy']
    assert not buys.empty
    assert buys.timestamp.dt.day_of_week.eq(0).all()
    assert buys.scheduled_decision.all() and not buys.risk_reduction.any()
    first_publication=target.index[target.sum(axis=1)>0][0]
    assert buys.timestamp.min()>=first_publication+pd.Timedelta('4h')
    assert equity.usdt.min()>=0 and equity.filter(like='base_').min().min()>=0


@pytest.mark.parametrize('field',['settings','runtime','source_sha256','source_manifest'])
def test_freeze_is_exclusive_and_rejects_rehashed_changes(tmp_path,field):
    manifest={'source':'fixture','datasets':{}}
    proof=freeze(manifest,tmp_path)
    assert read_freeze(tmp_path,manifest)==proof
    with pytest.raises(FileExistsError): freeze(manifest,tmp_path)
    proof[field]={}
    proof['protocol_sha256']=hashlib.sha256(canonical({k:v for k,v in proof.items() if k!='protocol_sha256'})).hexdigest()
    (tmp_path/'frozen_protocol.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='Frozen'): read_freeze(tmp_path,manifest)


def test_prior_forward_and_paper_protocol_sources_are_preserved():
    root=Path(__file__).resolve().parents[1]
    for name in ('THETABOT_FORWARD_PROTOCOL_2026-10-16.json','THETABOT_DEPTH_PROTOCOL_2026-10-23.json'):
        proof=json.loads((root/'docs/evaluation'/name).read_text())
        assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in proof['source_sha256'].items())
