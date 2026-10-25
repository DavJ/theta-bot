"""Calendar arithmetic, actual asset accounting, and fixed omission execution."""
import hashlib
import json
import gzip
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts import research_robustness as study
from theta_bot_averaging.robustness import (
    SYMBOLS,UNIVERSES,omission_targets,calendar_diagnostics,attribute_spot_pnl,
)
from spot_bot.portfolio.momentum_refinement import _redistribute_capped


@pytest.mark.parametrize('excluded',SYMBOLS)
def test_omission_reallocation_matches_capped_inverse_score_solution(excluded):
    scores=np.array([[0,0,0],[7,0,0],[0,2,3],[1,9,3],[20,1,1],[2,2,0]],dtype=float)
    original=pd.DataFrame([_redistribute_capped(row,.5,.25) for row in scores],columns=SYMBOLS)
    masked=scores.copy()
    masked[:,SYMBOLS.index(excluded)]=0
    expected=pd.DataFrame([_redistribute_capped(row,.5,.25) for row in masked],columns=SYMBOLS)
    pd.testing.assert_frame_equal(omission_targets(original,'without_'+excluded),expected)


def test_original_universe_is_an_independent_unchanged_copy():
    source=pd.DataFrame([[.2,.1,.2]],columns=SYMBOLS)
    actual=omission_targets(source,'all_three')
    pd.testing.assert_frame_equal(actual,source)
    actual.iloc[0,0]=0
    assert source.iloc[0,0]==.2


@pytest.mark.parametrize('problem',['unknown','negative','nan','asset_cap','total_cap','column_order'])
def test_invalid_source_targets_reject(problem):
    source=pd.DataFrame([[.2,.1,.2]],columns=SYMBOLS)
    universe='all_three'
    if problem=='unknown': universe='best_after_backtest'
    elif problem=='negative': source.iloc[0,0]=-.01
    elif problem=='nan': source.iloc[0,0]=np.nan
    elif problem=='asset_cap': source.iloc[0,0]=.26
    elif problem=='total_cap': source.iloc[0]=.2
    else: source=source[list(reversed(SYMBOLS))]
    with pytest.raises(ValueError,match='Fixed universe'): omission_targets(source,universe)


@pytest.fixture(scope='module')
def source():
    rng=np.random.default_rng(54187)
    index=pd.date_range('2022-01-01',periods=150*6,freq='4h',tz='UTC')
    result={}
    for j,s in enumerate(SYMBOLS):
        price=100*np.exp(np.cumsum(rng.normal(.001,.008*(j+1),len(index))))
        result[s]=pd.DataFrame({'open':price,'high':price*1.01,'low':price*.99,
                               'close':price,'volume':100.},index=index)
    return result


@pytest.mark.parametrize('universe',UNIVERSES[1:])
@pytest.mark.parametrize('lag',[1,2])
def test_omitted_asset_is_never_traded_or_held_with_both_execution_lags(source,universe,lag):
    original=study.entry.completed_entry_targets(source)['entry_hysteresis']
    targets=omission_targets(original,universe)
    risk=study.prior.PortfolioRisk(volatility_target=.20)
    eq,trades,_=study.entry.run_intraday_spot(source,targets,initial_usdt=10000,
        daily_control=True,rebalance_destination='band_edge',risk=risk,
        completed_covariances=study.entry.completed_risk_covariances(source,risk),
        signal_delay_bars=lag-1,**study.entry.NOMINAL)
    excluded=universe.removeprefix('without_')
    assert not trades.empty and not trades.symbol.eq(excluded).any()
    assert eq['base_'+excluded].eq(0).all()
    assert study.entry.reconcile_spot_path(source,eq,trades,10000)['reconciled']
    assert eq.usdt.min()>=0 and eq.filter(like='base_').min().min()>=0
    buys=trades[trades.side=='buy']
    assert buys.timestamp.dt.day_of_week.eq(0).all() and buys.scheduled_decision.all()


def calendar_account():
    index=pd.date_range('2022-01-01','2024-04-01',freq='4h',inclusive='left',tz='UTC')
    changes=np.full(27,.01)
    changes[[1,12,24]]=[-.10,.20,-.20]
    nav=np.full(len(index),10000.)
    for month,change in zip(pd.date_range('2022-01-01',periods=27,freq='MS',tz='UTC'),changes):
        final=(month+pd.offsets.MonthBegin(1))-pd.Timedelta('4h')
        nav[index>=final]*=1+change
    return pd.DataFrame({'timestamp':index,'equity':nav}),changes


def test_calendar_returns_rolling_windows_and_partial_year_use_same_funded_path():
    equity,changes=calendar_account()
    result=calendar_diagnostics(equity,10000)
    assert len(result['months'])==27
    np.testing.assert_allclose([r['return'] for r in result['months']],changes,atol=1e-14)
    assert result['months'][0]['starting_nav']==10000
    for row,start,end in zip(result['calendar_years'],(0,12,24),(12,24,27)):
        assert row['return']==pytest.approx(np.prod(1+changes[start:end])-1)
    assert [r['partial_year'] for r in result['calendar_years']]==[False,False,True]
    assert result['calendar_years'][-1]['last_month']=='2024-03'
    for window in (12,24):
        expected=[np.prod(1+changes[i:i+window])-1 for i in range(28-window)]
        actual=result['rolling_months'][str(window)]
        np.testing.assert_allclose([r['return'] for r in actual['windows']],expected,atol=1e-14)
        assert actual['count']==28-window
        assert actual['negative_count']==sum(v<0 for v in expected)
    assert result['rolling_months']['12']['negative_count']==3
    assert result['longest_month_end_underwater_run']==11


def test_concentration_is_compounded_log_arithmetic_not_sum_of_simple_returns():
    equity,_=calendar_account()
    result=calendar_diagnostics(equity,10000)
    expected_share=(np.log(1.2)+2*np.log(1.01))/(np.log(1.2)+24*np.log(1.01))
    assert result['best_three_share_of_positive_log_return']==pytest.approx(expected_share)
    assert result['mathematical_return_excluding_best_three_months']==pytest.approx(.9*.8*1.01**22-1)
    assert result['best_month']['month']=='2023-01'
    assert result['worst_month']['month']=='2024-01'


def test_flat_cash_and_short_history_distinguish_zero_from_unavailable_windows():
    index=pd.date_range('2022-01-01','2022-03-01',freq='4h',inclusive='left',tz='UTC')
    result=calendar_diagnostics(pd.DataFrame({'timestamp':index,'equity':10000.}),10000)
    assert result['total_return']==0
    assert result['best_three_share_of_positive_log_return'] is None
    assert result['rolling_months']['12']['count']==0
    assert result['rolling_months']['12']['worst_return'] is None
    assert result['rolling_months']['12']['negative_fraction'] is None


@pytest.mark.parametrize('problem',['partial_start','partial_end','missing_bar','naive','duplicate','zero_nav'])
def test_calendar_rejects_incomplete_or_invalid_accounts(problem):
    equity,_=calendar_account()
    if problem=='partial_start': equity=equity.iloc[1:]
    elif problem=='partial_end': equity=equity.iloc[:-1]
    elif problem=='missing_bar': equity=equity.drop(100)
    elif problem=='naive': equity.timestamp=equity.timestamp.dt.tz_localize(None)
    elif problem=='duplicate': equity.loc[100,'timestamp']=equity.loc[99,'timestamp']
    else: equity.loc[0,'equity']=0
    with pytest.raises(ValueError): calendar_diagnostics(equity,10000)


def attributable_account():
    index=pd.date_range('2022-01-01',periods=2,freq='4h',tz='UTC')
    markets={s:pd.DataFrame({'open':[50.,50.],'close':[50.,50.]},index=index) for s in SYMBOLS}
    markets['BTCUSDT']=pd.DataFrame({'open':[100.,115.],'close':[110.,120.]},index=index)
    eq=pd.DataFrame({'timestamp':index,'equity':[1007.,1009.],
        'base_BTCUSDT':[1.,0.],'base_ETHUSDT':0.,'base_BNBUSDT':0.})
    trades=pd.DataFrame({'timestamp':index,'symbol':'BTCUSDT','fee':1.,'slippage':2.})
    return markets,eq,trades


def test_attribution_includes_owned_gap_returns_intrabar_returns_and_both_costs():
    markets,equity,trades=attributable_account()
    result=attribute_spot_pnl(markets,equity,trades,1000)
    assert result['assets']['BTCUSDT']=={'gross_holding_pnl':15.,'fees':2.,'impact':4.,'net_pnl_contribution':9.}
    assert result['assets']['ETHUSDT']['net_pnl_contribution']==0
    assert result['total_net_pnl']==9 and result['max_bar_residual']==0 and result['terminal_residual']==0


@pytest.mark.parametrize('problem',['nav','inventory','cost','source_clock'])
def test_attribution_corruption_is_detected(problem):
    markets,equity,trades=attributable_account()
    if problem=='nav': equity.loc[0,'equity']+=.1
    elif problem=='inventory': equity.loc[0,'base_BTCUSDT']=-1
    elif problem=='cost': trades.loc[0,'fee']=np.nan
    else: markets['BTCUSDT']=markets['BTCUSDT'].iloc[1:]
    with pytest.raises(ValueError): attribute_spot_pnl(markets,equity,trades,1000)


@pytest.mark.parametrize('field',['settings','runtime','source_sha256','source_manifest','preceding_evidence'])
def test_freeze_exclusivity_and_rehashed_corruption_rejection(tmp_path,monkeypatch,field):
    monkeypatch.setattr(study,'prior_identity',lambda:{'report_sha256':'fixed','archive_sha256':'archive'})
    manifest={'source':'fixture'}
    proof=study.freeze(manifest,tmp_path)
    assert study.read_freeze(tmp_path,manifest)==proof
    with pytest.raises(FileExistsError): study.freeze(manifest,tmp_path)
    proof[field]={}
    body={k:v for k,v in proof.items() if k!='protocol_sha256'}
    proof['protocol_sha256']=hashlib.sha256(study.entry.canonical(body)).hexdigest()
    (tmp_path/'frozen_protocol.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='Frozen'): study.read_freeze(tmp_path,manifest)


@pytest.mark.parametrize('problem',['clean','csv','summary'])
def test_preceding_evidence_checks_inner_identity_after_outer_rehash(tmp_path,monkeypatch,problem):
    report={'identity':'fixed'}
    prior=tmp_path/'report.json.gz'
    prior.write_bytes(gzip.compress(json.dumps(report).encode(),mtime=0))
    csv=b'nav\n10000\n'
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('account.csv',csv if problem!='csv' else b'nav\n20000\n')
        z.writestr('csv_hashes.json',json.dumps({'account.csv':hashlib.sha256(csv).hexdigest()}))
        z.writestr('summary.json',json.dumps(report if problem!='summary' else {'identity':'changed'}))
    payload=buffer.getvalue()
    part=tmp_path/'evidence.part001'
    part.write_bytes(payload)
    digest=hashlib.sha256(payload).hexdigest()
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'bytes':len(payload),'sha256':digest,
        'parts':[{'file':part.name,'bytes':len(payload),'sha256':digest}]}))
    monkeypatch.setattr(study,'PRIOR',prior)
    monkeypatch.setattr(study,'PRIOR_ARCHIVE',manifest)
    if problem=='clean':
        assert study.verify_prior_archive()==digest
    else:
        with pytest.raises(ValueError,match='Preceding'): study.verify_prior_archive()
