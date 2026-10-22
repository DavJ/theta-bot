"""Arithmetic identities, causal prediction, matched controls and freeze integrity."""
import json

import numpy as np
import pandas as pd
import pytest

from theta_bot_averaging.mobius_research import (
    FAMILIES, WINDOW, HORIZON, LOG_HURDLE, basis, mobius_values,
    rolling_forecast, predictive_metrics, buy_mask,
)
from scripts.research_mobius_theta import freeze, read_freeze


def prices(size=180):
    rng = np.random.default_rng(721)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(.0002,.01,size))),
                     index=pd.date_range('2022-01-01',periods=size,freq='4h',tz='UTC'))


def test_mobius_values_and_divisor_inversion():
    mu = np.r_[0,mobius_values(100)]
    assert mu[1:11].tolist() == [1,-1,-1,0,-1,1,-1,0,0,1]
    for n in range(1,101):
        assert sum(mu[d] for d in range(1,n+1) if n%d==0) == int(n==1)
    rng = np.random.default_rng(31)
    primitive = rng.normal(size=101)
    mixed = np.array([sum(primitive[d] for d in range(1,n+1) if n%d==0)
                      for n in range(101)])
    recovered = [sum(mu[d]*mixed[n//d] for d in range(1,n+1) if n%d==0)
                 for n in range(1,101)]
    np.testing.assert_allclose(recovered,primitive[1:],atol=1e-14)


def test_exact_theta_scale_inversion_and_mertens_partial_summation():
    mu = mobius_values(32)
    n = np.arange(1,65)
    for q in (.2,.5,.9):
        for z in (0.,.17,.83,1.4):
            theta_terms = np.array([2*np.sum(q**((k*n)**2)*np.cos(2*k*n*z))
                                    for k in range(1,33)])
            assert mu@theta_terms == pytest.approx(2*q*np.cos(2*z),abs=2e-14)
            mertens = mu.cumsum()
            abel = mertens[-1]*theta_terms[-1] + mertens[:-1]@(theta_terms[:-1]-theta_terms[1:])
            assert abel == pytest.approx(mu@theta_terms,abs=2e-14)


@pytest.mark.parametrize('family',FAMILIES)
def test_forecasts_and_buy_masks_do_not_change_when_future_changes(family):
    close = prices()
    cut = 130
    expected = rolling_forecast(close.iloc[:cut],family)
    changed = close.copy()
    changed.iloc[cut:] *= np.exp(np.linspace(2,4,len(close)-cut))
    full = rolling_forecast(changed,family)
    np.testing.assert_allclose(expected,full.iloc[:cut],equal_nan=True,atol=1e-13)
    assert buy_mask(expected.to_frame()).equals(buy_mask(full.iloc[:cut].to_frame()))
    assert full.iloc[:WINDOW].isna().all()
    assert full.iloc[WINDOW:].notna().all()


@pytest.mark.parametrize('pair',[('theta_full','fourier_full'),('mobius','fourier_squarefree')])
def test_independently_computed_predictions_and_decisions_agree(pair):
    close=prices()
    a,b=(rolling_forecast(close,f) for f in pair)
    np.testing.assert_allclose(a,b,equal_nan=True,atol=2e-13,rtol=1e-12)
    assert buy_mask(a.to_frame('s')).equals(buy_mask(b.to_frame('s')))


def test_one_window_forecast_matches_legacy_demeaned_fit_and_future_phase():
    close=prices(49)
    r=np.diff(np.log(close))
    mean=r.mean()
    phase=np.linspace(0,2*np.pi,48)
    x=basis(phase,'mobius')
    coefficients=np.linalg.lstsq(x,r-mean,rcond=None)[0]
    future_phase=np.arange(48,54)*2*np.pi/47
    expected=6*mean+np.sum(basis(future_phase,'mobius')@coefficients)
    assert rolling_forecast(close,'mobius').iloc[-1] == pytest.approx(expected,abs=1e-13)


def test_mean_predictor_and_missing_warmup_cannot_buy():
    close=prices()
    expected=np.log(close/close.shift(WINDOW))*HORIZON/WINDOW
    np.testing.assert_allclose(rolling_forecast(close,'mean'),expected,equal_nan=True,atol=1e-14)
    f=pd.DataFrame({'s':[np.nan,LOG_HURDLE-1e-7,LOG_HURDLE,LOG_HURDLE+1e-7]})
    assert buy_mask(f).s.tolist()==[False,False,False,True]


def test_prediction_scoring_requires_mature_nonoverlapping_period_targets():
    close=prices(120)
    pred=rolling_forecast(close,'mobius')
    start,end=close.index[60],close.index[97]
    a=predictive_metrics(close,pred,start,end)
    # Sample anchors 60,66,72,78,84,90; the final one matures at 97.
    assert a['samples']==6
    changed=close.copy()
    changed.loc[changed.index>=end]*=3
    b=predictive_metrics(changed,rolling_forecast(changed,'mobius'),start,end)
    assert a==b


@pytest.mark.parametrize('bad',[0,-1,True,2.5])
def test_invalid_mobius_limits_fail(bad):
    with pytest.raises(ValueError): mobius_values(bad)


@pytest.mark.parametrize('kind',['nan','negative','missing_bar','naive','duplicate'])
def test_invalid_market_inputs_fail(kind):
    close=prices()
    if kind=='nan': close.iloc[70]=np.nan
    elif kind=='negative': close.iloc[70]=-1
    elif kind=='missing_bar': close=close.drop(close.index[70])
    elif kind=='naive': close.index=close.index.tz_localize(None)
    else: close.index=pd.DatetimeIndex(list(close.index[:70])+[close.index[69]]+list(close.index[71:]))
    with pytest.raises(ValueError): rolling_forecast(close,'mobius')


def test_freeze_is_exclusive_and_rejects_tampering(tmp_path):
    manifest={'source':'fixture','datasets':{}}
    proof=freeze(manifest,tmp_path)
    assert read_freeze(tmp_path,manifest)==proof
    with pytest.raises(FileExistsError): freeze(manifest,tmp_path)
    with pytest.raises(ValueError): read_freeze(tmp_path,{'source':'changed'})
    path=tmp_path/'frozen_protocol.json'
    changed=json.loads(path.read_text())
    changed['settings']['q']=.8
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError): read_freeze(tmp_path,manifest)


def test_existing_forward_source_fingerprint_is_preserved():
    from spot_bot.forward_validation import source_fingerprints
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'docs/evaluation/THETABOT_FORWARD_PROTOCOL_2026-10-16.json').read_text())
    assert source_fingerprints()==old['source_sha256']
