"""Configured-risk CSV audit and immutable follow-up protocol contracts."""
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts import research_risk_utilization as study
from spot_bot.portfolio.risk import PortfolioRisk


@pytest.fixture(scope='module')
def source():
    rng=np.random.default_rng(68128)
    index=pd.date_range('2022-01-01',periods=180*6,freq='4h',tz='UTC')
    result={}
    for j,s in enumerate(('BTCUSDT','ETHUSDT','BNBUSDT')):
        price=100*np.exp(np.cumsum(rng.normal(.0008,.025+.005*j,len(index))))
        result[s]=pd.DataFrame({'open':price,'high':price*1.02,'low':price*.98,
                               'close':price,'volume':100.},index=index)
    return result


def replay(source,target=.15,lag=1):
    weights=pd.DataFrame(1/6,index=next(iter(source.values())).index,columns=list(source))
    risk=PortfolioRisk(volatility_target=target)
    equity,trades,_=study.entry.run_intraday_spot(source,weights,initial_usdt=10000,
        daily_control=True,rebalance_destination='band_edge',risk=risk,
        completed_covariances=study.entry.completed_risk_covariances(source,risk),
        signal_delay_bars=lag-1,**study.entry.NOMINAL)
    covariance=study.entry.independent_covariances(source,lag)
    return equity,trades,covariance


@pytest.mark.parametrize('target',[.15,.175,.20])
@pytest.mark.parametrize('lag',[1,2])
def test_raw_accounts_reconcile_at_each_configured_request_without_mutation(source,target,lag):
    equity,trades,covariance=replay(source,target,lag)
    before_eq,before_trades=equity.copy(deep=True),trades.copy(deep=True)
    result=study.audit_configured_risk(source,equity,trades,10000,study.entry.NOMINAL,covariance,target)
    assert result['reconciled'] and result['configured_annual_volatility']==target
    assert result['max_risk_volatility_error']<1e-10
    assert result['max_unscheduled_risk_scale_error']<1e-10
    assert ((~equity.scheduled_decision)&(equity.risk_scale<.99)&(equity.exposure>0)).any()
    assert equity.usdt.min()>=0 and equity.filter(like='base_').min().min()>=0
    pd.testing.assert_frame_equal(equity,before_eq)
    pd.testing.assert_frame_equal(trades,before_trades)


def test_higher_request_cannot_be_audited_as_original_15_percent(source):
    equity,trades,covariance=replay(source,.20)
    with pytest.raises(ValueError,match='risk audit residual'):
        study.audit_configured_risk(source,equity,trades,10000,study.entry.NOMINAL,covariance,.15)


@pytest.mark.parametrize('field',['estimated_annual_volatility_after_fills','risk_scale','planned_exposure'])
def test_corrupt_raw_risk_fields_are_detected(source,field):
    equity,trades,covariance=replay(source)
    row=equity.index[(~equity.scheduled_decision)&equity.estimated_annual_volatility_after_fills.notna()][0]
    equity.loc[row,field]+=.001
    with pytest.raises(ValueError,match='risk audit residual'):
        study.audit_configured_risk(source,equity,trades,10000,study.entry.NOMINAL,covariance,.15)


def test_estimate_without_history_is_rejected(source):
    equity,trades,covariance=replay(source)
    equity.loc[0,'estimated_annual_volatility_after_fills']=0.
    with pytest.raises(ValueError,match='without completed history'):
        study.audit_configured_risk(source,equity,trades,10000,study.entry.NOMINAL,covariance,.15)


def test_short_cash_only_warmup_account_has_no_empty_reduction_error(source):
    short={s:f.iloc[:20*6] for s,f in source.items()}
    equity,trades,covariance=replay(short)
    assert trades.empty
    assert study.audit_configured_risk(short,equity,trades,10000,study.entry.NOMINAL,covariance,.15)['reconciled']


def test_unscheduled_buy_annotation_is_rejected(source):
    equity,trades,covariance=replay(source)
    row=trades.index[trades.side=='buy'][0]
    trades.loc[row,'scheduled_decision']=False
    with pytest.raises(ValueError,match='bought inventory'):
        study.audit_configured_risk(source,equity,trades,10000,study.entry.NOMINAL,covariance,.15)


@pytest.mark.parametrize('target',[0.,-.1,np.inf,np.nan])
def test_invalid_request_is_rejected_before_audit(target):
    with pytest.raises(ValueError,match='Positive finite'):
        study.audit_configured_risk({},None,None,10000,{},None,target)


@pytest.mark.parametrize('field',['settings','runtime','source_sha256','source_manifest','preceding_report'])
def test_freeze_is_exclusive_and_rejects_even_rehashed_changes(tmp_path,monkeypatch,field):
    monkeypatch.setattr(study,'prior_identity',lambda:{'sha256':'fixture','protocol_sha256':'prior'})
    manifest={'source':'fixture'}
    proof=study.freeze(manifest,tmp_path)
    assert study.read_freeze(tmp_path,manifest)==proof
    with pytest.raises(FileExistsError): study.freeze(manifest,tmp_path)
    proof[field]={}
    body={k:v for k,v in proof.items() if k!='protocol_sha256'}
    proof['protocol_sha256']=hashlib.sha256(study.entry.canonical(body)).hexdigest()
    (tmp_path/'frozen_protocol.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='Frozen'): study.read_freeze(tmp_path,manifest)


def test_changed_preceding_report_invalidates_existing_freeze(tmp_path,monkeypatch):
    identity={'sha256':'first','protocol_sha256':'prior'}
    monkeypatch.setattr(study,'prior_identity',lambda:dict(identity))
    study.freeze({},tmp_path)
    identity['sha256']='second'
    with pytest.raises(ValueError,match='preceding report'): study.read_freeze(tmp_path,{})


def test_preceding_identity_binds_actual_compressed_bytes_and_protocol(tmp_path,monkeypatch):
    path=tmp_path/'prior.json.gz'
    payload=gzip.compress(b'{"frozen_protocol":{"protocol_sha256":"fixed"}}',mtime=0)
    path.write_bytes(payload)
    monkeypatch.setattr(study,'ROOT',tmp_path)
    monkeypatch.setattr(study,'PRIOR',path)
    identity=study.prior_identity()
    assert identity=={'file':'prior.json.gz','sha256':hashlib.sha256(payload).hexdigest(),'protocol_sha256':'fixed'}
