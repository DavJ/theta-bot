"""Causal risk information, actual spot fills and minimum-turnover rebalances."""
import numpy as np
import pandas as pd
import pytest

from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.execution_risk import completed_risk_covariances
from spot_bot.portfolio.risk import PortfolioRisk
from tests.test_spot_flow import markets
from scripts.audit_cost_risk import audit_ledger, independent_covariances
from scripts.research_cost_risk import improvement_gates, POLICIES, SCENARIOS, PERIODS


ZERO = dict(fee_rate=0., slippage_bps=0., spread_bps=0.)


def flat(n=60):
    data = {'BTCUSDT': markets(n)['BTCUSDT']}
    data['BTCUSDT'][['open', 'high', 'low', 'close']] = 100.
    return data, pd.DataFrame(.25, index=data['BTCUSDT'].index, columns=list(data))


def test_off_switch_preserves_every_legacy_field_and_fill():
    data = markets(500)
    target = pd.DataFrame(.15, index=data['BTCUSDT'].index, columns=list(data))
    old = run_intraday_spot(data, target, daily_control=True)
    off = run_intraday_spot(data, target, daily_control=True, risk=None, rebalance_destination='target')
    pd.testing.assert_frame_equal(old[0], off[0])
    pd.testing.assert_frame_equal(old[1], off[1])
    assert old[2] == off[2]


def test_band_edge_rebalance_trades_only_to_nearest_boundary():
    data, target = flat(60)
    # Monday Jan 1 00h has no mature target; the following Monday is first buy.
    _, full, _ = run_intraday_spot(data, target, daily_control=True, **ZERO)
    eq, edge, _ = run_intraday_spot(data, target, daily_control=True,
        rebalance_destination='band_edge', **ZERO)
    assert full.qty.iloc[0] == pytest.approx(2.5)
    assert edge.qty.iloc[0] == pytest.approx(2.4)
    assert edge.notional.sum() < full.notional.sum()
    assert reconcile_spot_path(data, eq, edge)['reconciled']


def test_band_does_not_block_inactivity_exit_or_cap_sale():
    data, target = flat(60)
    target.iloc[43:] = 0.
    _, fills, _ = run_intraday_spot(data, target, daily_control=True,
        rebalance_band=.1, rebalance_destination='band_edge', **ZERO)
    assert fills.side.tolist() == ['buy', 'sell']
    assert fills.qty.iloc[1] == pytest.approx(fills.qty.iloc[0])
    data, target = flat(60)
    frame = data['BTCUSDT']
    frame.iloc[43:, frame.columns.get_indexer(['open', 'high', 'low', 'close'])] = 200.
    eq, fills, _ = run_intraday_spot(data, target, daily_control=True,
        rebalance_band=.1, rebalance_destination='band_edge', **ZERO)
    assert fills.side.tolist() == ['buy', 'sell']
    assert eq.exposure.iloc[48] == pytest.approx(.25)


def test_boundary_adjustment_does_not_relax_aggregate_or_asset_caps():
    data = markets(100)
    target = pd.DataFrame(1/6, index=data['BTCUSDT'].index, columns=list(data))
    eq, fills, _ = run_intraday_spot(data, target, rebalance_destination='band_edge', min_notional=0., **ZERO)
    assert eq.usdt.min() >= 0 and eq.filter(like='base_').min().min() >= 0
    assert reconcile_spot_path(data, eq, fills)['reconciled']
    # Check actual open allocation from the independent holdings ledger.
    holdings = {s: 0. for s in data}
    cash = 1000.
    for ts, rows in fills.groupby('timestamp', sort=True):
        for row in rows.itertuples():
            q = row.qty if row.side == 'buy' else -row.qty
            holdings[row.symbol] += q
            cash -= q * row.price + row.fee
        nav = cash + sum(holdings[s] * data[s].loc[ts, 'open'] for s in data)
        weights = np.array([holdings[s] * data[s].loc[ts, 'open'] / nav for s in data])
        assert weights.sum() <= .5 + 1e-10 and weights.max() <= .25 + 1e-10


def test_daily_covariance_never_publishes_a_partial_day():
    data = markets(6 * 85)
    risk = PortfolioRisk(volatility_target=.15)
    full = completed_risk_covariances(data, risk)
    prefix = {s: f.iloc[:6 * 80 + 4] for s, f in data.items()}
    earlier = completed_risk_covariances(prefix, risk)
    np.testing.assert_allclose(earlier, full[:len(earlier)], equal_nan=True)
    np.testing.assert_allclose(earlier[-1], earlier[-5])
    assert np.isnan(full[6 * 60 + 4]).all()
    assert np.isfinite(full[6 * 60 + 5]).all()


@pytest.mark.parametrize('delay', [0, 1])
def test_covariance_and_targets_share_completed_information_lag(delay):
    data, target = flat(15)
    cov = np.zeros((15, 1, 1))
    cov[5:] = .04
    eq, fills, _ = run_intraday_spot(data, target, risk=PortfolioRisk(volatility_target=.15),
        completed_covariances=cov, signal_delay_bars=delay, **ZERO)
    cuts = fills.loc[fills.risk_reduction]
    assert cuts.timestamp.iloc[0] == target.index[6 + delay]
    # The source candle's future high and low do not retroactively alter orders.
    modified = {s: f.copy() for s, f in data.items()}
    modified['BTCUSDT'].loc[target.index[5], ['high', 'low']] = [500., 1.]
    other = run_intraday_spot(modified, target, risk=PortfolioRisk(volatility_target=.15),
        completed_covariances=cov, signal_delay_bars=delay, **ZERO)
    pd.testing.assert_frame_equal(fills, other[1])
    assert reconcile_spot_path(data, eq, fills)['reconciled']


def test_actual_holdings_are_cut_between_daily_and_monday_decisions():
    data, target = flat(60)
    cov = np.zeros((60, 1, 1))
    cov[44:] = .01
    eq, fills, _ = run_intraday_spot(data, target, daily_control=True,
        rebalance_band=.2, risk=PortfolioRisk(volatility_target=.15),
        completed_covariances=cov, **ZERO)
    cuts = fills.loc[fills.risk_reduction]
    assert cuts.timestamp.iloc[0] == target.index[45]
    assert not cuts.scheduled_decision.iloc[0]
    assert cuts.notional.iloc[0] < 200.
    assert eq.exposure.iloc[45] < .08
    assert (fills.loc[~fills.scheduled_decision, 'side'] == 'sell').all()


def test_missing_history_requests_cash_and_minimum_notional_dust_is_disclosed():
    data, target = flat(20)
    cov = np.full((20, 1, 1), np.nan)
    eq, fills, _ = run_intraday_spot(data, target,
        risk=PortfolioRisk(volatility_target=.15), completed_covariances=cov, **ZERO)
    assert fills.empty and eq.exposure.eq(0).all()
    cov[:] = 0.
    # A tiny necessary reduction is below the unchanged spot minimum.
    cov[4:] = (.15 / (.248 * np.sqrt(365))) ** 2
    eq, fills, _ = run_intraday_spot(data, target,
        risk=PortfolioRisk(volatility_target=.15), completed_covariances=cov, **ZERO)
    assert len(fills) == 1
    assert eq.planned_exposure.iloc[5] < eq.exposure.iloc[5]


@pytest.mark.parametrize('kind', ['shape', 'partial_nan', 'infinity', 'negative', 'asymmetric'])
def test_invalid_covariance_information_is_rejected(kind):
    data = markets(12)
    target = pd.DataFrame(.15, index=data['BTCUSDT'].index, columns=list(data))
    cov = np.zeros((12, 3, 3))
    if kind == 'shape': cov = cov[:-1]
    elif kind == 'partial_nan': cov[0, 0, 0] = np.nan
    elif kind == 'infinity': cov[0, 0, 0] = np.inf
    elif kind == 'negative': cov[0, 0, 0] = -.01
    elif kind == 'asymmetric': cov[0, 1, 0] = .01
    with pytest.raises(ValueError, match='[Cc]ovariance'):
        run_intraday_spot(data, target, risk=PortfolioRisk(volatility_target=.15), completed_covariances=cov)


@pytest.mark.parametrize('kwargs', [dict(rebalance_destination='guess'), dict(risk='guess'),
    dict(risk=PortfolioRisk(volatility_target=.15)), dict(completed_covariances=np.zeros((12, 1, 1)))])
def test_invalid_optional_execution_configuration_is_rejected(kwargs):
    data, target = flat(12)
    with pytest.raises(ValueError): run_intraday_spot(data, target, **kwargs)


def test_guarded_execution_is_prefix_invariant_including_partial_final_day():
    data = markets(6 * 90)
    risk = PortfolioRisk(volatility_target=.15)
    target = pd.DataFrame(.15, index=data['BTCUSDT'].index, columns=list(data))
    prefix = {s: f.iloc[:487] for s, f in data.items()}
    a = run_intraday_spot(prefix, target.iloc[:487], daily_control=True,
        risk=risk, completed_covariances=completed_risk_covariances(prefix, risk),
        rebalance_destination='band_edge')
    b = run_intraday_spot(data, target, daily_control=True,
        risk=risk, completed_covariances=completed_risk_covariances(data, risk),
        rebalance_destination='band_edge')
    pd.testing.assert_frame_equal(a[0], b[0].iloc[:487])
    pd.testing.assert_frame_equal(a[1], b[1].loc[b[1].timestamp < target.index[487]].reset_index(drop=True))
    assert reconcile_spot_path(data, b[0], b[1])['reconciled']


@pytest.mark.parametrize('corruption', [None, 'cash', 'drawdown', 'volatility', 'fee'])
def test_independent_covariance_and_csv_audit_detect_material_corruption(corruption):
    data = markets(6 * 90)
    risk = PortfolioRisk(volatility_target=.15)
    target = pd.DataFrame(.15, index=data['BTCUSDT'].index, columns=list(data))
    eq, fills, _ = run_intraday_spot(data, target, daily_control=True,
        risk=risk, completed_covariances=completed_risk_covariances(data, risk),
        rebalance_destination='band_edge')
    covariance = independent_covariances(data, 1)
    if corruption == 'cash': eq.loc[400, 'usdt'] += 1.
    elif corruption == 'drawdown': eq.loc[400, 'intraday_drawdown_bound'] += .01
    elif corruption == 'volatility': eq.loc[400, 'estimated_annual_volatility_after_fills'] += .01
    elif corruption == 'fee': fills.loc[0, 'fee'] += 1.
    costs = dict(fee_rate=.001, slippage_bps=5., spread_bps=2.)
    if corruption is None:
        result = audit_ledger(data, eq, fills, 1000., costs, covariance)
        assert result['reconciled'] and result['max_risk_volatility_error'] < 1e-12
    else:
        with pytest.raises(ValueError, match='audit residual'):
            audit_ledger(data, eq, fills, 1000., costs, covariance)


def test_risk_only_improvement_cannot_pass_the_profit_gate_or_select_an_ablation():
    rows = {s: {period: {p: {'accounting': {'reconciled': True},
                'net_pnl': 100. if p == 'spot_control' else (10000. if p == 'band_edge' else 90.),
                'max_intraday_drawdown_bound': -.21 if p == 'spot_control' else -.15}
                for p in POLICIES} for period in PERIODS} for s in SCENARIOS}
    gates = improvement_gates(rows)
    assert gates['risk_improvement_screen_passed']
    assert not gates['profit_improvement_screen_passed']
    rows['joint_stress']['development']['guarded_edge']['max_intraday_drawdown_bound'] = -.201
    assert not improvement_gates(rows)['risk_improvement_screen_passed']
