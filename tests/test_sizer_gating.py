import pytest

from spot_bot.portfolio.sizer import compute_target_position


def test_sizer_gates_off_state():
    target = compute_target_position(
        equity_usdt=1000.0,
        price=100.0,
        desired_exposure=0.8,
        risk_budget=0.5,
        max_exposure=1.0,
        risk_state="OFF",
    )
    assert target == 0.0


def test_sizer_scales_when_on():
    target = compute_target_position(
        equity_usdt=1000.0,
        price=100.0,
        desired_exposure=0.8,
        risk_budget=0.5,
        max_exposure=1.0,
        risk_state="ON",
    )
    assert target == pytest.approx(4.0)
