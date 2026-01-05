import pandas as pd

from spot_bot.strategies.kalman import KalmanStrategy


def _df_from_prices(prices):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="H")
    return pd.DataFrame({"close": prices}, index=idx)


def test_exposure_moves_with_price_vs_level():
    strategy = KalmanStrategy(q_level=1e-4, q_trend=1e-6, r=1e-3, k=2.0, min_bars=5)
    below_df = _df_from_prices([100.0] * 10 + [95.0])
    above_df = _df_from_prices([100.0] * 10 + [105.0])

    intent_below = strategy.generate_intent(below_df)
    intent_above = strategy.generate_intent(above_df)

    assert intent_below.desired_exposure > intent_above.desired_exposure
    assert 0.0 <= intent_below.desired_exposure <= 1.0
    assert 0.0 <= intent_above.desired_exposure <= 1.0


def test_deterministic_outputs():
    strategy = KalmanStrategy(min_bars=5)
    df = _df_from_prices([100, 101, 102, 103, 104, 105, 104, 103, 102, 101])
    intent1 = strategy.generate_intent(df)
    intent2 = strategy.generate_intent(df)
    assert intent1.desired_exposure == intent2.desired_exposure
    assert not pd.isna(intent1.desired_exposure)


def test_min_history_guard():
    strategy = KalmanStrategy(min_bars=20)
    df = _df_from_prices([100, 99, 98])
    intent = strategy.generate_intent(df)
    assert intent.desired_exposure == 0.0
