import pandas as pd

from spot_bot.strategies.mean_reversion import MeanReversionStrategy


def _make_strategy() -> MeanReversionStrategy:
    return MeanReversionStrategy(
        ema_span=3, std_lookback=3, entry_z=0.1, full_z=1.0, min_exposure=0.2, max_exposure=0.8
    )


def test_mean_reversion_long_signal_bounds():
    strategy = _make_strategy()
    df = pd.DataFrame({"close": [100, 99, 98, 97, 96]})

    intent = strategy.generate_intent(df)

    assert 0.0 <= intent.desired_exposure <= 1.0
    assert intent.desired_exposure >= 0.2
    assert "Mean reversion" in intent.reason


def test_mean_reversion_flat_when_above_ema():
    strategy = _make_strategy()
    df = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

    intent = strategy.generate_intent(df)

    assert intent.desired_exposure == 0.0
    assert "No mean reversion signal" in intent.reason
