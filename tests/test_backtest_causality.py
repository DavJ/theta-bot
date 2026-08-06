from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from spot_bot.backtest import fast_backtest as fb
from spot_bot.backtest.fast_backtest import run_backtest


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "BTCUSDT_1H_real.csv.gz"


def _load_real_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required real dataset is missing: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _run(df: pd.DataFrame, strategy: str, *, fee_rate: float, slippage_bps: float, spread_bps: float):
    return run_backtest(
        df=df,
        timeframe="1h",
        strategy_name=strategy,
        psi_mode="none",
        psi_window=200,
        rv_window=500,
        conc_window=200,
        base=2.0,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        max_exposure=1.0,
        initial_usdt=1000.0,
        min_notional=5.0,
        log=False,
        fill_margin_bps=1.0,
        limit_timeout_bars=1,
    )


def test_t1_random_signal_is_not_profitable():
    df = _load_real_data()
    returns = []
    sharpes = []

    for seed in range(10):
        rng = np.random.default_rng(seed)

        def _noise_intents(features, strategy, risk_state, risk_budget, max_exposure):
            return pd.Series(rng.uniform(0.0, 1.0, size=len(features)), index=features.index, dtype=float)

        with patch.object(fb, "_compute_intents_with_regime", side_effect=_noise_intents):
            _, _, summary = _run(df, "kalman", fee_rate=0.001, slippage_bps=5.0, spread_bps=2.0)
        returns.append(float(summary["total_return"]))
        sharpes.append(float(summary["sharpe"]))

    assert float(np.median(returns)) < 0.0
    assert float(np.median(sharpes)) < 0.5


def test_t2_strategy_and_inverse_cannot_both_win():
    df = _load_real_data()

    _, _, base_summary = _run(df, "kalman", fee_rate=0.0, slippage_bps=5.0, spread_bps=2.0)

    original = fb._compute_intents_with_regime

    def _inverse_intents(features, strategy, risk_state, risk_budget, max_exposure):
        base = original(features, strategy, risk_state, risk_budget, max_exposure)
        return (1.0 - base).clip(lower=0.0, upper=1.0)

    with patch.object(fb, "_compute_intents_with_regime", side_effect=_inverse_intents):
        _, _, inverse_summary = _run(df, "kalman", fee_rate=0.0, slippage_bps=5.0, spread_bps=2.0)

    assert not (
        float(base_summary["total_return"]) > 0.05
        and float(inverse_summary["total_return"]) > 0.05
    )


def test_t3_fill_price_is_not_systematically_better_than_bar_close():
    df = _load_real_data()
    _, trades_df, _ = _run(df, "kalman_mr_dual", fee_rate=0.001, slippage_bps=5.0, spread_bps=2.0)
    assert len(trades_df) >= 100

    better = np.where(
        trades_df["side"].to_numpy() == "buy",
        trades_df["price"].to_numpy() < trades_df["bar_close"].to_numpy(),
        trades_df["price"].to_numpy() > trades_df["bar_close"].to_numpy(),
    )
    share = float(np.mean(better))
    assert 0.35 < share < 0.65


def test_t4_extra_signal_delay_does_not_improve_returns():
    df = _load_real_data()
    original = fb._compute_intents_with_regime
    delayed_returns = []

    for extra_delay in (1, 2, 3):
        def _delayed(features, strategy, risk_state, risk_budget, max_exposure, d=extra_delay):
            base = original(features, strategy, risk_state, risk_budget, max_exposure)
            return base.shift(d).fillna(0.0)

        with patch.object(fb, "_compute_intents_with_regime", side_effect=_delayed):
            _, _, summary = _run(df, "kalman", fee_rate=0.001, slippage_bps=5.0, spread_bps=2.0)
        delayed_returns.append(float(summary["total_return"]))

    assert delayed_returns[1] <= delayed_returns[0] + 0.02
    assert delayed_returns[2] <= delayed_returns[1] + 0.02


def test_t5_slippage_changes_results_and_hurts_performance():
    df = _load_real_data()
    _, _, summary_zero = _run(df, "kalman", fee_rate=0.001, slippage_bps=0.0, spread_bps=2.0)
    _, _, summary_high = _run(df, "kalman", fee_rate=0.001, slippage_bps=50.0, spread_bps=2.0)

    ret_zero = float(summary_zero["total_return"])
    ret_high = float(summary_high["total_return"])
    assert abs(ret_zero - ret_high) > 1e-9
    assert ret_high < ret_zero
