from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from spot_bot.backtest.fast_backtest import (
    _compute_intents_with_regime,
    _compute_risk_series_raw,
    _normalize_df,
    run_backtest,
)
from spot_bot.features import FeatureConfig, compute_features
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.strategies.meanrev_dual_kalman import MeanRevDualKalmanStrategy


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "BTCUSDT_1H_real.csv.gz"
BASE_PARAMS = {
    "timeframe": "1h",
    "psi_mode": "scale_phase",
    "psi_window": 256,
    "rv_window": 120,
    "conc_window": 256,
    "base": 10.0,
    "initial_usdt": 1000.0,
    "min_notional": 10.0,
    "max_exposure": 1.0,
    "log": False,
}


@lru_cache(maxsize=1)
def _load_real_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required real-data fixture is missing: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@lru_cache(maxsize=1)
def _dual_strategy_intent() -> pd.Series:
    df = _load_real_data()
    df_norm = _normalize_df(df)
    features = compute_features(
        df_norm,
        FeatureConfig(
            base=BASE_PARAMS["base"],
            rv_window=BASE_PARAMS["rv_window"],
            conc_window=BASE_PARAMS["conc_window"],
            psi_mode=BASE_PARAMS["psi_mode"],
            psi_window=BASE_PARAMS["psi_window"],
        ),
    )
    for col in ("open", "high", "low", "close", "volume"):
        features[col] = pd.to_numeric(df_norm[col], errors="coerce").values
    features["timestamp"] = df_norm["timestamp"].to_numpy()
    valid_mask = (
        features["C"].notna()
        & features["S"].notna()
        & features["close"].notna()
        & features["rv"].notna()
    )
    features = features.loc[valid_mask].copy()
    risk_state_raw, risk_budget_raw = _compute_risk_series_raw(features, RegimeEngine({}))
    return _compute_intents_with_regime(
        features,
        MeanRevDualKalmanStrategy(),
        risk_state_raw,
        risk_budget_raw,
        BASE_PARAMS["max_exposure"],
    )


def _run(intent_override: pd.Series | None = None, **overrides):
    params = {
        **BASE_PARAMS,
        "strategy_name": "meanrev",
        "fee_rate": 0.001,
        "slippage_bps": 5.0,
        "spread_bps": 2.0,
    }
    params.update(overrides)
    return run_backtest(_load_real_data(), intent_override=intent_override, **params)


def test_random_signal_is_loss_making_with_costs():
    returns = []
    sharpes = []

    for seed in range(10):
        rng = np.random.default_rng(seed)
        intent = pd.Series(rng.uniform(0.0, 1.0, len(_load_real_data())), index=_load_real_data().index)
        _, _, summary = _run(intent_override=intent)
        returns.append(summary["total_return"])
        sharpes.append(summary["sharpe"])

    assert float(np.median(returns)) < 0.0
    assert float(np.median(sharpes)) < 0.5


def test_strategy_and_inverse_do_not_both_clear_five_percent():
    signal = _dual_strategy_intent()
    _, _, summary_signal = _run(intent_override=signal, fee_rate=0.0)
    _, _, summary_inverse = _run(intent_override=1.0 - signal, fee_rate=0.0)

    assert not (
        summary_signal["total_return"] > 0.05
        and summary_inverse["total_return"] > 0.05
    )


def test_fills_are_not_systematically_better_than_bar_close():
    rng = np.random.default_rng(0)
    intent = pd.Series(rng.uniform(0.0, 1.0, len(_load_real_data())), index=_load_real_data().index)
    _, trades_df, _ = _run(intent_override=intent)

    assert len(trades_df) >= 100
    better_than_close = (
        ((trades_df["side"] == "buy") & (trades_df["price"] < trades_df["bar_close"]))
        | ((trades_df["side"] == "sell") & (trades_df["price"] > trades_df["bar_close"]))
    )
    better_share = float(better_than_close.mean())

    assert 0.35 < better_share < 0.65


def test_extra_signal_delay_does_not_improve_return():
    signal = _dual_strategy_intent()
    returns = []

    for extra_lag in range(4):
        delayed = signal.shift(extra_lag).fillna(0.0)
        _, _, summary = _run(intent_override=delayed)
        returns.append(summary["total_return"])

    for earlier, later in zip(returns, returns[1:]):
        assert later <= earlier + 0.02


def test_slippage_changes_results_and_hurts_performance():
    rng = np.random.default_rng(1)
    intent = pd.Series(rng.uniform(0.0, 1.0, len(_load_real_data())), index=_load_real_data().index)

    _, _, summary_zero = _run(intent_override=intent, slippage_bps=0.0)
    _, _, summary_high = _run(intent_override=intent, slippage_bps=50.0)

    assert summary_zero["total_return"] != summary_high["total_return"]
    assert summary_high["total_return"] < summary_zero["total_return"]
