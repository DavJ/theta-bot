"""Purged five-day forecasts and causal spot target allocations for research."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spot_bot.features import FeatureConfig, compute_features
from spot_bot.portfolio.trend import TrendPortfolio
from spot_bot.strategies.meanrev_dual_kalman import MeanRevDualKalmanStrategy


def price_features(markets):
    closes = pd.DataFrame({s: f.close for s, f in markets.items()})
    result = {}
    for symbol, market in markets.items():
        close = closes[symbol]
        returns = close.pct_change(fill_method=None)
        features = pd.DataFrame({f"r{n}": close.pct_change(n, fill_method=None)
                                 for n in (1, 5, 20, 30, 60)}, index=close.index)
        for n in (20, 60):
            features[f"vol{n}"] = returns.rolling(n, min_periods=n).std(ddof=0)
        features["range"] = (market.high - market.low) / close
        features["relative_volume"] = market.volume / market.volume.rolling(20, min_periods=20).mean() - 1
        features["market_r5"] = closes.BTCUSDT.pct_change(5, fill_method=None)
        features["relative_r20"] = features.r20 - closes.BTCUSDT.pct_change(20, fill_method=None)
        theta = compute_features(market, FeatureConfig(rv_window=20, conc_window=60, psi_window=60))
        features["phase_concentration"] = theta.C
        features["phase_scale"] = theta.psi
        features["theta"] = MeanRevDualKalmanStrategy(price_space="log_vol", volatility_window=60).generate_series(theta)
        result[symbol] = features.replace([np.inf, -np.inf], np.nan)
    return result


@dataclass(frozen=True)
class ForecastConfig:
    algorithm: str = "ridge"
    horizon: int = 5
    training_window: int = 504
    minimum_training: int = 126
    refit_every: int = 20
    recency_half_life_days: float = 126.

    def __post_init__(self):
        if (not all(isinstance(v, int) for v in (self.horizon, self.training_window,
                                                self.minimum_training, self.refit_every))
                or self.algorithm not in {"ridge", "boost"} or self.horizon < 1
                or self.minimum_training < 20 or self.training_window < self.minimum_training
                or self.refit_every < 1 or not np.isfinite(self.recency_half_life_days)
                or self.recency_half_life_days <= 0):
            raise ValueError("Invalid walk-forward forecast configuration")


def training_recency_weights(rows, current_row, config):
    """Exponential age of completed targets, normalized to mean weight one."""
    ages = current_row - np.asarray(rows) - config.horizon
    if len(ages) == 0 or (ages < 0).any():
        raise ValueError("Training weights require already completed outcomes")
    weights = np.exp2(-ages / config.recency_half_life_days)
    return weights / weights.mean()


def walk_forward_forecast(features: pd.DataFrame, close: pd.Series,
                          config=ForecastConfig()) -> pd.DataFrame:
    """Fit only rows whose future target has matured by the current close."""
    if (not features.index.equals(close.index) or not features.index.is_unique
            or not features.index.is_monotonic_increasing or features.columns.duplicated().any()
            or not isinstance(close.index, pd.DatetimeIndex) or close.index.tz is None
            or not close.index.to_series().diff().dropna().eq(pd.Timedelta("1D")).all()
            or not np.isfinite(close).all() or (close <= 0).any()):
        raise ValueError("Matching ordered feature/price timestamps required")
    values = features.to_numpy(dtype=float)
    targets = (close.shift(-config.horizon) / close - 1).to_numpy()
    complete = np.isfinite(values).all(axis=1)
    predictions = np.full(len(close), np.nan)
    training_counts = np.zeros(len(close), dtype=int)
    effective_counts = np.zeros(len(close))
    training_last = [pd.NaT] * len(close)
    model, fitted_at, last_label_at, count, effective = None, -config.refit_every, pd.NaT, 0, 0.
    for i in range(len(close)):
        if not complete[i]:
            continue
        if model is None or i - fitted_at >= config.refit_every:
            stop = i - config.horizon
            rows = np.arange(max(0, stop - config.training_window + 1), stop + 1)
            rows = rows[complete[rows] & np.isfinite(targets[rows])]
            if len(rows) < config.minimum_training:
                continue
            y = targets[rows]
            sigma = float(y.std(ddof=0))
            y = np.clip(y, -3 * sigma, 3 * sigma)
            weights = training_recency_weights(rows, i, config)
            if config.algorithm == "ridge":
                model = make_pipeline(StandardScaler(), Ridge(alpha=10))
                model.fit(values[rows], y, standardscaler__sample_weight=weights,
                          ridge__sample_weight=weights)
            else:
                model = HistGradientBoostingRegressor(max_iter=100, learning_rate=.05,
                    max_depth=2, max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=10,
                    early_stopping=False, random_state=20261006)
                model.fit(values[rows], y, sample_weight=weights)
            fitted_at, count = i, len(rows)
            effective = float(weights.sum() ** 2 / np.square(weights).sum())
            last_label_at = close.index[rows[-1] + config.horizon] + pd.Timedelta("1D")
        predictions[i] = float(model.predict(values[i:i+1])[0])
        training_counts[i], training_last[i] = count, last_label_at
        effective_counts[i] = effective
    result = pd.DataFrame({"forecast": predictions, "training_count": training_counts,
                           "effective_training_count": effective_counts,
                           "last_training_outcome_at": pd.to_datetime(training_last, utc=True)}, index=close.index)
    valid = result.forecast.notna()
    if (result.loc[valid, "last_training_outcome_at"] > result.index[valid] + pd.Timedelta("1D")).any():
        raise AssertionError("An unmatured outcome reached training")
    return result


def forecast_targets(forecast: pd.DataFrame, close: pd.DataFrame, *, horizon=5, cost_hurdle=.0032):
    if not forecast.index.equals(close.index) or not forecast.columns.equals(close.columns):
        raise ValueError("Forecast allocation must match the market grid")
    vol = close.pct_change(fill_method=None).rolling(60, min_periods=60).std(ddof=0).clip(lower=.0001)
    conviction = ((forecast - cost_hurdle) / (.5 * vol * np.sqrt(horizon))).clip(0, 1)
    inverse_vol = 1 / vol
    weights = inverse_vol.div(inverse_vol.sum(axis=1), axis=0) * conviction
    return weights.fillna(0).clip(0, .5)


@dataclass(frozen=True)
class TargetPortfolio(TrendPortfolio):
    """Precomputed completed-close targets, executed with the core's daily lag."""
    targets: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False, compare=False)

    def weights(self, close):
        if (self.targets.empty or not self.targets.index.is_unique
                or not self.targets.index.is_monotonic_increasing
                or not close.index.isin(self.targets.index).all()
                or not close.columns.equals(self.targets.columns)):
            raise ValueError("Target portfolio must cover the identical market grid")
        targets = self.targets.loc[close.index].copy()
        if (not np.isfinite(targets.to_numpy()).all() or (targets < 0).any().any()
                or targets.sum(axis=1).max() > self.max_exposure + 1e-12
                or targets.max().max() > self.asset_cap + 1e-12):
            raise ValueError("Precomputed spot targets exceed allocation limits")
        return targets
