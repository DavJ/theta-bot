"""Causal calibration and correlated fusion of same-horizon return forecasts.

Source scores are not measurements of price. They are first calibrated against
completed next-day outcomes; the Kalman state then represents expected return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastKalman:
    mean: float = 0.0
    variance: float = 1e-4
    persistence: float = 0.95

    def update(self, forecasts, measurement_covariance, process_variance):
        z = np.asarray(forecasts, dtype=float)
        r = np.asarray(measurement_covariance, dtype=float)
        if (z.ndim != 1 or len(z) == 0 or r.shape != (len(z), len(z))
                or not np.isfinite(z).all() or not np.isfinite(r).all()
                or not np.isfinite(process_variance) or process_variance < 0):
            raise ValueError("Expected finite forecasts, matching covariance and nonnegative process variance")
        if not np.allclose(r, r.T) or np.linalg.eigvalsh(r).min() <= 0:
            raise ValueError("Measurement covariance must be symmetric positive definite")
        prior_mean = self.persistence * self.mean
        prior_variance = self.persistence ** 2 * self.variance + process_variance
        h = np.ones(len(z))
        innovation_covariance = prior_variance * np.outer(h, h) + r
        gain = np.linalg.solve(innovation_covariance, prior_variance * h)
        self.mean = float(prior_mean + gain @ (z - prior_mean))
        # Joseph form avoids negative covariance from cancellation.
        self.variance = float(max((1 - gain @ h) ** 2 * prior_variance + gain @ r @ gain, 1e-12))
        return self.mean, self.variance, gain


def calibrate_and_fuse(scores: pd.DataFrame, daily_close: pd.Series) -> pd.DataFrame:
    """Use score(t-1), return(t) pairs available when day t has closed.

    Fixed protocol: 252-day calibration, 126 observations minimum, individual
    slope ridge .01, 60 realized forecast-error vectors, 50% diagonal shrinkage.
    All parameters are frozen independently of validation outcomes.
    """
    scores = scores.astype(float)
    realized = daily_close.pct_change(fill_method=None)
    predictors = scores.shift(1)
    volatility = realized.rolling(20, min_periods=2).std(ddof=0).clip(lower=0.0001)
    n = len(scores.columns)
    filt = ForecastKalman()
    previous_forecasts = None
    errors = []
    rows = []
    for i in range(len(scores)):
        observed_return = float(realized.iloc[i])
        if previous_forecasts is not None and np.isfinite(observed_return):
            errors.append(observed_return - previous_forecasts)
            errors = errors[-60:]
        start = max(0, i - 251)
        x = predictors.iloc[start:i + 1].to_numpy()
        y = realized.iloc[start:i + 1].to_numpy()
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if valid.sum() < 126 or not np.isfinite(scores.iloc[i].to_numpy()).all():
            previous_forecasts = None
            rows.append({})
            continue
        x, y = x[valid], y[valid]
        x_mean, y_mean = x.mean(axis=0), y.mean()
        centered = x - x_mean
        slopes = (centered * (y - y_mean)[:, None]).mean(axis=0) / ((centered ** 2).mean(axis=0) + 0.01)
        forecasts = y_mean + slopes * (scores.iloc[i].to_numpy() - x_mean)
        daily_variance = float(max(volatility.iloc[i] ** 2, 1e-8))
        if len(errors) >= 20:
            covariance = np.atleast_2d(np.cov(np.asarray(errors), rowvar=False, ddof=0))
            covariance = 0.5 * covariance + 0.5 * np.diag(np.diag(covariance))
            covariance += np.eye(n) * max(daily_variance * 1e-4, 1e-10)
        else:
            covariance = daily_variance * (0.5 * np.eye(n) + 0.5 * np.ones((n, n)))
        if previous_forecasts is None:
            filt.variance = daily_variance
        mean, variance, gain = filt.update(forecasts, covariance, daily_variance * 0.05 ** 2)
        previous_forecasts = forecasts
        diagonal = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(diagonal, diagonal)
        off_diagonal = correlation[~np.eye(n, dtype=bool)]
        row = {f"forecast_{name}": float(value) for name, value in zip(scores.columns, forecasts)}
        row.update({f"gain_{name}": float(value) for name, value in zip(scores.columns, gain)})
        row.update(fused_return=mean, mean_return=float(forecasts.mean()), posterior_variance=variance,
                   forecast_error_mean_correlation=float(off_diagonal.mean()) if n > 1 else 0.0)
        rows.append(row)
    return pd.DataFrame(rows, index=scores.index)
