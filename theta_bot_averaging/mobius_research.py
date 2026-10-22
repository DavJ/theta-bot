"""Fixed causal theta/Möbius diagnostic; no exchange or order access."""
from __future__ import annotations

import numpy as np
import pandas as pd

FAMILIES = ('theta_full', 'mobius', 'fourier_full', 'fourier_squarefree',
            'fundamental', 'mean')
WINDOW = 48
HARMONICS = 8
Q = .9
HORIZON = 6
LOG_HURDLE = .0032


def mobius_values(limit):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('Positive integer Möbius limit required')
    values = np.ones(limit + 1, dtype=int)
    values[0] = 0
    prime = np.ones(limit + 1, dtype=bool)
    for p in range(2, limit + 1):
        if prime[p]:
            values[p::p] *= -1
            values[p*p::p*p] = 0
            prime[p*p::p] = False
    return values[1:]


def basis(phase, family, q=Q, harmonics=HARMONICS):
    if (family not in FAMILIES or not np.isfinite(q) or not 0 < q < 1
            or isinstance(harmonics, bool) or not isinstance(harmonics, int) or harmonics < 1):
        raise ValueError('Invalid fixed-basis specification')
    phase = np.asarray(phase, dtype=float)
    if phase.ndim != 1 or not np.isfinite(phase).all():
        raise ValueError('Finite one-dimensional phase required')
    if family == 'mean':
        return np.empty((len(phase), 0))
    n = np.arange(1, harmonics + 1)
    if family == 'fundamental':
        n = n[:1]
    elif family == 'fourier_squarefree':
        n = n[mobius_values(harmonics) != 0]
    matrix = np.cos(2 * phase[:, None] * n)
    if family in ('theta_full', 'mobius'):
        matrix *= q ** (n*n)
    if family == 'mobius':
        matrix *= mobius_values(harmonics)
    return matrix


def rolling_forecast(close, family, window=WINDOW, horizon=HORIZON, q=Q, harmonics=HARMONICS):
    """A timestamp uses only its close and the preceding window of returns."""
    if (not isinstance(close, pd.Series) or not isinstance(close.index, pd.DatetimeIndex)
            or close.index.tz is None or not close.index.is_unique
            or not close.index.is_monotonic_increasing
            or not close.index.to_series().diff().dropna().eq(pd.Timedelta('4h')).all()
            or isinstance(window, bool) or not isinstance(window, int) or window < 4
            or isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1
            or not np.isfinite(close.to_numpy(dtype=float)).all() or (close <= 0).any()):
        raise ValueError('Complete positive UTC-aware 4h closes and integer windows required')
    train_phase = np.arange(window) * (2 * np.pi / (window - 1))
    future_phase = np.arange(window, window + horizon) * (2 * np.pi / (window - 1))
    train = basis(train_phase, family, q, harmonics)
    future = basis(future_phase, family, q, harmonics)
    result = np.full(len(close), np.nan)
    if len(close) <= window:
        return pd.Series(result, index=close.index, name=family)
    returns = np.diff(np.log(close.to_numpy(dtype=float)))
    windows = np.lib.stride_tricks.sliding_window_view(returns, window)
    means = windows.mean(axis=1)
    # Fixed least-squares operator; no full-sample fit or future labels.
    operator = np.linalg.pinv(train) if train.shape[1] else np.empty((0, window))
    weights = future.sum(axis=0) @ operator
    result[window:] = horizon * means + (windows - means[:, None]) @ weights
    return pd.Series(result, index=close.index, name=family)


def buy_mask(forecasts):
    if not isinstance(forecasts, pd.DataFrame):
        raise ValueError('Aligned forecast frame required')
    return forecasts.gt(LOG_HURDLE) & forecasts.notna()


def predictive_metrics(close, prediction, start, end, horizon=HORIZON):
    if not close.index.equals(prediction.index):
        raise ValueError('Prediction and price dates must align')
    index = close.index
    first, stop = pd.Timestamp(start), pd.Timestamp(end)
    if first.tz is None or stop.tz is None or first >= stop:
        raise ValueError('Explicit ordered UTC period required')
    # A close is available four hours after its open-time label.
    available = index + pd.Timedelta('4h')
    maturity = available + horizon * pd.Timedelta('4h')
    anchor = pd.Timestamp('2022-01-01', tz='UTC')
    sampled = ((index - anchor) // pd.Timedelta('4h')) % horizon == 0
    truth = np.log(close.shift(-horizon) / close)
    valid = ((available >= first) & (maturity <= stop) & sampled
             & prediction.notna() & truth.notna())
    y, p = truth.loc[valid].to_numpy(), prediction.loc[valid].to_numpy()
    if not len(y):
        return {'samples': 0, 'mse': None, 'mae': None, 'sign_accuracy': None, 'correlation': None}
    correlation = (float(np.corrcoef(p, y)[0, 1])
                   if len(y) > 1 and np.std(p) > 0 and np.std(y) > 0 else None)
    return {'samples': len(y), 'mse': float(np.mean((p-y)**2)),
            'mae': float(np.mean(np.abs(p-y))),
            'sign_accuracy': float(np.mean(np.sign(p) == np.sign(y))),
            'correlation': correlation}
