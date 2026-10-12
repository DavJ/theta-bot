"""Optional causal five-day Bayesian and conditional Monte Carlo spot signals.

The simulations add a model, not new market information. Probabilities remain
model estimates. Nothing here connects to an exchange or enables live orders.
"""
from __future__ import annotations

from dataclasses import dataclass
import zlib

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t


FORECAST_FEATURES = ('r1', 'r5', 'r20', 'r60', 'vol20', 'relative_volume',
                     'market_r5', 'phase_concentration', 'theta')
SIGNAL_POLICIES = ('spot_control', 'ready_control', 'bayes_buy_gate',
                   'mc_buy_gate', 'fused_buy_gate', 'bayes_blend',
                   'mc_blend', 'fused_blend')


@dataclass(frozen=True)
class ProbabilisticConfig:
    horizon: int = 5
    training_window: int = 504
    minimum_training: int = 252
    analogue_count: int = 64
    simulations: int = 4096
    seed: int = 20261012
    slope_precision: float = 20.
    intercept_precision: float = 1.
    prior_shape: float = 3.
    prior_scale: float = .01
    calibration_window: int = 60
    minimum_calibration: int = 20

    def __post_init__(self):
        integers = ('horizon', 'training_window', 'minimum_training', 'analogue_count',
                    'simulations', 'seed', 'calibration_window', 'minimum_calibration')
        scalars = (self.slope_precision, self.intercept_precision, self.prior_shape, self.prior_scale)
        if (any(type(getattr(self, name)) is not int for name in integers)
                or self.horizon < 1 or self.minimum_training < 2 * self.horizon
                or self.training_window < self.minimum_training
                or not 2 <= self.analogue_count <= self.minimum_training
                or self.simulations < 32 or not 0 <= self.seed < 2**32
                or not 2 <= self.minimum_calibration <= self.calibration_window
                or not np.isfinite(scalars).all() or min(scalars) <= 0 or self.prior_shape <= 1):
            raise ValueError('Invalid probabilistic forecast configuration')


def _training_arrays(x, y, current):
    x, y, current = np.asarray(x, float), np.asarray(y, float), np.asarray(current, float)
    if (x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1
            or y.shape != (len(x),) or current.shape != (x.shape[1],)
            or not all(np.isfinite(a).all() for a in (x, y, current))):
        raise ValueError('Finite matching training and current feature arrays required')
    return x, y, current


def bayesian_prediction(x, y, current, config=ProbabilisticConfig()):
    """Conjugate NIG posterior on already standardized, nonoverlapping outcomes."""
    x, y, current = _training_arrays(x, y, current)
    design = np.column_stack((np.ones(len(x)), x))
    query = np.r_[1., current]
    precision = np.diag(np.r_[config.intercept_precision,
                             np.full(x.shape[1], config.slope_precision)]) + design.T @ design
    coefficients = np.linalg.solve(precision, design.T @ y)
    shape = config.prior_shape + len(y) / 2.
    scale = config.prior_scale + .5 * float(y @ y - coefficients @ precision @ coefficients)
    leverage = float(query @ np.linalg.solve(precision, query))
    if scale <= 0 or leverage < -1e-12:
        raise ValueError('Invalid Bayesian posterior covariance')
    leverage = max(leverage, 0.)
    return {'mean': float(query @ coefficients), 'variance': scale / (shape - 1) * (1 + leverage),
            'latent_variance': scale / (shape - 1) * leverage,
            'df': 2 * shape, 'scale': float(np.sqrt(scale / shape * (1 + leverage))),
            'observations': len(y)}


def conditional_monte_carlo(x, y, current, rng, hurdles, config=ProbabilisticConfig()):
    """Sample whole, completed H-day return blocks attached to nearby past states."""
    x, y, current = _training_arrays(x, y, current)
    if len(x) < config.analogue_count:
        raise ValueError('Insufficient completed analogues')
    distances = np.square(x - current).sum(axis=1)
    neighbors = np.argsort(distances, kind='stable')[:config.analogue_count]
    population = y[neighbors]
    draws = rng.choice(population, size=config.simulations, replace=True)
    variance = float(draws.var(ddof=1))
    result = {'mean': float(draws.mean()), 'variance': variance,
        'q05': float(np.quantile(draws, .05)), 'q50': float(np.quantile(draws, .5)),
        'q95': float(np.quantile(draws, .95)), 'numerical_mean_se': np.sqrt(variance / len(draws)),
        'exact_analogue_mean': float(population.mean()),
        'exact_analogue_variance': float(population.var(ddof=0)),
        'analogue_count': len(population), 'neighbors': neighbors}
    for name, hurdle in hurdles.items():
        probability = float((draws > hurdle).mean())
        result.update({f'prob_{name}': probability,
            f'exact_prob_{name}': float((population > hurdle).mean()),
            f'numerical_prob_se_{name}': float(np.sqrt(probability * (1 - probability) / len(draws)))})
    return result


def conservative_fusion(means, variances, errors, config=ProbabilisticConfig()):
    """Nonnegative minimum-error mixture with a predictive-variance floor."""
    means, variances, errors = np.asarray(means, float), np.asarray(variances, float), np.asarray(errors, float)
    if errors.size == 0:
        errors = np.empty((0, 2))
    if (means.shape != (2,) or variances.shape != (2,) or errors.ndim != 2 or errors.shape[1] != 2
            or not all(np.isfinite(a).all() for a in (means, variances, errors)) or (variances < 0).any()):
        raise ValueError('Matching finite two-expert forecasts and errors required')
    errors = errors[-config.calibration_window:]
    weight = .5
    if len(errors) >= config.minimum_calibration:
        second_moment = errors.T @ errors / len(errors)
        regularized = .5 * second_moment + .5 * np.diag(np.diag(second_moment)) + np.eye(2) * 1e-10
        denominator = regularized[0, 0] + regularized[1, 1] - 2 * regularized[0, 1]
        weight = float(np.clip((regularized[1, 1] - regularized[0, 1]) / denominator, 0., 1.))
        empirical_variance = float(np.array([weight, 1 - weight]) @ second_moment @ np.array([weight, 1 - weight]))
    else:
        empirical_variance = float(np.dot([weight, 1 - weight], variances))
    variance = max(empirical_variance, float(variances.min()), 1e-12)
    return {'mean': float(np.dot([weight, 1 - weight], means)), 'variance': variance,
            'weight_bayes': weight, 'weight_mc': 1 - weight, 'calibration_count': len(errors)}


def walk_forward_probabilistic(features, close, *, symbol, hurdles,
                               config=ProbabilisticConfig()):
    """Every label, scaler and mixture weight is known at the current daily close."""
    if (not isinstance(symbol, str) or not symbol or not hurdles
            or any(not np.isfinite(h) or h < 0 for h in hurdles.values())
            or not features.index.equals(close.index) or features.columns.duplicated().any()
            or features.shape[1] < 1 or not isinstance(close.index, pd.DatetimeIndex)
            or close.index.tz is None or close.empty or not close.index.is_unique
            or not close.index.is_monotonic_increasing
            or not close.index.to_series().diff().dropna().eq(pd.Timedelta('1D')).all()
            or not np.isfinite(close).all() or (close <= 0).any()):
        raise ValueError('Matching complete timezone-aware daily feature/price grid required')
    x = features.to_numpy(dtype=float)
    prices = np.log(close.to_numpy(dtype=float))
    y = np.full(len(close), np.nan)
    y[:-config.horizon] = prices[config.horizon:] - prices[:-config.horizon]
    complete = np.isfinite(x).all(axis=1)
    past_predictions = np.full((len(close), 2), np.nan)
    errors, rows, last_calibration = [], [], pd.NaT
    asset_id = zlib.crc32(symbol.encode('utf-8'))
    for i, when in enumerate(close.index):
        matured = i - config.horizon
        if matured >= 0 and matured % config.horizon == 0 and np.isfinite(past_predictions[matured]).all():
            errors.append(y[matured] - past_predictions[matured])
            errors = errors[-config.calibration_window:]
            last_calibration = when + pd.Timedelta('1D')
        row = {'ready': False, 'seed': config.seed, 'horizon_days': config.horizon,
            'available_at': when + pd.Timedelta('1D'), 'training_count': 0,
            'last_training_outcome_at': pd.NaT, 'last_analogue_outcome_at': pd.NaT,
            'last_fusion_outcome_at': last_calibration, 'bayes_observations': 0,
            'mc_draws': config.simulations, 'fused_calibration_count': len(errors)}
        for expert in ('bayes', 'mc', 'fused', 'unconditional'):
            row[expert + '_mean'] = np.nan
            row[expert + '_variance'] = np.nan
            for name in hurdles:
                row[f'{expert}_prob_{name}'] = np.nan
        row.update({f'input_{name}': float(value) for name, value in zip(features.columns, x[i])})
        training = np.arange(max(0, matured - config.training_window + 1), max(0, matured + 1))
        training = training[complete[training] & np.isfinite(y[training])]
        row['training_count'] = len(training)
        if not complete[i] or len(training) < config.minimum_training:
            rows.append(row)
            continue
        mean, scale = x[training].mean(axis=0), np.maximum(x[training].std(axis=0, ddof=0), 1e-8)
        standardized = (x[training] - mean) / scale
        current = (x[i] - mean) / scale
        # Counting eligible rows backwards cannot create label blocks <H days apart.
        nonoverlap = np.arange(len(training) - 1, -1, -config.horizon)[::-1]
        bayes = bayesian_prediction(standardized[nonoverlap], y[training[nonoverlap]], current, config)
        day_id = (when.as_unit('ns').value // (86400 * 10**9)) % 2**32
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, asset_id, day_id]))
        mc = conditional_monte_carlo(standardized, y[training], current, rng, hurdles, config)
        neighbors = mc.pop('neighbors')
        fused = conservative_fusion([bayes['mean'], mc['mean']],
                                     [bayes['variance'], mc['variance']], errors, config)
        past_predictions[i] = [bayes['mean'], mc['mean']]
        row.update(ready=True, last_training_outcome_at=close.index[training[-1] + config.horizon] + pd.Timedelta('1D'),
            last_analogue_outcome_at=close.index[training[neighbors].max() + config.horizon] + pd.Timedelta('1D'),
            unconditional_mean=float(y[training].mean()), unconditional_variance=float(y[training].var(ddof=0)))
        for expert, prediction in (('bayes', bayes), ('mc', mc), ('fused', fused)):
            row.update({f'{expert}_{key}': value for key, value in prediction.items()})
        for name, hurdle in hurdles.items():
            row[f'bayes_prob_{name}'] = float(student_t.sf((hurdle - bayes['mean']) / bayes['scale'], bayes['df']))
            row[f'fused_prob_{name}'] = float(norm.sf((hurdle - fused['mean']) / np.sqrt(fused['variance'])))
            row[f'unconditional_prob_{name}'] = float((y[training] > hurdle).mean())
        row.update({f'training_mean_{name}': float(value) for name, value in zip(features.columns, mean)})
        row.update({f'training_scale_{name}': float(value) for name, value in zip(features.columns, scale)})
        rows.append(row)
    result = pd.DataFrame(rows, index=close.index)
    for name in ('available_at', 'last_training_outcome_at', 'last_analogue_outcome_at', 'last_fusion_outcome_at'):
        result[name] = pd.to_datetime(result[name], utc=True)
    return result


def probabilistic_signal_policies(base_target, forecasts=None, *, hurdle_name='nominal',
                                  hurdle=0., enabled=False):
    """Combine models with the old exposure; off does not inspect forecast input."""
    if not enabled:
        return {'spot_control': (base_target.copy(), None)}
    if (not forecasts or set(forecasts) != set(base_target.columns)
            or not np.isfinite(hurdle) or hurdle < 0):
        raise ValueError('Matching forecast assets and finite cost hurdle required')
    daily_index = next(iter(forecasts.values())).index
    if any(not f.index.equals(daily_index) for f in forecasts.values()):
        raise ValueError('Forecast assets must share a complete daily grid')
    ready = pd.DataFrame({s: f.ready for s, f in forecasts.items()})
    daily_target = base_target.reindex(daily_index + pd.Timedelta('20h'))
    daily_target.index = daily_index
    if daily_target.isna().any().any():
        raise ValueError('Base target does not cover every daily availability time')
    matched = daily_target.where(ready, 0.)

    def publish(frame, boolean=False):
        frame = frame.copy()
        frame = frame.reindex(columns=base_target.columns)
        frame.index = frame.index + pd.Timedelta('20h')
        frame = frame.reindex(base_target.index, method='ffill').fillna(False if boolean else 0.)
        return frame.astype(bool) if boolean else frame.astype(float)

    result = {'spot_control': (base_target.copy(), None), 'ready_control': (publish(matched), None)}
    for expert in ('bayes', 'mc', 'fused'):
        means = pd.DataFrame({s: f[expert + '_mean'] for s, f in forecasts.items()})
        probabilities = pd.DataFrame({s: f[f'{expert}_prob_{hurdle_name}'] for s, f in forecasts.items()})
        allowed = ready & means.gt(hurdle) & probabilities.ge(.55)
        result[expert + '_buy_gate'] = publish(matched), publish(allowed, boolean=True)
        conviction = ((probabilities - .5) / .1).clip(0., 1.).where(means.gt(hurdle), 0.).fillna(0.)
        result[expert + '_blend'] = publish(matched * (.5 + .5 * conviction)), None
    return result
