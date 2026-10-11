"""Causal, deliberately simple payoff forecasts; estimates are not calibrated alpha."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

from spot_bot.research.long_options import entry_cost


@dataclass(frozen=True)
class OptionForecastConfig:
    method: str = 'empirical'
    minimum_edge: float = .1
    minimum_daily_closes: int = 252
    maximum_daily_closes: int = 730
    volatility_days: int = 60
    volatility_scale: float = 1.

    def __post_init__(self):
        if (self.method not in ('warmup', 'empirical', 'gaussian_rv')
                or type(self.minimum_daily_closes) is not int or self.minimum_daily_closes < 2
                or type(self.maximum_daily_closes) is not int or self.maximum_daily_closes < self.minimum_daily_closes
                or type(self.volatility_days) is not int or not 2 <= self.volatility_days < self.minimum_daily_closes
                or not np.isfinite([self.minimum_edge, self.volatility_scale]).all()
                or not 0 <= self.minimum_edge <= 10 or not 0 < self.volatility_scale <= 5):
            raise ValueError('Invalid causal option forecast configuration')


class CausalOptionEntryFilter:
    """Each decision uses history strictly before the execution day's UTC midnight."""
    def __init__(self, markets, config=OptionForecastConfig()):
        self.config = config
        self.daily = {s: f.close.resample('1D').last() for s, f in markets.items()}
        nodes, weights = hermgauss(128)
        self.normal_nodes, self.normal_weights = nodes * math.sqrt(2.), weights / math.sqrt(math.pi)
        self._history_cache = {}

    def __call__(self, row, spot, costs, when):
        when = pd.Timestamp(when).tz_convert('UTC')
        expiry = pd.Timestamp(row['expiry']).tz_convert('UTC')
        if expiry <= when or not np.isfinite(spot) or spot <= 0:
            raise ValueError('Invalid forecast execution spot or expiry')
        horizon = math.ceil((expiry - when).total_seconds() / 86400.)
        strike, side = float(row['contract']['strike']), row['contract']['option_type']
        if side not in ('call', 'put') or not np.isfinite(strike) or strike <= 0:
            raise ValueError('Invalid forecast option strike or side')
        debit, _, _ = entry_cost(row, spot, costs)
        symbol, cutoff = row['asset'] + 'USDT', when.normalize()
        key = symbol, cutoff, horizon
        if key not in self._history_cache:
            history = self.daily[symbol].loc[self.daily[symbol].index < cutoff].tail(self.config.maximum_daily_closes)
            values = history.to_numpy(dtype=float)
            self._history_cache[key] = history, values
        history, values = self._history_cache[key]
        base = {'forecast_method': self.config.method, 'minimum_edge': self.config.minimum_edge,
            'volatility_scale': self.config.volatility_scale, 'history_cutoff': str(cutoff),
            'last_completed_daily_close': str(history.index[-1]) if len(history) else None,
            'history_daily_closes': len(values), 'forecast_horizon_days': horizon,
            'execution_spot': float(spot), 'expiry': str(expiry), 'forecast_strike': strike, 'forecast_side': side,
            'accepted': False, 'expected_net_receipt_per_unit': None, 'buy_debit_per_unit': float(debit),
            'expected_edge': None, 'estimated_win_probability': None, 'forecast_samples': 0,
            'historical_annualized_volatility': None}
        if len(values) < self.config.minimum_daily_closes:
            return {**base, 'reason': 'insufficient_history'}
        if (not np.isfinite(values).all() or (values <= 0).any()
                or len(history) > 1 and not (np.diff(history.index.as_unit('ns').asi8) == 86400_000_000_000).all()):
            return {**base, 'reason': 'invalid_history'}
        if self.config.method == 'warmup':
            return {**base, 'accepted': True, 'reason': 'warmup_only'}
        if len(values) <= horizon:
            return {**base, 'reason': 'insufficient_horizon_history'}
        logs = np.log(values)
        volatility = float(np.std(np.diff(logs)[-self.config.volatility_days:], ddof=1))
        base['historical_annualized_volatility'] = volatility * math.sqrt(365.25)
        if self.config.method == 'empirical':
            returns = logs[horizon:] - logs[:-horizon]
            mean = float(returns.mean())
            terminal_logs = mean + self.config.volatility_scale * (returns - mean)
            weights = np.full(len(returns), 1. / len(returns))
        else:
            variance = (volatility * self.config.volatility_scale) ** 2 * horizon
            terminal_logs = -.5 * variance + math.sqrt(variance) * self.normal_nodes
            weights = self.normal_weights
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            terminal = spot * np.exp(terminal_logs)
        if not np.isfinite(terminal).all() or (terminal <= 0).any():
            return {**base, 'reason': 'invalid_forecast_distribution'}
        intrinsic = np.maximum(terminal - strike if side == 'call' else strike - terminal, 0.)
        delivery_fee = np.minimum(costs.delivery_fee_rate * terminal, costs.fee_cap * intrinsic)
        impact = (intrinsic - delivery_fee) * (costs.conversion_impact if costs.family == 'inverse' else 0.)
        net = (intrinsic - delivery_fee - impact) * (1 - costs.conversion_fee)
        receipt = float(np.dot(weights, net))
        edge = receipt / debit - 1.
        accepted = bool(edge >= self.config.minimum_edge)
        return {**base, 'accepted': accepted, 'reason': 'estimated_edge' if accepted else 'insufficient_edge',
            'expected_net_receipt_per_unit': receipt, 'buy_debit_per_unit': float(debit),
            'expected_edge': float(edge), 'estimated_win_probability': float(np.dot(weights, net > debit)),
            'forecast_samples': len(terminal)}
