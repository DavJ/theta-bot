"""Causal daily market replay with one shared cash account across spot assets."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd

from spot_bot.core.engine import EngineParams, simulate_execution
from spot_bot.core.portfolio import apply_fill
from spot_bot.core.trade_planner import plan_trade
from spot_bot.core.types import PortfolioState
from spot_bot.portfolio.trend import TrendPortfolio


def run_portfolio_backtest(markets: dict[str, pd.DataFrame], policy: TrendPortfolio, *,
                           start=None, end=None, initial_usdt=1000.0,
                           fee_rate=0.001, slippage_bps=5.0, spread_bps=2.0,
                           min_notional=10.0):
    if not markets or initial_usdt <= 0 or min_notional < 0:
        raise ValueError("Positive capital and nonempty markets required")
    if (not np.isfinite([initial_usdt, min_notional, fee_rate, slippage_bps, spread_bps]).all()
            or not 0 <= fee_rate < 0.5 or not 0 <= slippage_bps < 5000 or not 0 <= spread_bps < 5000):
        raise ValueError("Invalid capital or cost assumptions")
    first = next(iter(markets.values())).index
    if (not isinstance(first, pd.DatetimeIndex) or first.tz is None or first.empty
            or not first.is_unique or not first.is_monotonic_increasing
            or not first.to_series().diff().dropna().eq(pd.Timedelta("1D")).all()):
        raise ValueError("Daily bars require complete unique ordered UTC-aware dates")
    for frame in markets.values():
        if not frame.index.equals(first):
            raise ValueError("All markets must have identical dates; missing bars are not filled")
        if not {"open", "high", "low", "close", "volume"}.issubset(frame.columns):
            raise ValueError("Each market requires OHLCV")
        values = frame[["open", "high", "low", "close", "volume"]]
        if (not np.isfinite(values.to_numpy()).all() or (values.iloc[:, :4] <= 0).any().any()
                or (values.volume < 0).any()
                or (frame.low > frame[["open", "close"]].min(axis=1)).any()
                or (frame.high < frame[["open", "close"]].max(axis=1)).any()):
            raise ValueError("Invalid market OHLCV")
    closes = pd.DataFrame({s: f.close for s, f in markets.items()})
    opens = pd.DataFrame({s: f.open for s, f in markets.items()})
    # All targets for day t use only closes up to day t-1.
    targets = policy.weights(closes).shift(1).fillna(0.0)
    start = pd.to_datetime(start, utc=True) if start is not None else first[0]
    end = pd.to_datetime(end, utc=True) if end is not None else first[-1] + pd.Timedelta("1D")
    dates = first[(first >= start) & (first < end)]
    if dates.empty:
        raise ValueError("Empty evaluation period")
    params = EngineParams(fee_rate=fee_rate, slippage_bps=slippage_bps,
                          spread_bps=spread_bps, min_notional=min_notional,
                          allow_loss_exits=True, execution_policy="market")
    cash = float(initial_usdt)
    holdings = {s: PortfolioState(cash, 0.0, cash, 0.0) for s in markets}
    equity_rows, trade_rows = [], []
    peak = initial_usdt
    single_cap = policy.max_exposure if policy.approach == "btc_breakout" else policy.asset_cap
    for ts in dates:
        prices = opens.loc[ts]
        nav = cash + sum(holdings[s].base * prices[s] for s in markets)
        current = pd.Series({s: holdings[s].base * prices[s] / nav for s in markets})
        desired = targets.loc[ts]
        rebalance = ts.dayofweek == policy.rebalance_weekday
        final = desired.copy() if rebalance else current.copy()
        inactive = desired <= 0
        final.loc[inactive] = 0.0
        final = final.clip(upper=single_cap)
        if final.sum() > policy.max_exposure:
            final *= policy.max_exposure / final.sum()
        cap_reduction = (current > single_cap + 1e-12) | (current.sum() > policy.max_exposure + 1e-12)
        # Quantities are fixed using one known NAV, before execution. Sell first
        # so proceeds are available to other assets through the shared account.
        order = sorted(markets, key=lambda s: final[s] - current[s])
        for symbol in order:
            if (not inactive[symbol] and not cap_reduction[symbol]
                    and abs(final[symbol] - current[symbol]) < policy.rebalance_band):
                continue
            state = replace(holdings[symbol], usdt=cash, equity=nav, exposure=float(current[symbol]))
            plan = plan_trade(state, float(prices[symbol]), float(final[symbol]), min_notional,
                              allow_loss_exits=True)
            plan = replace(plan, order_type="market", limit_price=None)
            fill = simulate_execution(plan, float(prices[symbol]), params, portfolio=state)
            updated = apply_fill(state, fill)
            cash = updated.usdt
            holdings[symbol] = updated
            if fill.status == "filled" and fill.filled_base != 0:
                trade_rows.append({"timestamp": ts, "symbol": symbol,
                                   "side": "buy" if fill.filled_base > 0 else "sell",
                                   "qty": abs(fill.filled_base), "price": fill.avg_price,
                                   "notional": abs(fill.filled_base) * fill.avg_price,
                                   "fee": fill.fee_paid, "slippage": fill.slippage_paid,
                                   "execution_type": "market", "decision_nav": nav})
        asset_values = {s: holdings[s].base * closes.loc[ts, s] for s in markets}
        equity = cash + sum(asset_values.values())
        peak = max(peak, equity)
        equity_rows.append({"timestamp": ts, "equity": equity, "usdt": cash,
                            "drawdown": equity / peak - 1, "exposure": sum(asset_values.values()) / equity,
                            **{f"base_{s}": holdings[s].base for s in markets},
                            **{f"exposure_{s}": value / equity for s, value in asset_values.items()}})
    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows, columns=["timestamp", "symbol", "side", "qty", "price",
                                              "notional", "fee", "slippage", "execution_type", "decision_nav"])
    returns = equity.equity.pct_change(fill_method=None)
    returns.iloc[0] = equity.equity.iloc[0] / initial_usdt - 1
    vol = float(returns.std(ddof=0))
    fees, impact = float(trades.fee.sum()), float(trades.slippage.sum())
    net = float(equity.equity.iloc[-1]) - initial_usdt
    summary = {"final_equity": float(equity.equity.iloc[-1]), "total_return": net / initial_usdt,
               "net_pnl": net, "fees_paid_total": fees, "slippage_paid_total": impact,
               "gross_pnl_cost_addback": net + fees + impact,
               "maxDD": float(equity.drawdown.min()), "trades_count": len(trades),
               "turnover": float(trades.notional.sum()) / initial_usdt,
               "mean_exposure": float(equity.exposure.mean()), "max_close_exposure": float(equity.exposure.max()),
               "sharpe": float(returns.mean() / vol * np.sqrt(365)) if vol > 0 else 0.0}
    return equity, trades, summary
