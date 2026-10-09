"""Causal shared-cash spot replay on complete 4h bars with core fill accounting."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd

from spot_bot.core.engine import EngineParams, simulate_execution
from spot_bot.core.portfolio import apply_fill
from spot_bot.core.trade_planner import plan_trade
from spot_bot.core.types import PortfolioState


def run_intraday_spot(markets, completed_targets, *, start=None, end=None,
                      initial_usdt=1000., fee_rate=.001, slippage_bps=5., spread_bps=2.,
                      max_exposure=.5, asset_cap=.25, rebalance_band=.01,
                      min_notional=10., daily_control=False, signal_delay_bars=0,
                      completed_buy_mask=None):
    """Lag completed targets and optional buy permissions; sells never need permission."""
    if not markets or not 0 < asset_cap <= max_exposure <= 1:
        raise ValueError("Owned spot capital and nonnegative spot target limits required")
    if (not np.isfinite([initial_usdt, fee_rate, slippage_bps, spread_bps, min_notional,
                         rebalance_band, max_exposure, asset_cap]).all()
            or initial_usdt <= 0 or min_notional < 0 or rebalance_band < 0
            or not 0 <= fee_rate < .5 or not 0 <= slippage_bps < 5000 or not 0 <= spread_bps < 5000
            or not isinstance(signal_delay_bars, int) or not 0 <= signal_delay_bars <= 6):
        raise ValueError("Invalid spot costs, capital or signal delay")
    if daily_control and signal_delay_bars not in (0, 1):
        raise ValueError("Daily control supports only nominal or one-bar timing stress")
    symbols = list(markets)
    first = markets[symbols[0]].index
    if (not isinstance(first, pd.DatetimeIndex) or first.tz is None or first.empty
            or not first.is_unique or not first.is_monotonic_increasing
            or not first.to_series().diff().dropna().eq(pd.Timedelta("4h")).all()):
        raise ValueError("Complete ordered unique UTC-aware 4h grid required")
    index = first.tz_convert("UTC").as_unit("ns")
    arrays = {}
    for name in ("open", "high", "low", "close"):
        arrays[name] = np.column_stack([f[name].to_numpy(dtype=float) for f in markets.values()])
    for frame in markets.values():
        if not frame.index.equals(first):
            raise ValueError("All spot markets require identical dates; no gap filling")
        if (not np.isfinite(frame[["open", "high", "low", "close", "volume"]].to_numpy()).all()
                or (frame[["open", "high", "low", "close"]] <= 0).any().any()
                or (frame.volume < 0).any()
                or (frame.low > frame[["open", "close"]].min(axis=1)).any()
                or (frame.high < frame[["open", "close"]].max(axis=1)).any()):
            raise ValueError("Invalid spot OHLCV")
    if (list(completed_targets.columns) != symbols or not completed_targets.index.equals(first)
            or not np.isfinite(completed_targets.to_numpy()).all()
            or (completed_targets < 0).any().any()
            or (completed_targets > asset_cap + 1e-12).any().any()
            or (completed_targets.sum(axis=1) > max_exposure + 1e-12).any()):
        raise ValueError("Spot targets must align, be finite, nonnegative and within caps")
    targets = completed_targets.shift(1 + signal_delay_bars).fillna(0).to_numpy(dtype=float)
    if completed_buy_mask is None:
        buy_allowed = np.ones_like(targets, dtype=bool)
    else:
        if (list(completed_buy_mask.columns) != symbols or not completed_buy_mask.index.equals(first)
                or not completed_buy_mask.dtypes.eq(bool).all() or completed_buy_mask.isna().any().any()):
            raise ValueError("Completed buy mask must be aligned, nonnullable boolean data")
        buy_allowed = completed_buy_mask.shift(1 + signal_delay_bars, fill_value=False).to_numpy(dtype=bool)
    begin = pd.Timestamp(start) if start is not None else index[0]
    finish = pd.Timestamp(end) if end is not None else index[-1] + pd.Timedelta("4h")
    locations = np.flatnonzero((index >= begin) & (index < finish))
    if len(locations) == 0:
        raise ValueError("Empty spot evaluation period")
    params = EngineParams(fee_rate=fee_rate, slippage_bps=slippage_bps, spread_bps=spread_bps,
                          min_notional=min_notional, allow_loss_exits=True, execution_policy="market")
    cash = float(initial_usdt)
    holdings = [PortfolioState(cash, 0, cash, 0) for _ in symbols]
    peak = observed_peak = bound_peak = float(initial_usdt)
    equity_rows, trade_rows = [], []
    for i in locations:
        ts, prices = index[i], arrays["open"][i]
        qty = np.array([h.base for h in holdings])
        nav = cash + qty @ prices
        if nav <= 0 or cash < -1e-8 or (qty < -1e-12).any():
            raise ValueError("Spot-only balance invariant violated")
        observed_peak = max(observed_peak, nav)
        bound_peak = max(bound_peak, nav)
        open_dd, open_bound_dd = nav / observed_peak - 1, nav / bound_peak - 1
        decision_peak = bound_peak
        current = qty * prices / nav
        desired = targets[i]
        decision = not daily_control or ts.hour == 4 * signal_delay_bars
        rebalance = not daily_control or ts.dayofweek == 0
        final = current.copy()
        if decision:
            final = desired.copy() if rebalance else current.copy()
            inactive = desired <= 0
            final[inactive] = 0
            final = np.minimum(final, asset_cap)
            if final.sum() > max_exposure:
                final *= max_exposure / final.sum()
            # Compare to actual owned positions, not a signal-state approximation.
            # Block only increases; preserve inactivity exits and required cap sales.
            final = np.where(buy_allowed[i], final, np.minimum(final, current))
            cap_reduction = (current > asset_cap + 1e-12) | (current.sum() > max_exposure + 1e-12)
            for j in np.argsort(final - current, kind="stable"):
                if not inactive[j] and not cap_reduction[j] and abs(final[j] - current[j]) < rebalance_band:
                    continue
                state = replace(holdings[j], usdt=cash, equity=nav, exposure=float(current[j]))
                plan = plan_trade(state, prices[j], final[j], min_notional, allow_loss_exits=True)
                if not buy_allowed[i, j] and plan.delta_base > 0:
                    continue
                plan = replace(plan, order_type="market", limit_price=None)
                fill = simulate_execution(plan, prices[j], params, portfolio=state)
                updated = apply_fill(state, fill)
                cash, holdings[j] = updated.usdt, updated
                if fill.status == "filled" and fill.filled_base != 0:
                    trade_rows.append({"timestamp": ts, "symbol": symbols[j],
                        "side": "buy" if fill.filled_base > 0 else "sell", "qty": abs(fill.filled_base),
                        "price": fill.avg_price, "mid_price": float(prices[j]),
                        "notional": abs(fill.filled_base) * fill.avg_price, "fee": fill.fee_paid,
                        "slippage": fill.slippage_paid, "cash_after": cash, "base_after": updated.base})
        qty = np.array([h.base for h in holdings])
        post_nav = cash + qty @ prices
        observed_peak = max(observed_peak, post_nav)
        post_drawdown = post_nav / observed_peak - 1
        high = cash + qty @ arrays["high"][i]
        low = cash + qty @ arrays["low"][i]
        bound_peak = max(bound_peak, high)
        bound_dd = min(open_bound_dd, post_nav / decision_peak - 1, low / bound_peak - 1)
        equity = cash + qty @ arrays["close"][i]
        peak, observed_peak = max(peak, equity), max(observed_peak, equity)
        equity_rows.append({"timestamp": ts, "equity": equity, "usdt": cash,
            "drawdown": equity / peak - 1,
            "observed_drawdown": min(open_dd, post_drawdown, equity / observed_peak - 1),
            "intraday_drawdown_bound": bound_dd, "high_bound": high, "low_bound": low,
            "exposure": (equity - cash) / equity,
            **{f"base_{s}": qty[j] for j, s in enumerate(symbols)}})
    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows, columns=["timestamp", "symbol", "side", "qty", "price", "mid_price",
                                              "notional", "fee", "slippage", "cash_after", "base_after"])
    net = float(equity.equity.iloc[-1]) - initial_usdt
    fees, impact = float(trades.fee.sum()), float(trades.slippage.sum())
    years = len(locations) / (6 * 365.25)
    daily = equity.set_index("timestamp").equity.resample("1D").last()
    returns = daily.pct_change(fill_method=None)
    returns.iloc[0] = daily.iloc[0] / initial_usdt - 1
    vol = float(returns.std(ddof=0))
    summary = {"final_equity": float(equity.equity.iloc[-1]), "net_pnl": net,
        "total_return": net / initial_usdt, "cagr": (1 + net / initial_usdt) ** (1 / years) - 1,
        "fees_paid_total": fees, "slippage_paid_total": impact, "gross_pnl_cost_addback": net + fees + impact,
        "maxDD": float(equity.drawdown.min()), "max_observed_drawdown": float(equity.observed_drawdown.min()),
        "max_intraday_drawdown_bound": float(equity.intraday_drawdown_bound.min()),
        "trades_count": len(trades), "turnover": float(trades.notional.sum()) / initial_usdt,
        "mean_exposure": float(equity.exposure.mean()), "max_close_exposure": float(equity.exposure.max()),
        "minimum_cash": float(equity.usdt.min()),
        "minimum_inventory": float(equity[[f"base_{s}" for s in symbols]].min().min()),
        "sharpe": float(returns.mean() / vol * np.sqrt(365)) if vol > 0 else 0.,
        "rolling_gains": rolling_gains(daily, initial_usdt)}
    return equity, trades, summary


def rolling_gains(daily, initial_usdt):
    """Calendar-day windows on one continuous cash account, including initial NAV."""
    values = np.r_[initial_usdt, daily.to_numpy(dtype=float)]
    result = {}
    for days in (7, 30, 90):
        gains = values[days:] / values[:-days] - 1 if len(values) > days else np.array([])
        result[str(days)] = {"windows": len(gains), "doublings": int((gains >= 1).sum()),
            "doubling_frequency": float((gains >= 1).mean()) if len(gains) else None,
            "maximum_return": float(gains.max()) if len(gains) else None,
            "minimum_return": float(gains.min()) if len(gains) else None}
    return result


def reconcile_spot_path(markets, equity, trades, initial_usdt=1000.):
    """Independent cash-flow/inventory ledger; verify every close, not cost addback."""
    index = pd.DatetimeIndex(equity.timestamp).as_unit("ns")
    cash_flow = pd.Series(0., index=index)
    inventory = {s: pd.Series(0., index=index) for s in markets}
    for row in trades.itertuples(index=False):
        signed_qty = row.qty if row.side == "buy" else -row.qty
        cash_flow.loc[row.timestamp] -= signed_qty * row.price + row.fee
        inventory[row.symbol].loc[row.timestamp] += signed_qty
    cash = initial_usdt + cash_flow.cumsum()
    expected = cash.copy()
    inventory_error = 0.
    for symbol, changes in inventory.items():
        held = changes.cumsum()
        expected += held * markets[symbol].close.reindex(index).to_numpy()
        inventory_error = max(inventory_error, float(np.max(np.abs(held.to_numpy() - equity[f"base_{symbol}"].to_numpy()))))
    cash_error = float(np.max(np.abs(cash.to_numpy() - equity.usdt.to_numpy())))
    equity_error = float(np.max(np.abs(expected.to_numpy() - equity.equity.to_numpy())))
    if cash_error > 1e-7 or equity_error > 1e-7 or inventory_error > 1e-9:
        raise ValueError("Spot ledger does not reconcile")
    if cash.min() < -1e-7 or any(changes.cumsum().min() < -1e-9 for changes in inventory.values()):
        raise ValueError("Independent ledger detected borrowing or shorting")
    return {"max_cash_error": cash_error, "max_inventory_error": inventory_error,
            "max_equity_error": equity_error, "reconciled": True}
