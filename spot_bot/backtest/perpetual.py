"""Offline USDT perpetual ledger with conservative hourly mark/funding bounds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from spot_bot.portfolio.risk import PortfolioRisk


def apply_perpetual_fill(quantity, entry, delta, price):
    """Return signed inventory, average entry and realized P&L; no fees here."""
    if not np.isfinite([quantity, entry, delta, price]).all() or price <= 0 or entry < 0:
        raise ValueError("Finite inventory/entry and positive fill required")
    new = quantity + delta
    if quantity == 0 or quantity * delta >= 0:
        average = ((abs(quantity) * entry + abs(delta) * price) / abs(new)) if new else 0.
        return new, average, 0.
    reduced = min(abs(quantity), abs(delta))
    realized = reduced * np.sign(quantity) * (price - entry)
    average = 0. if abs(new) < 1e-12 else (price if quantity * new < 0 else entry)
    return (0. if abs(new) < 1e-12 else new), average, float(realized)


def funding_payment_bound(before, after, rate, high, low):
    """Worse possible payment for the before/after settlement-hour inventory."""
    if not np.isfinite([before, after, rate, high, low]).all() or not 0 < low <= high:
        raise ValueError("Valid funding quantities/rate/mark bounds required")
    amounts = [q * rate * (high if q * rate >= 0 else low) for q in (before, after)]
    return float(max(amounts))


def _frame(frame, frequency):
    result = frame.copy()
    if not isinstance(result.index, pd.DatetimeIndex) or result.index.tz is None:
        raise ValueError("Timezone-aware market grid required")
    result.index = result.index.tz_convert("UTC").astype("datetime64[ns, UTC]")
    if (result.empty or not result.index.is_unique or not result.index.is_monotonic_increasing
            or not result.index.to_series().diff().dropna().eq(pd.Timedelta(frequency)).all()
            or not {"open", "high", "low", "close"}.issubset(result)):
        raise ValueError("Complete ordered OHLC grid required")
    v = result[["open", "high", "low", "close"]]
    if (not np.isfinite(v.to_numpy()).all() or (v <= 0).any().any()
            or (result.low > result[["open", "close"]].min(axis=1)).any()
            or (result.high < result[["open", "close"]].max(axis=1)).any()):
        raise ValueError("Invalid perpetual OHLC")
    return result


def run_perpetual_backtest(execution, marks, funding, targets, signal_close, *, start=None, end=None,
                          initial_usdt=1000., fee_rate=.001, slippage_bps=5., spread_bps=2.,
                          min_notional=10., gross_cap=1., asset_cap=.5, rebalance_band=.01,
                          daily_rebalance=True, volatility_target=.20, maintenance_ratio=.05,
                          uniform_cap_scaling=True):
    if (not execution or set(execution) != set(marks) or set(execution) != set(funding)
            or not np.isfinite([initial_usdt, fee_rate, slippage_bps, spread_bps, min_notional,
                                  gross_cap, asset_cap, rebalance_band, maintenance_ratio]).all()
            or initial_usdt <= 0 or not 0 <= fee_rate < .5
            or not 0 <= slippage_bps < 5000 or not 0 <= spread_bps < 5000
            or min_notional < 0 or not 0 < asset_cap <= gross_cap <= 1
            or not 0 <= rebalance_band <= gross_cap or not 0 < maintenance_ratio < 1
            or (volatility_target is not None and (not np.isfinite(volatility_target) or volatility_target <= 0))):
        raise ValueError("Invalid perpetual research inputs/costs/budgets")
    symbols = list(execution)
    execution = {s: _frame(f, "1D") for s, f in execution.items()}
    marks = {s: _frame(f, "1h") for s, f in marks.items()}
    daily, hourly = execution[symbols[0]].index, marks[symbols[0]].index
    if (not daily.equals(daily.floor("D")) or any(not f.index.equals(daily) for f in execution.values())
            or any(not f.index.equals(hourly) for f in marks.values())):
        raise ValueError("Perpetual assets require identical complete grids")
    expected = pd.date_range(daily[0], daily[-1] + pd.Timedelta("1D"), freq="1h", inclusive="left").astype("datetime64[ns, UTC]")
    if not hourly.equals(expected):
        raise ValueError("Hourly marks must cover every complete execution day")
    targets, signal_close = targets.copy(), signal_close.copy()
    for f in (targets, signal_close):
        if not isinstance(f.index, pd.DatetimeIndex) or f.index.tz is None:
            raise ValueError("Timezone-aware signal grid required")
        f.index = f.index.tz_convert("UTC").astype("datetime64[ns, UTC]")
    if (not targets.index.equals(daily) or not signal_close.index.equals(daily)
            or list(targets.columns) != symbols or list(signal_close.columns) != symbols
            or not np.isfinite(targets.to_numpy()).all() or not np.isfinite(signal_close.to_numpy()).all()
            or (signal_close <= 0).any().any()
            or targets.abs().sum(axis=1).max() > gross_cap + 1e-12
            or targets.abs().max().max() > asset_cap + 1e-12):
        raise ValueError("Identical finite completed targets within signed caps required")
    desired = targets.shift(1).fillna(0.).to_numpy()
    covariance = PortfolioRisk(volatility_target=volatility_target).covariances(signal_close) if volatility_target else None
    prices = np.stack([execution[s].open.to_numpy() for s in symbols], axis=1)
    mark = {c: np.stack([marks[s][c].to_numpy() for s in symbols], axis=1) for c in ("open", "high", "low", "close")}
    events = {}
    for j, symbol in enumerate(symbols):
        f = funding[symbol].copy()
        if not {"event_time", "last_funding_rate"}.issubset(f):
            raise ValueError("Actual funding timestamps and rates required")
        f.event_time = pd.to_datetime(f.event_time, utc=True, format="mixed").astype("datetime64[ns, UTC]")
        f.last_funding_rate = pd.to_numeric(f.last_funding_rate, errors="raise").astype(float)
        if (not f.event_time.is_unique or not f.event_time.is_monotonic_increasing
                or not np.isfinite(f.last_funding_rate).all() or (f.last_funding_rate.abs() > .5).any()
                or not f.event_time.dt.floor("h").isin(hourly).all()):
            raise ValueError("Invalid or uncovered actual funding events")
        slots = hourly.get_indexer(pd.DatetimeIndex(f.event_time.dt.floor("h")))
        for slot, ts, rate in zip(slots, f.event_time, f.last_funding_rate):
            events.setdefault(int(slot), []).append((j, ts, float(rate)))
    start = pd.to_datetime(start, utc=True) if start is not None else daily[0]
    end = pd.to_datetime(end, utc=True) if end is not None else daily[-1] + pd.Timedelta("1D")
    if (start != start.floor("D") or end != end.floor("D") or start >= end
            or start < daily[0] or end > daily[-1] + pd.Timedelta("1D")):
        raise ValueError("Nonempty complete-day evaluation interval required")
    rows = np.flatnonzero((hourly >= start) & (hourly < end))
    if len(rows) == 0 or len(rows) % 24:
        raise ValueError("No complete perpetual evaluation days")
    q, entry = np.zeros(len(symbols)), np.zeros(len(symbols))
    cash, realized_total, fees_total, funding_total = float(initial_usdt), 0., 0., 0.
    observed_peak = peak_bound = close_peak = float(initial_usdt)
    trades, settlements, equities = [], [], []
    margin_breaches, max_hourly_gross, minimum_cash, negative_cash_hours = 0, 0., initial_usdt, 0
    for h in rows:
        ts, d = hourly[h], h // 24
        hour_minimum_cash = cash
        nav = cash + float(q @ (mark["open"][h] - entry))
        if nav <= 0:
            raise ValueError("Perpetual account exhausted; no liquidation outcome fabricated")
        observed_peak, peak_bound = max(observed_peak, nav), max(peak_bound, nav)
        open_drawdown = nav / observed_peak - 1
        before = q.copy()
        if h % 24 == 0:
            day_observed, day_bound, day_low, day_high = nav / observed_peak - 1, nav / peak_bound - 1, nav, nav
            current = q * mark["open"][h] / nav
            requested = desired[d]
            rebalance = daily_rebalance or ts.dayofweek == 0
            final = requested.copy() if rebalance else current.copy()
            inactive = (requested == 0) | (requested * current < 0)
            if not rebalance:
                final[inactive] = 0.
            if uniform_cap_scaling:
                maximum, gross = np.abs(final).max(), np.abs(final).sum()
                scale = min(1., asset_cap / maximum if maximum else 1., gross_cap / gross if gross else 1.)
                final *= scale
            else:
                # Reproduce the original spot control's individual clipping.
                final = np.clip(final, -asset_cap, asset_cap)
                gross = np.abs(final).sum()
                if gross > gross_cap:
                    final *= gross_cap / gross
            risk_scale = 1.
            if covariance is not None:
                cov = covariance[d - 1] if d > 0 else None
                if cov is None or not np.isfinite(cov).all():
                    risk_scale = 0.
                else:
                    annual = float(np.sqrt(max(0., final @ cov @ final) * 365))
                    risk_scale = min(1., volatility_target / annual) if annual else 1.
                final *= risk_scale
            cap_reduction = (np.abs(current) > asset_cap + 1e-12) | (np.abs(current).sum() > gross_cap + 1e-12)
            risk_reduction = (np.abs(final) < np.abs(current) - 1e-12) & (risk_scale < 1)
            quantities = final * nav / mark["open"][h]
            order = sorted(range(len(symbols)), key=lambda j: abs(quantities[j]) - abs(q[j]))
            for j in order:
                if (abs(final[j] - current[j]) < rebalance_band and not inactive[j]
                        and not cap_reduction[j] and not risk_reduction[j]):
                    continue
                delta = float(quantities[j] - q[j])
                if abs(delta) * prices[d, j] < min_notional or abs(delta) < 1e-12:
                    continue
                fill = prices[d, j] * (1 + np.sign(delta) * (slippage_bps + spread_bps / 2) / 10000)
                fee = abs(delta) * fill * fee_rate
                q[j], entry[j], realized = apply_perpetual_fill(q[j], entry[j], delta, fill)
                cash += realized - fee
                hour_minimum_cash = min(hour_minimum_cash, cash)
                realized_total += realized
                fees_total += fee
                trades.append({"timestamp": ts, "symbol": symbols[j], "signed_delta": delta,
                               "price": fill, "reference_price": prices[d, j], "fee": fee,
                               "notional": abs(delta) * fill, "slippage": abs(delta) * abs(fill - prices[d, j]),
                               "realized_pnl": realized, "decision_nav": nav, "quantity_after": q[j]})
        post_fill = cash + float(q @ (mark["open"][h] - entry))
        observed_peak, peak_bound = max(observed_peak, post_fill), max(peak_bound, post_fill)
        post_fill_drawdown = post_fill / observed_peak - 1
        cash_before = cash
        paid_in_hour, received_in_hour = 0., 0.
        for j, event_at, rate in events.get(h, ()):
            payment = funding_payment_bound(before[j], q[j], rate, mark["high"][h, j], mark["low"][h, j])
            cash -= payment
            funding_total += payment
            paid_in_hour += max(0., payment)
            received_in_hour += max(0., -payment)
            settlements.append({"timestamp": event_at, "symbol": symbols[j], "rate": rate,
                                "quantity_before": before[j], "quantity_after": q[j], "payment": payment})
        upper_price = np.where(q >= 0, mark["high"][h], mark["low"][h])
        lower_price = np.where(q >= 0, mark["low"][h], mark["high"][h])
        # Different assets settle with millisecond jitter. Opposing transfers
        # can cancel in final cash; include every possible intermediate balance.
        high_nav = cash_before + received_in_hour + float(q @ (upper_price - entry))
        low_nav = cash_before - paid_in_hour + float(q @ (lower_price - entry))
        hour_minimum_cash = min(hour_minimum_cash, cash_before - paid_in_hour)
        peak_bound = max(peak_bound, high_nav)
        close_nav = cash + float(q @ (mark["close"][h] - entry))
        observed_peak = max(observed_peak, close_nav)
        if close_nav <= 0:
            raise ValueError("Perpetual NAV nonpositive; margin simulation invalid")
        day_observed = min(day_observed, open_drawdown, post_fill_drawdown,
                           close_nav / observed_peak - 1)
        day_bound = min(day_bound, low_nav / peak_bound - 1)
        day_low, day_high = min(day_low, low_nav), max(day_high, high_nav)
        gross_max = float(np.abs(q) @ mark["high"][h])
        margin_breaches += int(low_nav < maintenance_ratio * gross_max)
        minimum_cash = min(minimum_cash, hour_minimum_cash)
        negative_cash_hours += int(hour_minimum_cash < 0)
        max_hourly_gross = max(max_hourly_gross, float(np.abs(q) @ mark["close"][h]) / close_nav)
        if not np.isclose(cash, initial_usdt + realized_total - fees_total - funding_total, atol=1e-8, rtol=1e-12):
            raise AssertionError("Futures collateral cash ledger does not reconcile")
        if h % 24 == 23:
            close_peak = max(close_peak, close_nav)
            equities.append({"timestamp": ts.floor("D"), "equity": close_nav, "collateral_cash": cash,
                "unrealized_pnl": float(q @ (mark["close"][h] - entry)), "drawdown": close_nav / close_peak - 1,
                "observed_drawdown": day_observed, "intraday_drawdown_bound": day_bound,
                "hourly_low_nav_bound": day_low, "hourly_high_nav_bound": day_high,
                "gross_exposure": float(np.abs(q) @ mark["close"][h]) / close_nav,
                "net_exposure": float(q @ mark["close"][h]) / close_nav,
                "realized_pnl_cumulative": realized_total, "fees_cumulative": fees_total,
                "funding_net_paid_cumulative": funding_total,
                **{f"qty_{s}": q[j] for j, s in enumerate(symbols)},
                **{f"entry_{s}": entry[j] for j, s in enumerate(symbols)}})
    equity = pd.DataFrame(equities)
    trade = pd.DataFrame(trades, columns=["timestamp", "symbol", "signed_delta", "price", "reference_price", "fee",
                                        "notional", "slippage", "realized_pnl", "decision_nav", "quantity_after"])
    settled = pd.DataFrame(settlements, columns=["timestamp", "symbol", "rate", "quantity_before", "quantity_after", "payment"])
    net = float(equity.equity.iloc[-1]) - initial_usdt
    returns = equity.equity.pct_change(fill_method=None)
    returns.iloc[0] = equity.equity.iloc[0] / initial_usdt - 1
    volatility = float(returns.std(ddof=0))
    summary = {"final_equity": float(equity.equity.iloc[-1]), "total_return": net / initial_usdt, "net_pnl": net,
        "fees_paid_total": float(trade.fee.sum()), "slippage_paid_total": float(trade.slippage.sum()),
        "funding_net_paid_total": float(settled.payment.sum()), "funding_paid_total": float(settled.payment.clip(lower=0).sum()),
        "funding_received_total": -float(settled.payment.clip(upper=0).sum()),
        "maxDD": float(equity.drawdown.min()), "max_observed_drawdown": float(equity.observed_drawdown.min()),
        "max_intraday_drawdown_bound": float(equity.intraday_drawdown_bound.min()),
        "trades_count": len(trade), "settlement_events": len(settled), "hourly_marks_checked": len(rows),
        "margin_breach_hours": margin_breaches, "max_hourly_close_gross_exposure": max_hourly_gross,
        "minimum_collateral_cash": float(minimum_cash), "negative_collateral_hours": negative_cash_hours,
        "mean_exposure": float(equity.gross_exposure.mean()),
        "turnover": float(trade.notional.sum()) / initial_usdt,
        "sharpe": float(returns.mean() / volatility * np.sqrt(365)) if volatility else 0.}
    return equity, trade, settled, summary
