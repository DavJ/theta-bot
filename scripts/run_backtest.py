from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from spot_bot.backtest.fast_backtest import run_backtest


def _read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Run unified spot backtest on OHLCV CSV.")
    p.add_argument("--csv", required=True, help="OHLCV CSV (timestamp,open,high,low,close,volume).")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--strategy", default="kalman_mr_dual")

    p.add_argument("--psi_mode", default="none")
    p.add_argument("--psi_window", type=int, default=200)
    p.add_argument("--rv_window", type=int, default=500)
    p.add_argument("--conc_window", type=int, default=200)
    p.add_argument("--base", type=float, default=2.0)

    p.add_argument("--max_exposure", type=float, default=1.0)
    p.add_argument("--initial_usdt", type=float, default=1000.0)
    p.add_argument("--min_notional", type=float, default=5.0)
    p.add_argument("--step_size", type=float, default=None)
    p.add_argument("--bar_state", default="closed")

    p.add_argument("--fee_rate", type=float, default=0.001)
    p.add_argument("--slippage_bps", type=float, default=2.0)
    p.add_argument("--spread_bps", type=float, default=0.0)

    p.add_argument("--hyst_k", type=float, default=5.0)
    p.add_argument("--hyst_floor", type=float, default=0.02)
    p.add_argument("--k_vol", type=float, default=0.5)
    p.add_argument("--edge_bps", type=float, default=5.0)
    p.add_argument("--max_delta_e_min", type=float, default=0.3)
    p.add_argument("--alpha_floor", type=float, default=6.0)
    p.add_argument("--alpha_cap", type=float, default=6.0)
    p.add_argument(
        "--vol_hyst_mode",
        choices=["increase", "decrease", "none"],
        default="increase",
        help="Volatility hysteresis mode: increase (higher vol -> higher threshold), "
        "decrease (higher vol -> lower threshold), none (no vol adjustment)",
    )
    p.add_argument(
        "--rv_ref_window",
        type=int,
        default=None,
        help="Reference volatility window in bars (default: 30 days based on timeframe)",
    )
    p.add_argument(
        "--conf_power",
        type=float,
        default=1.0,
        help="Power to apply to confidence when scaling risk budget (default: 1.0)",
    )
    p.add_argument(
        "--hyst_conf_k",
        type=float,
        default=0.0,
        help="Confidence-based hysteresis adjustment coefficient (default: 0.0 = disabled)",
    )

    p.add_argument("--out_equity", default=None)
    p.add_argument("--out_trades", default=None)
    p.add_argument("--print_metrics", action="store_true", default=False)

    args = p.parse_args()
    df = _read_ohlcv(Path(args.csv))

    equity_df, trades_df, metrics = run_backtest(
        df=df,
        timeframe=args.timeframe,
        strategy_name=args.strategy,
        psi_mode=args.psi_mode,
        psi_window=args.psi_window,
        rv_window=args.rv_window,
        conc_window=args.conc_window,
        base=args.base,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        max_exposure=args.max_exposure,
        initial_usdt=args.initial_usdt,
        min_notional=args.min_notional,
        step_size=args.step_size,
        bar_state=args.bar_state,
        log=False,
        hyst_k=args.hyst_k,
        hyst_floor=args.hyst_floor,
        spread_bps=args.spread_bps,
        k_vol=args.k_vol,
        edge_bps=args.edge_bps,
        max_delta_e_min=args.max_delta_e_min,
        alpha_floor=args.alpha_floor,
        alpha_cap=args.alpha_cap,
        vol_hyst_mode=args.vol_hyst_mode,
        rv_ref_window=args.rv_ref_window,
        conf_power=args.conf_power,
        hyst_conf_k=args.hyst_conf_k,
    )

    if args.out_equity:
        Path(args.out_equity).parent.mkdir(parents=True, exist_ok=True)
        equity_df.to_csv(args.out_equity, index=False)
    if args.out_trades:
        Path(args.out_trades).parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(args.out_trades, index=False)

    if args.print_metrics:
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

