from pathlib import Path
import pandas as pd

from spot_bot.backtest.fast_backtest import run_backtest

DATA_DIR = Path("data/ohlcv_1h")
SYMS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOTUSDT"]

# --- nastav si jedny společné parametry ---
TIMEFRAME = "1h"
STRATEGY = "kalman_mr_dual"     # případně uprav na to, co reálně používáš
PSI_MODE = "none"              # nebo "rolling" podle toho, co máte
PSI_WINDOW = 200
RV_WINDOW = 500
CONC_WINDOW = 200
BASE = 2.0
FEE_RATE =  0.001
SLIPPAGE_BPS =  2.0
SPREAD_BPS = 0.0
MAX_EXPOSURE = 1.0
INITIAL_USDT = 1000.0
MIN_NOTIONAL = 5.0
STEP_SIZE = None
HYST_K = 25.0    # 15.0
HYST_FLOOR = 0.30    # 0.12
K_VOL = 0.5 # multiplikativni parametr kontrolujici velikost dynamicky vypocitane hystereze
EDGE_BPS = 5.0 # minimalni zisk v procentech ...> kontroluje hysterezi (aditivni parametr)
MAX_DELTA_E_MIN = 0.3  # maximum cap for hysteresis threshold

def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # timestamp already ISO+00:00 in your files
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df

def main():
    rows = []
    for sym in SYMS:
        csv_path = DATA_DIR / f"{sym}.csv"
        if not csv_path.exists():
            print(f"[SKIP] missing {csv_path}")
            continue

        df = load_df(csv_path)

        equity_curve, trades, metrics = run_backtest(
            df=df,
            timeframe=TIMEFRAME,
            strategy_name=STRATEGY,
            psi_mode=PSI_MODE,
            psi_window=PSI_WINDOW,
            rv_window=RV_WINDOW,
            conc_window=CONC_WINDOW,
            base=BASE,
            fee_rate=FEE_RATE,
            slippage_bps=SLIPPAGE_BPS,
            max_exposure=MAX_EXPOSURE,
            initial_usdt=INITIAL_USDT,
            min_notional=MIN_NOTIONAL,
            step_size=STEP_SIZE,
            bar_state="closed",
            log=False,
            hyst_k=HYST_K,
            hyst_floor=HYST_FLOOR,
            spread_bps=SPREAD_BPS,
            k_vol=K_VOL,
            edge_bps=EDGE_BPS,
            max_delta_e_min=MAX_DELTA_E_MIN,
        )

        final_eq = float(metrics.get("final_equity", equity_curve["equity"].iloc[-1]))
        trades_n = len(trades)
        ret = (final_eq / INITIAL_USDT - 1.0) * 100.0

        print(f"[{sym}] final={final_eq:.2f} return={ret:.2f}% trades={trades_n}")
        rows.append((sym, final_eq, ret, trades_n))

    print("\n==== SUMMARY ====")
    for sym, final_eq, ret, n in rows:
        print(f"{sym:7s}  final={final_eq:10.2f}  return={ret:8.2f}%  trades={n:5d}")

if __name__ == "__main__":
    main()

