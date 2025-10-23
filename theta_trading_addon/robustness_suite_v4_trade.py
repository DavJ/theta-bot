#!/usr/bin/env python3
import argparse, subprocess, sys, os, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "theta_bot_averaging" / "theta_eval_hbatch_biquat_max.py"

def parse_symbols_mixed(symbols_arg: str):
    csv_paths, tickers = [], []
    for raw in [s.strip() for s in symbols_arg.split(",") if s.strip()]:
        if raw.lower().endswith(".csv"):
            csv_paths.append(raw)  # zachovat case
        else:
            tickers.append(raw.upper())
    return csv_paths, tickers

def ensure_prices(tickers, interval, limit=1000):
    if not tickers:
        return []
    maker = ROOT / "make_prices_csv.py"
    if not maker.exists():
        alt = ROOT.parent / "make_prices_csv.py"
        if alt.exists():
            maker = alt
        else:
            raise SystemExit("make_prices_csv.py nebyl nalezen v rootu.")
    cmd = [sys.executable, str(maker),
           "--symbols", ",".join(tickers),
           "--interval", interval,
           "--limit", str(limit),
           "--outdir", "prices"]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return [str(ROOT / "prices" / f"{t}_{interval}.csv") for t in tickers]

def sanitize_symbol_for_filename(p: str) -> str:
    # jak evaluátor pojmenovává: 'eval_h_<BASENAME s odstraněnými tečkami>/<podtržítky>.csv'
    base = os.path.basename(p)
    base = base.replace(".", "")
    return f"eval_h_{base}.csv"

def run_evaluator(csv_paths, args):
    out_dir = ROOT / "theta_trading_addon" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not EVAL_SCRIPT.exists():
        raise SystemExit(f"Evaluator {EVAL_SCRIPT} nebyl nalezen.")
    symbols_arg = ",".join([str(Path(p)) for p in csv_paths])
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--symbols", symbols_arg,
        "--interval", args.interval,
        "--window", str(args.window),
        "--horizon", str(args.horizon),
        "--minP", str(args.minP),
        "--maxP", str(args.maxP),
        "--nP", str(args.nP),
        "--sigma", str(args.sigma),
        "--lambda", str(args.lam),
        "--pred-ensemble", args.pred_ensemble,
        "--max-by", args.max_by,
        "--out", str(out_dir / "hbatch_biquat_summary.csv"),
    ]
    print("\n=== Spouštím evaluátor ===")
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    print("Uloženo:", out_dir / "hbatch_biquat_summary.csv")
    return out_dir / "hbatch_biquat_summary.csv"

def simulate_trades(perbar_df: pd.DataFrame, fees_bps: float, slippage_bps: float,
                    z_entry: float, z_window: int, position_cap: float,
                    tp_sigma: float, sl_sigma: float):
    df = perbar_df.copy()
    # fallback: pokud nejsou sloupce, pojmenujeme konzistentně
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc == "pred_delta": rename_map[col] = "pred_delta"
        if lc == "true_delta": rename_map[col] = "true_delta"
        if lc in ("last_price", "price", "close"): rename_map[col] = "price"
        if lc in ("entry_idx", "index"): rename_map[col] = "entry_idx"
        if lc in ("compare_idx", "exit_idx"): rename_map[col] = "compare_idx"
    df = df.rename(columns=rename_map)
    need = {"pred_delta", "true_delta", "price", "entry_idx", "compare_idx"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"Chybějící sloupce pro simulaci: {need - set(df.columns)}")

    # Z-score z predikovaných delt (rolling)
    df["pred_mean"] = df["pred_delta"].rolling(z_window, min_periods=z_window//2).mean()
    df["pred_std"] = df["pred_delta"].rolling(z_window, min_periods=z_window//2).std().replace(0, pd.NA)
    df["z"] = (df["pred_delta"] - df["pred_mean"]) / df["pred_std"]
    df["z"] = df["z"].fillna(0.0)

    # Simple horizon exit; TP/SL jen jako place-holder (bez intrabar H/L)
    in_pos = False
    pos_dir = 0
    entry_price = 0.0
    equity = 1.0
    records = []

    fee = fees_bps / 1e4
    slip = slippage_bps / 1e4

    for i, row in df.iterrows():
        price = row["price"]
        if not in_pos:
            if row["z"] >= z_entry:
                pos_dir = 1
                in_pos = True
                entry_price = price * (1 + slip)
            elif row["z"] <= -z_entry:
                pos_dir = -1
                in_pos = True
                entry_price = price * (1 - slip)
            else:
                continue
        else:
            # Exit na compare_idx (když dostupné)
            # Hledáme řádek, kde entry_idx == current.entry_idx a compare_idx existuje
            # V našem perbar je compare_idx vedle; použijeme přímo tento řádek jako exit bar.
            exit_price = price * (1 - slip) if pos_dir == 1 else price * (1 + slip)
            ret = (exit_price - entry_price) / entry_price if pos_dir == 1 else (entry_price - exit_price) / entry_price
            ret -= 2 * fee  # round-trip fee
            equity *= (1 + max(-position_cap, min(position_cap, ret)))
            records.append({
                "time": row.get("time", ""),
                "entry_idx": int(row["entry_idx"]),
                "exit_idx": int(row.get("compare_idx", row["entry_idx"])),
                "pos_dir": pos_dir,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "ret": ret,
                "equity": equity,
                "z": float(row["z"]),
            })
            in_pos = False
            pos_dir = 0
            entry_price = 0.0

    trades = pd.DataFrame.from_records(records)
    if trades.empty:
        return trades, {"trades": 0, "total_return": 0.0, "sharpe_trades": 0.0, "max_drawdown": 0.0}

    trades["pnl"] = trades["ret"]
    total_return = trades["equity"].iloc[-1] - 1.0
    hit_rate = (trades["pnl"] > 0).mean()
    # jednoduchá sharpe přes obchody
    if trades["pnl"].std() and trades["pnl"].std() > 0:
        sharpe_trades = trades["pnl"].mean() / trades["pnl"].std() * (len(trades) ** 0.5)
    else:
        sharpe_trades = 0.0
    # max DD
    eq = trades["equity"]
    running_max = eq.cummax()
    dd = (eq / running_max - 1.0).min()

    summary = {
        "trades": int(len(trades)),
        "hit_rate": float(hit_rate),
        "total_return": float(total_return),
        "sharpe_trades": float(sharpe_trades),
        "max_drawdown": float(dd),
    }
    return trades, summary

def load_perbar_for(csv_path: str):
    # evaluátor vytváří soubor typu eval_h_<basename without dots>.csv v CWD (ROOT)
    fname = sanitize_symbol_for_filename(csv_path)
    # zkusíme v ROOT a v theta_bot_averaging
    candidates = [
        ROOT / fname,
        ROOT / "theta_bot_averaging" / fname,
        ROOT / "theta_trading_addon" / "results" / fname,
    ]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c)
    # fallback: glob
    glob = list(ROOT.glob(f"**/{fname}"))
    if glob:
        return pd.read_csv(glob[0])
    raise FileNotFoundError(f"Per-bar CSV nenalezen: {fname}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Tickery (BTCUSDT,ETHUSDT) nebo CSV cesty (prices/...csv) nebo mix")
    ap.add_argument("--interval", required=True)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--minP", type=int, default=24)
    ap.add_argument("--maxP", type=int, default=480)
    ap.add_argument("--nP", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--lam", dest="lam", type=float, default=1e-3)
    ap.add_argument("--pred-ensemble", default="avg", choices=["avg", "max"])
    ap.add_argument("--max-by", default="transform", choices=["transform", "contrib"])
    ap.add_argument("--fees-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--z-entry", type=float, default=0.8)
    ap.add_argument("--z-window", type=int, default=96)
    ap.add_argument("--position-cap", type=float, default=1.0)
    ap.add_argument("--tp-sigma", type=float, default=1.25)
    ap.add_argument("--sl-sigma", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    (ROOT / "prices").mkdir(exist_ok=True)
    (ROOT / "theta_trading_addon" / "results").mkdir(parents=True, exist_ok=True)

    csv_paths, tickers = parse_symbols_mixed(args.symbols)
    # auto-download, pokud jsou tickery
    csv_paths += ensure_prices(tickers, args.interval, args.limit)

    if not csv_paths:
        print("[warn] No CSV paths to evaluate. Konec.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(args.out, index=False)
        print("Saved:", args.out)
        return

    # 1) Spustit evaluátor (generuje eval_h_*.csv)
    run_evaluator(csv_paths, args)

    # 2) Per-symbol trading simulace
    rows = []
    for p in csv_paths:
        print(f"\n=== Trading simulace pro {p} ===\n")
        try:
            perbar = load_perbar_for(p)
        except FileNotFoundError as e:
            print("[warn]", e)
            continue
        trades, summ = simulate_trades(
            perbar,
            fees_bps=args.fees_bps,
            slippage_bps=args.slippage_bps,
            z_entry=args.z_entry,
            z_window=args.z_window,
            position_cap=args.position_cap,
            tp_sigma=args.tp_sigma,
            sl_sigma=args.sl_sigma,
        )
        sym = os.path.basename(p).split("_")[0]
        trades_out = ROOT / "theta_trading_addon" / "results" / f"trades_{sym}.csv"
        trades.to_csv(trades_out, index=False)
        print("Uloženo trades:", trades_out)
        row = {"symbol": sym, **summ}
        rows.append(row)

    df_sum = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(args.out, index=False)
    print(df_sum if not df_sum.empty else "Empty summary")
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
