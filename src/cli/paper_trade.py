# Entry point for paper trading (historical or live)
import os, time, json
from paper_trading.runner import PaperRunner

def parse_symbols():
    syms = os.environ.get("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
    return [s.strip() for s in syms if s.strip()]

def main():
    cfg = {
        "mode": os.environ.get("MODE", "live"),  # live | historical
        "symbols": parse_symbols(),
        "interval": os.environ.get("INTERVAL", "1m"),
        "loop_seconds": int(os.environ.get("LOOP_SECONDS", "30")),
        "fee_bps": float(os.environ.get("FEE_BPS", "4")),
        "slip_bps": float(os.environ.get("SLIP_BPS", "1")),
        "max_w_per_symbol": float(os.environ.get("MAX_W_PER_SYMBOL", "0.5")),
        "hedge_ratio": float(os.environ.get("HEDGE_RATIO", "0.5")),
        "base_equity": float(os.environ.get("BASE_EQUITY", "1000")),
        "cooldown_steps": int(os.environ.get("COOLDOWN_STEPS", "2")),
        "csv_path": os.environ.get("CSV_PATH", ""),
        "binance_api_key": os.environ.get("BINANCE_API_KEY", ""),
        "binance_api_secret": os.environ.get("BINANCE_API_SECRET", ""),
        "trades_path": "paper_trades.csv",
        "equity_path": "equity_curve.csv",
    }
    runner = PaperRunner(cfg)
    if cfg["mode"] == "historical":
        runner.run_historical()
    else:
        runner.run_live()

if __name__ == "__main__":
    main()
