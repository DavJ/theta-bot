# Copilot Instructions — Log-Phase / Torus Algorithm  
## Binance BTC/USDT · 1h timeframe

You are implementing an experimental **log-phase (fractional log) algorithm** inspired by a mathematical dream.

The core idea:
- compare numbers **only by the fractional part of their value or logarithm**
- interpret the fractional part as **phase / noise**
- treat this phase as living on a **circle (S¹)**, not a line
- measure similarity using **circular distance**
- measure structure using **phase clustering**
- use the result primarily as a **risk / regime filter**, not guaranteed alpha

---

## 1. Objective

Create a runnable Python script that:

1. Downloads **BTC/USDT** OHLCV data from **Binance**
2. Uses **1h timeframe**
3. Computes close-to-close returns
4. Computes **log-phase features** using the fractional part of the logarithm
5. Measures **rolling phase concentration**
6. Performs basic **uniformity diagnostics**
7. Runs a **simple risk/regime backtest**
8. Produces plots for inspection

---

## 2. Environment & Dependencies

- Python ≥ 3.10
- Libraries:
  - `ccxt`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`

❌ Do NOT use seaborn  
❌ Do NOT require API keys (public OHLCV only)

---

## 3. Data Specification

- Exchange: Binance
- Symbol: `BTC/USDT`
- Timeframe: `1h`
- Use `ccxt.binance(enableRateLimit=True)`
- Download **3000–6000 candles** using pagination (`since`)
- Required DataFrame columns:

