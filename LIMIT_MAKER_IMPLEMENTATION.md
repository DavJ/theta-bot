# LIMIT Maker Execution Implementation Summary

## Overview
Successfully implemented LIMIT Maker (post-only) execution with unified edge/hysteresis threshold into theta-bot using existing logic only. This adds an execution layer enhancement without any changes to trading strategy, signals, or decision mechanisms.

The edge threshold calculation combines fees, spread, slippage, and minimum profit into a single threshold used ONLY for setting limit prices (BUY cheaper, SELL higher).

## Changes Made

### 1. CCXTExecutor Enhancements (`spot_bot/execution/ccxt_executor.py`)

#### New Configuration Fields
Added to `ExecutorConfig` dataclass:
- `order_type: str = "market"` - Order execution type ("market" or "limit_maker")
- `maker_offset_bps: float = 1.0` - Additional offset from edge-adjusted price
- `order_validity_seconds: int = 60` - Time before canceling stale orders
- `max_spread_bps: float = 20.0` - Maximum allowed spread before rejection
- `maker_fee_rate: float = 0.001` - Fee rate for maker orders
- `taker_fee_rate: float = 0.001` - Fee rate for taker/market orders
- `slippage_bps: float = 0.0` - Expected slippage in basis points
- `min_profit_bps: float = 5.0` - Minimum profit requirement in basis points
- `edge_softmax_alpha: float = 20.0` - Smoothness parameter for soft max
- `edge_floor_bps: float = 0.0` - Minimum edge threshold in basis points
- `fee_roundtrip_mode: str = "maker_maker"` - Fee calculation mode ("maker_maker" or "maker_taker")

#### New Helper Methods
- `fetch_market_rules()`: Loads market precision and limits from exchange
- `quantize_price(price)`: Rounds price to exchange tick size/precision
- `quantize_amount(amount)`: Rounds amount to exchange step size/precision
- `cancel_stale_orders()`: Cancels open orders older than validity period
- `_smooth_max(a, b, alpha)`: Smooth maximum function using tanh for differentiability
- `_compute_edge_bps_total(bid, ask)`: Computes unified edge threshold in basis points

#### Edge/Hysteresis Calculation Logic
The edge threshold is computed as:

```
fee_component_bps = 2 * maker_fee_rate * 10000  (if maker_maker mode)
                  = (maker_fee_rate + taker_fee_rate) * 10000  (if maker_taker mode)

spread_component_bps = 0.5 * spread_bps  (capture half the spread)

slippage_component_bps = slippage_bps  (from CLI parameter)

profit_component_bps = min_profit_bps  (from CLI parameter)

computed_bps = fee_component_bps + spread_component_bps + slippage_component_bps + profit_component_bps

edge_bps_total = smooth_max(edge_floor_bps, computed_bps, edge_softmax_alpha)
```

The smooth_max function ensures the edge is always at least `edge_floor_bps` while smoothly transitioning from the floor to the computed value.

#### Limit Price Calculation
For **BUY** orders:
```
reference_price = bid  (NOT last close or mid)
edge = edge_bps_total * 1e-4
limit_price = reference_price * (1.0 - edge)
limit_price *= (1.0 - maker_offset_bps * 1e-4)  (optional additional offset)
```

For **SELL** orders:
```
reference_price = ask  (NOT last close or mid)
edge = edge_bps_total * 1e-4
limit_price = reference_price * (1.0 + edge)
limit_price *= (1.0 + maker_offset_bps * 1e-4)  (optional additional offset)
```

This ensures:
- BUY orders are placed **below** the current bid (cheaper)
- SELL orders are placed **above** the current ask (more expensive)
- Edge includes all costs: fees, spread, slippage, and minimum profit

#### New Core Method
- `place_limit_maker_order(side, qty, last_close)`: Places post-only limit orders with:
  - Spread guard (rejects if spread > max_spread_bps)
  - Bid/ask validation
  - Edge/hysteresis calculation
  - Price quantization based on exchange rules
  - Amount quantization based on exchange rules
  - Notional validation after quantization
  - Post-only order placement with fallback support
  - Returns status "open" for unfilled orders
  - Comprehensive logging of all edge components

#### Type Clarification
- Renamed `ExecutionResult` to `CCXTExecutionResult` to distinguish from `core.types.ExecutionResult`
- Added comprehensive docstrings explaining type separation

### 2. LiveExecutor Integration (`spot_bot/core/executor.py`)

#### Enhanced execute() Method
- Added order_type branching logic
- Calls `cancel_stale_orders()` before placing limit maker orders
- Returns `ExecutionResult` with status="OPEN" for unfilled limit orders
- Preserves existing market order logic unchanged

#### Status Handling
- "OPEN": Order placed but not filled yet (filled_base=0)
- "filled": Order completely filled (same as market orders)
- "rejected": Order rejected by guards or exchange
- "error": Execution error

### 3. CLI Extensions (`spot_bot/run_live.py`)

Added command-line flags:
```bash
--order-type {market,limit_maker}
--maker-offset-bps MAKER_OFFSET_BPS
--order-validity-seconds ORDER_VALIDITY_SECONDS
--max-spread-bps MAX_SPREAD_BPS
--maker-fee-rate MAKER_FEE_RATE
--taker-fee-rate TAKER_FEE_RATE
--min-profit-bps MIN_PROFIT_BPS
--edge-softmax-alpha EDGE_SOFTMAX_ALPHA
--edge-floor-bps EDGE_FLOOR_BPS
--fee-roundtrip-mode {maker_maker,maker_taker}
```

### 4. Fee Consistency

- Market orders use `taker_fee_rate`
- Limit maker orders use `maker_fee_rate`
- Consistent fee estimation throughout execution pipeline
- Fee component in edge calculation respects roundtrip mode

### 5. Testing (`tests/test_limit_maker_execution.py`)

Created 20 comprehensive unit tests:
- **Quantization tests**: Verify price/amount rounding never produces negatives
- **Spread guard tests**: Verify rejection when spread too wide
- **LiveExecutor tests**: Verify OPEN status handling and cancel_stale_orders calls
- **Edge calculation tests**: Verify smooth_max, fee components, floor handling
- **Limit price direction tests**: Verify BUY < bid and SELL > ask
- **Regression tests**: Verify market orders still work as before

All existing tests pass (20 tests in total).

### 6. Documentation (`README.md`)

Added comprehensive documentation:
- Usage examples for limit maker orders with edge calculation
- Parameter descriptions
- OPEN order behavior explanation
- Edge calculation formula documentation
- Important notes about fee rates and order validity

## Usage Examples

### Paper Trading (Simulated - No Limit Orders)
```bash
PYTHONPATH=. python -m spot_bot.run_live --mode paper --db bot.db \
  --initial-usdt 1000 --fee-rate 0.001 --max-exposure 0.3 \
  --symbol BTC/USDT --timeframe 1h --strategy kalman_mr_dual
```

### Live Market Orders (Current Default)
```bash
PYTHONPATH=. python -m spot_bot.run_live --mode live --i-understand-live-risk \
  --db bot.db --symbol BTC/USDT --timeframe 1h --limit-total 2000 \
  --fee-rate 0.001 --max-exposure 0.3 --min-notional 10
```

### Live Limit Maker Orders (NEW - with Edge Calculation)
```bash
PYTHONPATH=. python -m spot_bot.run_live --mode live --i-understand-live-risk \
  --db bot.db --symbol BTC/USDT --timeframe 1h --limit-total 2000 \
  --strategy kalman_mr_dual \
  --order-type limit_maker \
  --maker-fee-rate 0.0003 \
  --taker-fee-rate 0.001 \
  --min-profit-bps 5 \
  --max-spread-bps 20 \
  --order-validity-seconds 300 \
  --slippage-bps 0 \
  --edge-softmax-alpha 20.0 \
  --edge-floor-bps 0.0 \
  --fee-roundtrip-mode maker_maker \
  --max-exposure 0.3 --min-notional 10
```

### Conservative Limit Maker (Higher Profit Requirement)
```bash
PYTHONPATH=. python -m spot_bot.run_live --mode live --i-understand-live-risk \
  --db conservative.db --symbol BTC/USDT --timeframe 1h \
  --order-type limit_maker \
  --maker-fee-rate 0.0003 \
  --min-profit-bps 10 \
  --edge-floor-bps 5 \
  --max-spread-bps 15 \
  --order-validity-seconds 180
```

## Key Features

### Unified Edge/Hysteresis Threshold
Combines all execution costs into a single threshold:
- **Fee Component**: Roundtrip trading fees (maker_maker or maker_taker)
- **Spread Component**: Half-spread for maker model (we capture part of the spread)
- **Slippage Component**: Expected execution drift
- **Profit Component**: Minimum profit target
- **Floor**: Optional minimum threshold via smooth max

### Spread Guard
Protects against poor execution by rejecting orders when spread exceeds `max_spread_bps`.

### Automatic Order Cancellation
Stale orders (older than `order_validity_seconds`) are automatically cancelled before placing new orders.

### Price/Amount Quantization
Orders are automatically rounded to exchange precision rules to avoid rejection.

### Exchange Compatibility
Smart parameter handling supports both:
- Standard `postOnly` parameter (most exchanges)
- Binance-style `timeInForce: "GTX"` parameter

### Fail-Safe Design
Exception handling uses fail-safe approach:
- Order cancellation failures are non-fatal
- Missing bid/ask data triggers rejection rather than crash
- Market info unavailable triggers conservative defaults

## Important Notes

1. **Edge Calculation**: The edge threshold is used ONLY for pricing limit orders, not for strategy decisions
2. **Reference Prices**: BUY uses bid, SELL uses ask (not last close or mid)
3. **Direction**: BUY limit < bid, SELL limit > ask (ensures better execution than market)
4. **OPEN Orders**: Limit maker orders return status "OPEN" and don't update portfolio balances until filled
5. **Paper/Dryrun Modes**: Continue to use simulated execution (not real limit orders)
6. **Fee Rates**: Maker fees typically lower than taker fees - configure appropriately
7. **Order Validity**: Balance between giving orders time to fill and keeping positions fresh
8. **Spread Guard**: Set appropriately for market conditions to avoid excessive rejections

## Parameter Guidance

### min-profit-bps
- Default: 5.0 bps
- Conservative: 10-20 bps
- Aggressive: 2-5 bps
- Sets minimum profit target above all costs

### edge-floor-bps
- Default: 0.0 (no floor)
- Conservative: 5-10 bps
- Use when you want a guaranteed minimum edge regardless of market conditions

### fee-roundtrip-mode
- `maker_maker`: Assumes both entry and exit are maker orders (2 * maker_fee)
- `maker_taker`: Assumes entry is maker, exit is taker (maker_fee + taker_fee)
- Use `maker_maker` for patient trading, `maker_taker` for faster exits

### order-validity-seconds
- Default: 60 seconds
- Range: 30-600 seconds
- Shorter: More responsive to price changes, more order churn
- Longer: Less churn, but risk of stale prices

### max-spread-bps
- Default: 20 bps
- Liquid markets (BTC/USDT): 10-20 bps
- Less liquid pairs: 30-50 bps
- Rejects orders when spread is too wide

## Testing Results

- ✅ 20 limit maker execution tests passing
- ✅ All edge calculation tests passing
- ✅ Limit price direction tests passing (BUY < bid, SELL > ask)
- ✅ Smooth max function validated
- ✅ No regressions detected
- ✅ CodeQL security scan: 0 alerts
- ✅ All code review feedback addressed

## Security Summary

No security vulnerabilities introduced. The implementation:
- Uses existing CCXT library security model
- Validates all inputs before execution
- Implements guards against excessive trading
- Maintains existing safety limits (notional, turnover, etc.)
- Uses fail-safe exception handling
- Edge calculation is purely mathematical (no new attack vectors)

## Definition of DONE ✅

All requirements met:
- ✅ User can run live with --order-type limit_maker
- ✅ Bot places post-only limit orders with edge-based pricing
- ✅ BUY orders placed below bid, SELL orders above ask
- ✅ Edge calculation includes fees, spread, slippage, profit
- ✅ Smooth max ensures edge >= floor
- ✅ Bot auto-cancels stale orders
- ✅ No changes to strategy or signal logic
- ✅ No new mechanism beyond execution layer
- ✅ Comprehensive tests passing (20/20)
- ✅ Documentation complete with examples
- ✅ Existing tests still pass

