import csv, json, math, os, time
from datetime import datetime

class PaperBroker:
    def __init__(self, fee_bps=4, slip_bps=1, base_equity=1000, max_w_per_symbol=0.5, hedge_ratio=0.5, cooldown_steps=2,
                 trades_path="paper_trades.csv", equity_path="equity_curve.csv"):
        self.fee = fee_bps / 1e4
        self.slip = slip_bps / 1e4
        self.equity = float(base_equity)
        self.max_w = float(max_w_per_symbol)
        self.hedge_ratio = float(hedge_ratio)
        self.cooldown = int(cooldown_steps)
        self.trades_path = trades_path
        self.equity_path = equity_path
        self.positions = {}  # symbol -> qty
        self.last_signal_step = {}  # symbol -> step idx
        self.step_idx = 0
        # init logs
        if not os.path.exists(self.trades_path):
            with open(self.trades_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp","symbol","side","qty","price","fee","reason"])
        if not os.path.exists(self.equity_path):
            with open(self.equity_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp","equity","positions_json"])

    def _log_trade(self, sym, side, qty, price, fee, reason):
        with open(self.trades_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([int(time.time()*1000), sym, side, f"{qty:.8f}", f"{price:.2f}", f"{fee:.6f}", reason])

    def _log_equity(self):
        with open(self.equity_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([int(time.time()*1000), f"{self.equity:.2f}", json.dumps(self.positions)])

    def step(self, signals: dict, prices: dict):
        # signals: {symbol: -1/0/1}, prices: {symbol: last_price}
        self.step_idx += 1
        # Compute target weights
        targets = {}
        active_syms = [s for s in signals.keys() if s in prices]
        if not active_syms:
            self._log_equity()
            return
        for i, sym in enumerate(active_syms):
            sig = int(signals[sym])
            # cooldown
            if self.cooldown > 0 and sym in self.last_signal_step and self.step_idx - self.last_signal_step[sym] < self.cooldown:
                sig = 0
            # target weight
            targets[sym] = max(-self.max_w, min(self.max_w, sig * self.max_w))
        # Rebalance positions to targets
        for sym in active_syms:
            price = float(prices[sym])
            target_val = targets.get(sym, 0.0) * self.equity
            current_qty = self.positions.get(sym, 0.0)
            current_val = current_qty * price
            delta_val = target_val - current_val
            if abs(delta_val) < 1e-6:
                continue
            # Simulate slippage/fee
            side = "BUY" if delta_val > 0 else "SELL"
            fill_price = price * (1 + self.slip) if side == "BUY" else price * (1 - self.slip)
            qty = delta_val / fill_price if fill_price > 0 else 0.0
            fee = abs(delta_val) * self.fee
            # Update positions/equity
            self.positions[sym] = current_qty + qty
            self.equity -= fee
            self._log_trade(sym, side, qty, fill_price, fee, reason="rebalance")
            if signals.get(sym, 0) != 0:
                self.last_signal_step[sym] = self.step_idx
        # Mark-to-market equity
        mtm = 0.0
        for sym, qty in self.positions.items():
            if sym in prices:
                mtm += qty * float(prices[sym])
        # cash = equity is implicitly cash + positions value; we track equity as cash + mtm
        # Here we simply recompute equity as previous equity of cash plus mtm delta is already reflected.
        # For simplicity, we leave equity updated only by fees; PnL realized in mtm below:
        # We output equity as cash+mtm
        eq_out = self.equity
        eq_out += mtm
        # To avoid double counting across steps, we keep only logging eq_out; equity internal stays as cash
        with open(self.equity_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([int(time.time()*1000), f"{eq_out:.2f}", json.dumps(self.positions)])
