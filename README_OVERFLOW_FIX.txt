Patch: Fix overflow in logistic exposure mapping

What:
- Replaces exp(k*z) with a numerically-stable variant by clipping the argument to [-60, 60].

Why:
- Prevents RuntimeWarning: overflow encountered in exp
- Makes kalman strategy and fast_backtest deterministic/stable for large |z|.

How to apply:
- Unzip into repo root (it will overwrite the two files):
    unzip -o theta-overflow-logistic-patch.zip

Files:
- spot_bot/strategies/kalman.py
- spot_bot/backtest/fast_backtest.py
