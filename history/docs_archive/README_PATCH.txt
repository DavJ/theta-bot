Patch: Theta FFT hybrid/dynamic dtype fix

Files included:
- tests_backtest/common/transforms_theta_fast.py

What it fixes:
- Ensures inputs are float64 (avoids dtype=object crashing np.fft).
- Uses rfft for purely real windows; full fft when window introduces complex phase.
- Adds robust 1D/2D/3D variants consistent with your CLI flags.

How to apply:
unzip ~/Downloads/theta_fft_fix_patch.zip -d <repo_root>
python -m py_compile tests_backtest/common/transforms_theta_fast.py