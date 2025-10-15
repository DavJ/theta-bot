# Theta FFT Variants Patch

This patch provides:
- Non-destructive patcher for `tests_backtest/cli/run_theta_benchmark.py`
- Ready `benchmark.sh` to run RAW / 1D / 2D / 3D / Wavelet and aggregate

## Use
unzip patch_theta_fft_variants.zip -d .
python patch_theta_fft_variants/scripts/apply_theta_patch.py
python -m py_compile tests_backtest/cli/run_theta_benchmark.py

chmod +x patch_theta_fft_variants/benchmark.sh
./patch_theta_fft_variants/benchmark.sh
sed -n '1,80p' reports_cmp/overall.md
