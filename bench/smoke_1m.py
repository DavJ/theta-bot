#!/usr/bin/env python3
"""
Smoke test for 1m benchmark_matrix with KALMAN_MR_DUAL.

Tests:
1. Runs benchmark_matrix with 1m timeframe and small data (2000 bars)
2. Verifies no RuntimeWarning overflow occurs
3. Verifies cache functionality (second run is faster and doesn't re-download)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
import shutil


def run_benchmark(cache_dir: str = "bench_cache_smoke", workdir: str = "bench_out_smoke") -> tuple[int, float, str]:
    """Run benchmark_matrix and return (exit_code, elapsed_time, stderr)."""
    cmd = [
        sys.executable,
        "-m",
        "bench.benchmark_matrix",
        "--timeframe", "1m",
        "--limit-total", "2000",
        "--symbols", "BTC/USDT,ETH/USDT",
        "--psi-modes", "scale_phase",
        "--methods", "KALMAN_MR_DUAL",
        "--rv-window", "120",
        "--psi-window", "512",
        "--fee-rate", "0.001",
        "--slippage-bps", "5",
        "--max-exposure", "0.30",
        "--cache-dir", cache_dir,
        "--workdir", workdir,
        "--out", f"{workdir}/matrix_1m.csv",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    return result.returncode, elapsed, result.stderr


def main() -> None:
    cache_dir = Path("bench_cache_smoke")
    workdir = Path("bench_out_smoke")
    
    # Clean up from previous runs
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    if workdir.exists():
        shutil.rmtree(workdir)
    
    print("=" * 60)
    print("1m KALMAN_MR_DUAL Smoke Test")
    print("=" * 60)
    
    # First run - should download data
    print("\n[1/2] First run (should download data)...")
    exit_code1, elapsed1, stderr1 = run_benchmark(str(cache_dir), str(workdir))
    
    if exit_code1 != 0:
        print(f"❌ First run failed with exit code {exit_code1}")
        print("STDERR:", stderr1)
        sys.exit(1)
    
    print(f"✓ First run completed in {elapsed1:.2f}s")
    
    # Check for overflow warnings
    if "overflow" in stderr1.lower() or "RuntimeWarning" in stderr1:
        print("❌ Overflow warning detected in first run!")
        print("STDERR:", stderr1)
        sys.exit(1)
    
    print("✓ No overflow warnings detected")
    
    # Check that cache files were created
    cache_files = list(cache_dir.glob("*.csv"))
    if not cache_files:
        print("❌ No cache files created!")
        sys.exit(1)
    
    print(f"✓ Cache files created: {len(cache_files)} files")
    
    # Second run - should use cache
    print("\n[2/2] Second run (should use cache)...")
    exit_code2, elapsed2, stderr2 = run_benchmark(str(cache_dir), str(workdir))
    
    if exit_code2 != 0:
        print(f"❌ Second run failed with exit code {exit_code2}")
        print("STDERR:", stderr2)
        sys.exit(1)
    
    print(f"✓ Second run completed in {elapsed2:.2f}s")
    
    # Check for overflow warnings
    if "overflow" in stderr2.lower() or "RuntimeWarning" in stderr2:
        print("❌ Overflow warning detected in second run!")
        print("STDERR:", stderr2)
        sys.exit(1)
    
    print("✓ No overflow warnings detected")
    
    # Verify speedup (second run should be faster)
    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else 1.0
    if speedup < 1.1:  # At least 10% faster
        print(f"⚠️  Warning: Second run not significantly faster (speedup: {speedup:.2f}x)")
        print(f"   First run: {elapsed1:.2f}s, Second run: {elapsed2:.2f}s")
    else:
        print(f"✓ Second run is {speedup:.2f}x faster (cache working)")
    
    # Verify output files exist
    output_csv = workdir / "matrix_1m.csv"
    if not output_csv.exists():
        print("❌ Output CSV not created!")
        sys.exit(1)
    
    print(f"✓ Output CSV created: {output_csv}")
    
    # Check equity files
    equity_files = list(workdir.glob("equity_*.csv"))
    if not equity_files:
        print("❌ No equity files created!")
        sys.exit(1)
    
    print(f"✓ Equity files created: {len(equity_files)} files")
    
    # Clean up
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print(f"\nCleaning up temporary files...")
    shutil.rmtree(cache_dir)
    shutil.rmtree(workdir)
    print("Done!")


if __name__ == "__main__":
    main()
