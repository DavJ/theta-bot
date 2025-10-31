#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_start.py
-------------
Quick start script for testing theta bot on real data.

This script provides an easy way to test the theta bot with real market data
in one command.

Usage:
    # Download and test BTCUSDT data
    python quick_start.py --symbol BTCUSDT --interval 1h --limit 2000
    
    # Use existing CSV file
    python quick_start.py --csv path/to/data.csv
"""

import argparse
import os
import sys
import subprocess
import json
from datetime import datetime


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")


def run_step(description, command, check_output=None):
    """
    Run a step in the pipeline.
    
    Parameters
    ----------
    description : str
        What the step does
    command : list
        Command to run
    check_output : str or None
        File to check for output
        
    Returns
    -------
    success : bool
        Whether the step succeeded
    """
    print(f"► {description}")
    print(f"  Command: {' '.join(command)}\n")
    
    try:
        result = subprocess.run(
            command,
            capture_output=False,
            timeout=600
        )
        
        if result.returncode == 0:
            print(f"\n✓ {description} completed\n")
            
            if check_output and os.path.exists(check_output):
                print(f"  Output saved to: {check_output}")
            
            return True
        else:
            print(f"\n✗ {description} failed with return code {result.returncode}\n")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"\n✗ {description} timed out\n")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Quick start: Test theta bot on real market data'
    )
    
    # Data source options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--csv',
        type=str,
        help='Use existing CSV file'
    )
    data_group.add_argument(
        '--symbol',
        type=str,
        help='Download data for symbol (e.g., BTCUSDT)'
    )
    
    # Download options
    parser.add_argument(
        '--interval',
        type=str,
        default='1h',
        help='Timeframe interval (default: 1h)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=2000,
        help='Number of candles to download (default: 2000)'
    )
    
    # Test options
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip data validation step'
    )
    parser.add_argument(
        '--skip-optimization',
        action='store_true',
        help='Skip hyperparameter optimization (saves time)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: skip validation and optimization'
    )
    
    # Output options
    parser.add_argument(
        '--outdir',
        type=str,
        default='quick_test',
        help='Output directory (default: quick_test)'
    )
    
    args = parser.parse_args()
    
    print_header("THETA BOT QUICK START")
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Step 1: Get data
    if args.csv:
        print_header("STEP 1: Loading existing data")
        csv_path = args.csv
        
        if not os.path.exists(csv_path):
            print(f"✗ File not found: {csv_path}")
            sys.exit(1)
        
        print(f"✓ Using existing file: {csv_path}")
    
    else:
        print_header("STEP 1: Downloading market data")
        
        csv_path = os.path.join(args.outdir, f"{args.symbol}_{args.interval}.csv")
        
        success = run_step(
            f"Downloading {args.symbol} {args.interval} data",
            [
                'python', 'download_market_data.py',
                '--symbol', args.symbol,
                '--interval', args.interval,
                '--limit', str(args.limit),
                '--outdir', args.outdir,
                '--output', f"{args.symbol}_{args.interval}.csv"
            ],
            check_output=csv_path
        )
        
        if not success:
            print("✗ Failed to download data")
            sys.exit(1)
    
    # Step 2: Validate (optional)
    if not args.skip_validation and not args.quick:
        print_header("STEP 2: Validating data")
        
        val_outdir = os.path.join(args.outdir, 'validation')
        
        success = run_step(
            "Running data validation",
            [
                'python', 'validate_real_data.py',
                '--csv', csv_path,
                '--permutation-tests', '20',
                '--outdir', val_outdir
            ],
            check_output=os.path.join(val_outdir, 'validation_results.json')
        )
        
        if not success:
            print("⚠ Data validation had issues, but continuing...")
    else:
        print_header("STEP 2: Validation (skipped)")
    
    # Step 3: Predictions
    print_header("STEP 3: Running predictions on real data")
    
    pred_outdir = os.path.join(args.outdir, 'predictions')
    
    success = run_step(
        "Running walk-forward predictions",
        [
            'python', 'theta_predictor.py',
            '--csv', csv_path,
            '--window', '512',
            '--outdir', pred_outdir
        ],
        check_output=os.path.join(pred_outdir, 'theta_prediction_metrics.csv')
    )
    
    if not success:
        print("✗ Predictions failed")
        sys.exit(1)
    
    # Show prediction results
    try:
        import pandas as pd
        metrics_file = os.path.join(pred_outdir, 'theta_prediction_metrics.csv')
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            print("\n📊 PREDICTION RESULTS:")
            print("-" * 70)
            for _, row in df.iterrows():
                print(f"  h={int(row['horizon']):2d}: "
                      f"correlation={row['correlation']:7.4f}, "
                      f"hit_rate={row['hit_rate']*100:5.1f}%")
            print("-" * 70)
    except:
        pass
    
    # Step 4: Control tests
    print_header("STEP 4: Running control tests")
    
    control_outdir = os.path.join(args.outdir, 'control_tests')
    
    success = run_step(
        "Running permutation and noise tests",
        [
            'python', 'theta_horizon_scan_updated.py',
            '--csv', csv_path,
            '--test-controls',
            '--outdir', control_outdir
        ],
        check_output=os.path.join(control_outdir, 'theta_resonance.csv')
    )
    
    if not success:
        print("⚠ Control tests had issues, but continuing...")
    
    # Step 5: Optimization (optional)
    if not args.skip_optimization and not args.quick:
        print_header("STEP 5: Optimizing hyperparameters")
        
        opt_outdir = os.path.join(args.outdir, 'optimization')
        
        success = run_step(
            "Running hyperparameter optimization (this may take a while)",
            [
                'python', 'optimize_hyperparameters.py',
                '--csv', csv_path,
                '--window', '512',
                '--outdir', opt_outdir
            ],
            check_output=os.path.join(opt_outdir, 'best_hyperparameters.json')
        )
        
        if success:
            # Show best parameters
            try:
                best_params_file = os.path.join(opt_outdir, 'best_hyperparameters.json')
                if os.path.exists(best_params_file):
                    with open(best_params_file, 'r') as f:
                        best_params = json.load(f)
                    
                    print("\n🎯 OPTIMIZED PARAMETERS:")
                    print("-" * 70)
                    print(f"  q = {best_params['q']:.2f}")
                    print(f"  n_terms = {best_params['n_terms']}")
                    print(f"  n_freqs = {best_params['n_freqs']}")
                    print(f"  lambda = {best_params['ridge_lambda']:.2f}")
                    print(f"\n  Validation correlation: {best_params['validation_correlation']:.4f}")
                    print(f"  Validation hit rate: {best_params['validation_hit_rate']:.1%}")
                    print("-" * 70)
            except:
                pass
    else:
        print_header("STEP 5: Optimization (skipped)")
    
    # Summary
    print_header("QUICK START COMPLETE")
    
    print(f"✓ All tests completed successfully!\n")
    print(f"📁 Results saved to: {args.outdir}/\n")
    print("Next steps:")
    print("1. Review prediction results in predictions/")
    print("2. Check control test results in control_tests/")
    if not args.skip_optimization and not args.quick:
        print("3. Review optimized parameters in optimization/")
        print("4. If results are good, proceed with paper trading")
    else:
        print("3. Run full optimization: python optimize_hyperparameters.py --csv " + csv_path)
        print("4. If results are good, proceed with paper trading")
    
    print("\n" + "="*70)
    print("See PRODUCTION_PREPARATION.md for detailed guidance")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
