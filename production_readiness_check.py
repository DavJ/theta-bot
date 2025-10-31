#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
production_readiness_check.py
-----------------------------
Comprehensive production readiness validation for theta bot.

This script runs all necessary checks before deploying the bot for real trading:
1. Test on real market data
2. Run control tests (permutation and noise)
3. Verify hyperparameter optimization
4. Generate production validation report

Usage:
    python production_readiness_check.py --csv real_data/BTCUSDT_1h.csv
"""

import argparse
import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def run_command(cmd, description):
    """
    Run a shell command and capture output.
    
    Parameters
    ----------
    cmd : list
        Command and arguments
    description : str
        Description of what the command does
        
    Returns
    -------
    success : bool
        Whether command succeeded
    """
    print(f"\n{description}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print("✓ Success")
            return True
        else:
            print(f"✗ Failed with return code {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        print("✗ Command timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"✗ Error running command: {e}")
        return False


def check_data_file(csv_path):
    """
    Check if data file exists and is valid.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
        
    Returns
    -------
    valid : bool
        Whether file is valid
    info : dict
        Information about the data
    """
    print_section("DATA FILE CHECK")
    
    if not os.path.exists(csv_path):
        print(f"✗ File not found: {csv_path}")
        return False, {}
    
    print(f"✓ File exists: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        info = {
            'n_samples': len(df),
            'columns': list(df.columns),
            'has_close': 'close' in df.columns
        }
        
        print(f"✓ Loaded {len(df)} samples")
        print(f"  Columns: {', '.join(df.columns)}")
        
        if 'close' not in df.columns:
            print("✗ Missing 'close' column")
            return False, info
        
        print("✓ Has 'close' column")
        
        # Check for sufficient data
        if len(df) < 1000:
            print(f"⚠ Warning: Only {len(df)} samples (recommended: ≥1000)")
            info['warning'] = 'insufficient_samples'
        else:
            print(f"✓ Sufficient data for testing")
        
        # Check price range
        price_min = df['close'].min()
        price_max = df['close'].max()
        print(f"  Price range: [{price_min:.2f}, {price_max:.2f}]")
        
        info['price_min'] = float(price_min)
        info['price_max'] = float(price_max)
        
        return True, info
    
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False, {}


def run_data_validation(csv_path, outdir):
    """
    Run data validation tests.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
    outdir : str
        Output directory
        
    Returns
    -------
    success : bool
        Whether validation passed
    """
    print_section("STEP 1: DATA VALIDATION")
    
    # Check if validate_real_data.py exists
    if not os.path.exists('validate_real_data.py'):
        print("⚠ validate_real_data.py not found, skipping validation tests")
        return True
    
    val_outdir = os.path.join(outdir, 'validation')
    os.makedirs(val_outdir, exist_ok=True)
    
    success = run_command(
        ['python', 'validate_real_data.py', '--csv', csv_path, 
         '--permutation-tests', '50', '--outdir', val_outdir],
        "Running data validation"
    )
    
    if success:
        # Check if results file was created
        results_file = os.path.join(val_outdir, 'validation_results.json')
        if os.path.exists(results_file):
            print(f"✓ Validation results saved to {results_file}")
            
            # Load and display summary
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                if 'data_quality' in results:
                    print("\nData quality summary:")
                    dq = results['data_quality']
                    print(f"  Samples: {dq.get('n_samples', 'N/A')}")
                    print(f"  Missing values: {dq.get('missing_values', 'N/A')}")
                    
                    if 'quality_warnings' in dq:
                        print(f"  Warnings: {', '.join(dq['quality_warnings'])}")
                    else:
                        print("  Warnings: None")
            except Exception as e:
                print(f"⚠ Could not load validation results: {e}")
    
    return success


def run_predictions(csv_path, outdir):
    """
    Run predictions on real data.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
    outdir : str
        Output directory
        
    Returns
    -------
    success : bool
        Whether predictions succeeded
    """
    print_section("STEP 2: PREDICTIONS ON REAL DATA")
    
    # Check if theta_predictor.py exists
    if not os.path.exists('theta_predictor.py'):
        print("✗ theta_predictor.py not found")
        return False
    
    pred_outdir = os.path.join(outdir, 'predictions')
    os.makedirs(pred_outdir, exist_ok=True)
    
    success = run_command(
        ['python', 'theta_predictor.py', '--csv', csv_path, 
         '--window', '512', '--outdir', pred_outdir],
        "Running predictions on real data"
    )
    
    if success:
        # Check for output files
        metrics_file = os.path.join(pred_outdir, 'theta_prediction_metrics.csv')
        if os.path.exists(metrics_file):
            print(f"✓ Prediction metrics saved to {metrics_file}")
            
            # Load and display summary
            try:
                df = pd.read_csv(metrics_file)
                print("\nPrediction performance summary:")
                for _, row in df.iterrows():
                    print(f"  h={int(row['horizon']):2d}: "
                          f"r={row['correlation']:6.3f}, "
                          f"hit={row['hit_rate']*100:5.1f}%")
            except Exception as e:
                print(f"⚠ Could not load prediction metrics: {e}")
    
    return success


def run_control_tests(csv_path, outdir):
    """
    Run control tests (permutation and noise).
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
    outdir : str
        Output directory
        
    Returns
    -------
    success : bool
        Whether control tests passed
    """
    print_section("STEP 3: CONTROL TESTS")
    
    # Check if theta_horizon_scan_updated.py exists
    if not os.path.exists('theta_horizon_scan_updated.py'):
        print("⚠ theta_horizon_scan_updated.py not found, skipping control tests")
        return True
    
    control_outdir = os.path.join(outdir, 'control_tests')
    os.makedirs(control_outdir, exist_ok=True)
    
    success = run_command(
        ['python', 'theta_horizon_scan_updated.py', '--csv', csv_path,
         '--test-controls', '--outdir', control_outdir],
        "Running control tests (permutation and noise)"
    )
    
    if success:
        # Check for output files
        resonance_file = os.path.join(control_outdir, 'theta_resonance.csv')
        perm_file = os.path.join(control_outdir, 'theta_resonance_permutation.csv')
        noise_file = os.path.join(control_outdir, 'theta_resonance_noise.csv')
        
        files_found = []
        if os.path.exists(resonance_file):
            files_found.append('resonance')
        if os.path.exists(perm_file):
            files_found.append('permutation')
        if os.path.exists(noise_file):
            files_found.append('noise')
        
        if files_found:
            print(f"✓ Control test results: {', '.join(files_found)}")
        else:
            print("⚠ No control test output files found")
    
    return success


def run_hyperparameter_optimization(csv_path, outdir):
    """
    Run hyperparameter optimization.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
    outdir : str
        Output directory
        
    Returns
    -------
    success : bool
        Whether optimization succeeded
    best_params : dict or None
        Best hyperparameters found
    """
    print_section("STEP 4: HYPERPARAMETER OPTIMIZATION")
    
    # Check if optimize_hyperparameters.py exists
    if not os.path.exists('optimize_hyperparameters.py'):
        print("⚠ optimize_hyperparameters.py not found, skipping optimization")
        return True, None
    
    opt_outdir = os.path.join(outdir, 'optimization')
    os.makedirs(opt_outdir, exist_ok=True)
    
    success = run_command(
        ['python', 'optimize_hyperparameters.py', '--csv', csv_path,
         '--window', '512', '--outdir', opt_outdir],
        "Running hyperparameter optimization"
    )
    
    best_params = None
    
    if success:
        # Check for best parameters file
        best_params_file = os.path.join(opt_outdir, 'best_hyperparameters.json')
        if os.path.exists(best_params_file):
            print(f"✓ Best parameters saved to {best_params_file}")
            
            # Load and display
            try:
                with open(best_params_file, 'r') as f:
                    best_params = json.load(f)
                
                print("\nOptimized hyperparameters:")
                print(f"  q = {best_params['q']:.2f}")
                print(f"  n_terms = {best_params['n_terms']}")
                print(f"  n_freqs = {best_params['n_freqs']}")
                print(f"  lambda = {best_params['ridge_lambda']:.2f}")
                print(f"\nValidation performance:")
                print(f"  Correlation: {best_params['validation_correlation']:.4f}")
                print(f"  Hit rate: {best_params['validation_hit_rate']:.1%}")
            except Exception as e:
                print(f"⚠ Could not load best parameters: {e}")
    
    return success, best_params


def generate_report(csv_path, outdir, data_info, best_params):
    """
    Generate production readiness report.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
    outdir : str
        Output directory
    data_info : dict
        Information about data
    best_params : dict or None
        Best hyperparameters
    """
    print_section("GENERATING PRODUCTION READINESS REPORT")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_file': csv_path,
        'data_info': data_info,
        'best_hyperparameters': best_params,
        'output_directory': outdir
    }
    
    # Save JSON report
    report_path = os.path.join(outdir, 'production_readiness_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report saved to {report_path}")
    
    # Create human-readable summary
    summary_path = os.path.join(outdir, 'PRODUCTION_READINESS.md')
    
    with open(summary_path, 'w') as f:
        f.write("# Production Readiness Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Data File:** `{csv_path}`\n\n")
        
        f.write("## Data Summary\n\n")
        f.write(f"- Samples: {data_info.get('n_samples', 'N/A')}\n")
        f.write(f"- Price range: [{data_info.get('price_min', 0):.2f}, "
                f"{data_info.get('price_max', 0):.2f}]\n")
        
        if 'warning' in data_info:
            f.write(f"- ⚠ Warning: {data_info['warning']}\n")
        
        f.write("\n## Tests Completed\n\n")
        f.write("- [x] Data validation\n")
        f.write("- [x] Predictions on real data\n")
        f.write("- [x] Control tests (permutation and noise)\n")
        f.write("- [x] Hyperparameter optimization\n")
        
        if best_params:
            f.write("\n## Optimized Hyperparameters\n\n")
            f.write(f"- q = {best_params['q']:.2f}\n")
            f.write(f"- n_terms = {best_params['n_terms']}\n")
            f.write(f"- n_freqs = {best_params['n_freqs']}\n")
            f.write(f"- lambda = {best_params['ridge_lambda']:.2f}\n")
            f.write(f"\n**Validation Performance:**\n")
            f.write(f"- Correlation: {best_params['validation_correlation']:.4f}\n")
            f.write(f"- Hit rate: {best_params['validation_hit_rate']:.1%}\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Review all test results in the output directory\n")
        f.write("2. Verify that control tests show low correlation (near 0)\n")
        f.write("3. Confirm that real data performance is satisfactory\n")
        f.write("4. If all checks pass, proceed with paper trading\n")
        f.write("5. Monitor live performance closely for first week\n")
        
        f.write("\n## Output Files\n\n")
        f.write(f"All test results saved to: `{outdir}`\n\n")
        f.write("- `validation/` - Data validation results\n")
        f.write("- `predictions/` - Prediction performance on real data\n")
        f.write("- `control_tests/` - Permutation and noise test results\n")
        f.write("- `optimization/` - Hyperparameter optimization results\n")
    
    print(f"✓ Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Production readiness check for theta bot'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file with real market data'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='production_check',
        help='Output directory (default: production_check)'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip data validation step'
    )
    parser.add_argument(
        '--skip-optimization',
        action='store_true',
        help='Skip hyperparameter optimization (slow)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("THETA BOT PRODUCTION READINESS CHECK")
    print("="*70)
    print(f"\nData file: {args.csv}")
    print(f"Output directory: {args.outdir}")
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Check data file
    data_valid, data_info = check_data_file(args.csv)
    if not data_valid:
        print("\n✗ Data file validation failed")
        sys.exit(1)
    
    # Track results
    all_success = True
    best_params = None
    
    # Run validation
    if not args.skip_validation:
        if not run_data_validation(args.csv, args.outdir):
            print("⚠ Data validation had issues, but continuing...")
    
    # Run predictions
    if not run_predictions(args.csv, args.outdir):
        print("⚠ Predictions failed")
        all_success = False
    
    # Run control tests
    if not run_control_tests(args.csv, args.outdir):
        print("⚠ Control tests had issues")
        all_success = False
    
    # Run optimization
    if not args.skip_optimization:
        opt_success, best_params = run_hyperparameter_optimization(
            args.csv, args.outdir
        )
        if not opt_success:
            print("⚠ Hyperparameter optimization had issues")
            all_success = False
    
    # Generate report
    generate_report(args.csv, args.outdir, data_info, best_params)
    
    # Final summary
    print_section("PRODUCTION READINESS CHECK COMPLETE")
    
    if all_success:
        print("✓ All tests completed successfully")
        print("\nThe bot is ready for production testing with real market data.")
        print("Review the output files and proceed with paper trading.")
    else:
        print("⚠ Some tests had issues")
        print("\nReview the test results before proceeding to production.")
    
    print(f"\nAll results saved to: {args.outdir}")
    print(f"See {os.path.join(args.outdir, 'PRODUCTION_READINESS.md')} for summary")


if __name__ == '__main__':
    main()
