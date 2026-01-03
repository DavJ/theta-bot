#!/usr/bin/env python3
"""
Generate drift series from derivatives state.

Computes mu(t) directional pressure and D(t) determinism magnitude
from derivatives market data (funding, open interest, basis).

Example:
    python scripts/generate_drift_series.py \
        --symbols BTCUSDT ETHUSDT \
        --data-dir data/raw \
        --output-dir data/processed/drift \
        --window 7D \
        --quantile 0.85 \
        --alpha 1.0 --beta 1.0 --gamma 0.0 \
        --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from theta_bot_averaging.derivatives_state import (
    load_spot_series,
    load_funding_series,
    load_oi_series,
    load_basis_series,
    compute_zscore,
    compute_oi_change,
    compute_drift,
    compute_determinism,
    apply_combined_gate,
    generate_top_timestamps_report,
)
from theta_bot_averaging.derivatives_state.features import align_series
from theta_bot_averaging.derivatives_state.report import save_report

# Constant for millisecond to microsecond conversion
MS_TO_NS = 1_000_000


def generate_drift_for_symbol(
    symbol: str,
    data_dir: str,
    output_dir: str,
    window: str = "7D",
    quantile: float = 0.85,
    threshold: float = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
) -> tuple[pd.DataFrame, str]:
    """
    Generate drift series for a single symbol.
    
    Parameters
    ----------
    symbol : str
        Trading pair symbol
    data_dir : str
        Base data directory
    output_dir : str
        Output directory for drift files
    window : str
        Rolling window for z-score (default: "7D")
    quantile : float
        Quantile threshold for gating (default: 0.85)
    threshold : float, optional
        Fixed threshold for gating
    alpha : float
        Weight for overcrowding unwind term
    beta : float
        Weight for basis-pressure term
    gamma : float
        Weight for expiry/roll pressure term
        
    Returns
    -------
    tuple[pd.DataFrame, str]
        (drift_df, output_path)
    """
    print(f"\nProcessing {symbol}...")
    
    # Load data
    print(f"  Loading spot data...")
    spot_df = load_spot_series(symbol, data_dir)
    
    print(f"  Loading funding data...")
    funding_df = load_funding_series(symbol, data_dir)
    
    print(f"  Loading open interest data...")
    oi_df = load_oi_series(symbol, data_dir)
    
    print(f"  Loading basis data...")
    basis_df = load_basis_series(symbol, data_dir)
    
    if basis_df is None:
        raise ValueError(f"No basis data available for {symbol}")
    
    # Extract series
    funding_series = funding_df["fundingRate"]
    oi_series = oi_df["sumOpenInterest"]
    basis_series = basis_df["basis"]
    
    # Compute derived features
    print(f"  Computing OI change...")
    oi_change = compute_oi_change(oi_series)
    
    # Compute z-scores
    print(f"  Computing z-scores (window={window})...")
    z_funding = compute_zscore(funding_series, window=window)
    z_oi_change = compute_zscore(oi_change, window=window)
    z_basis = compute_zscore(basis_series, window=window)
    
    # Align all series to common index
    print(f"  Aligning series...")
    z_funding, z_oi_change, z_basis = align_series(
        z_funding, z_oi_change, z_basis, method="inner"
    )
    
    # Compute drift
    print(f"  Computing drift (alpha={alpha}, beta={beta}, gamma={gamma})...")
    mu, mu1, mu2, mu3 = compute_drift(
        z_oi_change=z_oi_change,
        z_funding=z_funding,
        z_basis=z_basis,
        rho=None,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    
    # Compute determinism
    print(f"  Computing determinism...")
    D = compute_determinism(mu)
    
    # Apply gating
    print(f"  Applying gating (quantile={quantile})...")
    active = apply_combined_gate(D, quantile=quantile, threshold=threshold)
    
    # Build output DataFrame
    drift_df = pd.DataFrame({
        "mu": mu,
        "D": D,
        "active": active.astype(int),
        "mu1": mu1,
        "mu2": mu2,
        "mu3": mu3,
        "z_funding": z_funding,
        "z_oi_change": z_oi_change,
        "z_basis": z_basis,
    })
    
    # Drop NaN rows
    drift_df = drift_df.dropna()
    
    # Save to CSV
    output_path = Path(output_dir) / f"{symbol}_1h.csv.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Reset index to include timestamp column
    drift_df_out = drift_df.copy()
    drift_df_out["timestamp"] = (drift_df_out.index.astype("int64") // MS_TO_NS).astype(int)
    drift_df_out = drift_df_out[["timestamp"] + [c for c in drift_df_out.columns if c != "timestamp"]]
    
    print(f"  Saving to {output_path}...")
    drift_df_out.to_csv(output_path, compression="gzip", index=False)
    
    print(f"  ✓ Saved {len(drift_df)} records to {output_path}")
    
    return drift_df, str(output_path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate drift series from derivatives state"
    )
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="Symbols to process (default: BTCUSDT ETHUSDT)",
    )
    ap.add_argument(
        "--data-dir",
        default="data/raw",
        help="Base data directory (default: data/raw)",
    )
    ap.add_argument(
        "--output-dir",
        default="data/processed/drift",
        help="Output directory (default: data/processed/drift)",
    )
    ap.add_argument(
        "--window",
        default="7D",
        help="Rolling window for z-score (default: 7D)",
    )
    ap.add_argument(
        "--quantile",
        type=float,
        default=0.85,
        help="Quantile threshold for gating (default: 0.85)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold for gating (optional)",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for overcrowding unwind term (default: 1.0)",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight for basis-pressure term (default: 1.0)",
    )
    ap.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="Weight for expiry/roll pressure term (default: 0.0)",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="Generate markdown report of top-20 timestamps",
    )
    args = ap.parse_args()
    
    print("=" * 80)
    print("DRIFT SERIES GENERATION")
    print("=" * 80)
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Z-score window: {args.window}")
    print(f"Quantile: {args.quantile}")
    print(f"Threshold: {args.threshold}")
    print(f"Weights: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    print("=" * 80)
    
    all_success = True
    
    for symbol in args.symbols:
        try:
            drift_df, output_path = generate_drift_for_symbol(
                symbol=symbol,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                window=args.window,
                quantile=args.quantile,
                threshold=args.threshold,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            )
            
            # Generate report if requested
            if args.report:
                report = generate_top_timestamps_report(
                    drift_df, top_n=20, symbol=symbol
                )
                report_path = Path(args.output_dir) / f"{symbol}_report.md"
                save_report(report, str(report_path))
                print(f"  ✓ Saved report to {report_path}")
                
        except Exception as e:
            print(f"  ✗ ERROR processing {symbol}: {e}")
            all_success = False
    
    print("\n" + "=" * 80)
    if all_success:
        print("✓ ALL SYMBOLS PROCESSED SUCCESSFULLY")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME SYMBOLS FAILED - Review errors above")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
