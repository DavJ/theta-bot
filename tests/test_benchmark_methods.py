"""
Test benchmark_methods.py script functionality.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def generate_synthetic_ohlcv(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    
    # Generate timestamps
    start = pd.Timestamp("2023-01-01", tz="UTC")
    timestamps = pd.date_range(start, periods=n_bars, freq="1h")
    
    # Generate price data with some trend and noise
    base_price = 30000.0
    trend = np.linspace(0, 0.1, n_bars)
    noise = np.cumsum(np.random.randn(n_bars) * 0.02)
    close = base_price * (1 + trend + noise)
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.01))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    
    # Generate volume
    volume = np.random.uniform(100, 1000, n_bars)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    
    return df


def test_benchmark_baseline_mode():
    """Test baseline mode with synthetic data."""
    from spot_bot.benchmark_methods import (
        generate_baseline_configs,
        load_pair_csv,
        run_single_backtest,
        compute_composite_score,
        create_leaderboards,
    )
    
    # Generate synthetic data
    df = generate_synthetic_ohlcv(n_bars=500)
    
    # Create temporary directory and save CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        csv_path = tmpdir / "BTCUSDT_1h.csv"
        df.to_csv(csv_path, index=False)
        
        # Load the CSV
        loaded_df = load_pair_csv(csv_path)
        assert len(loaded_df) == 500
        assert "close" in loaded_df.columns
        
        # Generate baseline configs
        configs = generate_baseline_configs()
        assert len(configs) == 4
        assert all("psi_mode" in c for c in configs)
        
        # Run a single backtest
        config = configs[0]
        result = run_single_backtest(
            pair="BTCUSDT",
            ohlcv_df=loaded_df,
            config_dict=config,
            fee_rate=0.0005,
            slippage_bps=0.0,
            initial_equity=1000.0,
            periods_per_year=24 * 365,
        )
        
        # Verify result structure
        assert result is not None
        assert "pair" in result
        assert "method_name" in result
        assert "sharpe" in result
        assert "max_drawdown" in result
        assert result["pair"] == "BTCUSDT"


def test_benchmark_grid_mode():
    """Test grid mode configuration generation."""
    from spot_bot.benchmark_methods import generate_grid_configs
    
    configs = generate_grid_configs()
    
    # Grid should have more configs than baseline
    # FFT modes: 2 modes × 2 psi_window = 4
    # Mellin modes: 2 modes × (2×2×2×2×1×1) = 2 × 16 = 32
    # Total: 4 + 32 = 36
    assert len(configs) == 36
    
    # Check that all configs have psi_mode
    assert all("psi_mode" in c for c in configs)
    
    # Check FFT configs only have psi_window varied
    fft_configs = [c for c in configs if c["psi_mode"] in ["cepstrum", "complex_cepstrum"]]
    assert len(fft_configs) == 4
    
    # Check Mellin configs have multiple parameters
    mellin_configs = [c for c in configs if "mellin" in c["psi_mode"]]
    assert len(mellin_configs) == 32
    assert all("mellin_sigma" in c for c in mellin_configs)


def test_load_pair_csv_with_filters():
    """Test CSV loading with date filters."""
    from spot_bot.benchmark_methods import load_pair_csv
    
    df = generate_synthetic_ohlcv(n_bars=1000)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        csv_path = tmpdir / "TEST_1h.csv"
        df.to_csv(csv_path, index=False)
        
        # Load without filters
        loaded = load_pair_csv(csv_path)
        assert len(loaded) == 1000
        
        # Load with start filter
        loaded_start = load_pair_csv(csv_path, start_date="2023-01-15")
        assert len(loaded_start) < 1000
        
        # Load with end filter
        loaded_end = load_pair_csv(csv_path, end_date="2023-01-15")
        assert len(loaded_end) < 1000
        
        # Load with both filters
        loaded_both = load_pair_csv(csv_path, start_date="2023-01-10", end_date="2023-01-20")
        assert len(loaded_both) < len(loaded_start)


def test_composite_score_calculation():
    """Test composite score calculation."""
    from spot_bot.benchmark_methods import compute_composite_score
    
    # Create sample results
    results_df = pd.DataFrame({
        "sharpe": [1.5, 0.5, 2.0],
        "max_drawdown": [-0.2, -0.1, -0.3],
        "turnover": [10.0, 20.0, 15.0],
    })
    
    scores = compute_composite_score(results_df, walk_forward=False)
    
    # Verify scores are computed
    assert len(scores) == 3
    assert all(pd.notna(scores))
    
    # Higher sharpe with lower drawdown should score better
    # scores[2] has highest sharpe (2.0) but highest drawdown (-0.3)
    # scores[0] has good sharpe (1.5) and medium drawdown (-0.2)
    # Score formula: sharpe - 0.5*abs(max_dd) - 0.05*turnover_norm
    # Higher score is better


def test_leaderboard_creation():
    """Test leaderboard creation from results."""
    from spot_bot.benchmark_methods import create_leaderboards
    
    # Create sample results with multiple pairs and methods
    results = []
    for pair in ["BTCUSDT", "ETHUSDT"]:
        for method in ["cepstrum", "complex_cepstrum"]:
            results.append({
                "pair": pair,
                "method_name": method,
                "sharpe": np.random.uniform(0.5, 2.0),
                "max_drawdown": np.random.uniform(-0.3, -0.1),
                "final_return": np.random.uniform(0.1, 0.5),
                "volatility": np.random.uniform(0.1, 0.3),
                "turnover": np.random.uniform(5.0, 20.0),
            })
    
    results_df = pd.DataFrame(results)
    
    # Create leaderboards
    methods_lb, pairs_lb = create_leaderboards(results_df, walk_forward=False)
    
    # Verify structure
    assert len(methods_lb) == 2  # 2 unique methods
    assert len(pairs_lb) == 2  # 2 unique pairs
    assert "method_name" in methods_lb.columns
    assert "pair" in pairs_lb.columns
    assert "score_mean" in methods_lb.columns
    assert "score_mean" in pairs_lb.columns


def test_walk_forward_mode():
    """Test walk-forward validation result structure."""
    from spot_bot.benchmark_methods import run_walk_forward_single
    
    df = generate_synthetic_ohlcv(n_bars=2000)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        csv_path = tmpdir / "BTCUSDT_1h.csv"
        df.to_csv(csv_path, index=False)
        
        # Load the CSV
        from spot_bot.benchmark_methods import load_pair_csv
        loaded_df = load_pair_csv(csv_path)
        
        # Run walk-forward
        config = {"psi_mode": "cepstrum"}
        result = run_walk_forward_single(
            pair="BTCUSDT",
            ohlcv_df=loaded_df,
            config_dict=config,
            train_bars=500,
            test_bars=200,
            fee_rate=0.0005,
            slippage_bps=0.0,
            initial_equity=1000.0,
            periods_per_year=24 * 365,
        )
        
        # Verify result structure
        assert result is not None
        assert "mean_sharpe" in result
        assert "mean_max_drawdown" in result
        assert "num_folds" in result
        assert result["num_folds"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
