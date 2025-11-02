#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_eval_metrics.py
-------------------
Unit tests for eval_metrics.py script.

Run with: python3 test_eval_metrics.py
"""

import unittest
import sys
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
import eval_metrics


class TestEvalMetrics(unittest.TestCase):
    """Test cases for eval_metrics module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, 'test_data')
        os.makedirs(self.data_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def create_test_csv(self, filename, n=100):
        """Create a test CSV file with synthetic price data."""
        np.random.seed(42)
        timestamps = pd.date_range(start='2024-01-01', periods=n, freq='1h')
        
        price_start = 50000.0
        returns = np.random.normal(0.0001, 0.02, n)
        prices = price_start * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, n)),
            'high': prices * (1 + np.random.uniform(0, 0.01, n)),
            'low': prices * (1 + np.random.uniform(-0.01, 0, n)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, n)
        })
        
        df['predicted_return'] = df['close'].pct_change().shift(1)
        
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        return filepath
    
    def test_compute_returns(self):
        """Test compute_returns function."""
        prices = np.array([100, 105, 103, 110])
        returns = eval_metrics.compute_returns(prices)
        
        expected = np.array([0.05, -0.019047619, 0.067961165])
        np.testing.assert_array_almost_equal(returns, expected, decimal=5)
    
    def test_compute_correlation(self):
        """Test compute_correlation function."""
        pred = np.array([0.01, 0.02, -0.01, 0.03])
        true = np.array([0.015, 0.018, -0.012, 0.028])
        
        corr = eval_metrics.compute_correlation(pred, true)
        self.assertGreater(corr, 0.9)  # Should be highly correlated
    
    def test_compute_hit_rate(self):
        """Test compute_hit_rate function."""
        pred = np.array([0.01, -0.02, 0.03, -0.01])
        true = np.array([0.015, -0.018, 0.028, 0.001])
        
        hit_rate = eval_metrics.compute_hit_rate(pred, true)
        self.assertEqual(hit_rate, 0.75)  # 3 out of 4 correct
    
    def test_compute_hit_rate_with_zeros(self):
        """Test that zeros in true returns count as misses."""
        pred = np.array([0.01, -0.02, 0.03, 0.01])
        true = np.array([0.015, -0.018, 0.0, 0.001])
        
        hit_rate = eval_metrics.compute_hit_rate(pred, true)
        self.assertEqual(hit_rate, 0.75)  # 3 out of 4 (zero counts as miss)
    
    def test_find_dataset_files(self):
        """Test finding dataset files in repository."""
        # Create test CSV
        self.create_test_csv('BTCUSDT_1h.csv')
        
        # Find datasets
        datasets = eval_metrics.find_dataset_files(self.test_dir)
        
        self.assertGreater(len(datasets), 0)
        self.assertTrue(any('BTCUSDT_1h.csv' in str(d[0]) for d in datasets))
    
    def test_evaluate_dataset(self):
        """Test evaluating a dataset file."""
        csv_path = self.create_test_csv('test_data.csv', n=200)
        
        metrics = eval_metrics.evaluate_dataset(
            csv_path,
            fee_mode='no_fees',
            taker_fee=0.001,
            start_capital=1000.0
        )
        
        self.assertIsNotNone(metrics)
        self.assertIn('corr_pred_true', metrics)
        self.assertIn('hit_rate', metrics)
        self.assertIn('total_pnl_usdt', metrics)
        self.assertIn('end_capital_usdt', metrics)
        
        # Check reasonable values
        self.assertTrue(-1 <= metrics['corr_pred_true'] <= 1)
        self.assertTrue(0 <= metrics['hit_rate'] <= 1)
        self.assertGreater(metrics['end_capital_usdt'], 0)
    
    def test_simulate_trading(self):
        """Test trading simulation."""
        # Create simple test data
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='1h'),
            'close': [100 + i for i in range(50)],
            'predicted_return': [0.01] * 50  # Always predict up
        })
        
        result = eval_metrics.simulate_trading(df, fee_rate=0.0, start_capital=1000.0)
        
        self.assertIn('total_pnl_usdt', result)
        self.assertIn('end_capital_usdt', result)
        self.assertIn('num_trades', result)
        
        # With always positive predictions and rising prices, should be profitable
        self.assertGreater(result['end_capital_usdt'], 1000.0)


class TestDownloadBinanceData(unittest.TestCase):
    """Test cases for download_binance_data module."""
    
    def test_import(self):
        """Test that download_binance_data can be imported."""
        import download_binance_data
        self.assertTrue(hasattr(download_binance_data, 'fetch_binance_klines'))
        self.assertTrue(hasattr(download_binance_data, 'download_historical_data'))


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestEvalMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestDownloadBinanceData))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
