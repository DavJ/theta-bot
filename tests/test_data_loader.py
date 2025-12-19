"""Test data loader with various CSV formats including Binance kline format."""

import io
import pandas as pd
import pytest
from theta_bot_averaging.data import load_dataset


def test_load_binance_kline_format_ms_timestamps(tmp_path):
    """Test loading Binance kline CSV with millisecond epoch timestamps."""
    # Create a test CSV with Binance kline format (ms epoch timestamps)
    # These timestamps represent candle OPEN times
    csv_content = """timestamp,open,high,low,close,volume
1711929600000,71280.00,71288.23,70844.14,71152.86,1116.66
1711933200000,71152.86,71172.08,70794.12,70819.25,882.39
1711936800000,70819.26,70902.80,70759.39,70866.98,628.48
"""
    
    # Write to temporary file
    csv_path = tmp_path / "test_binance.csv"
    csv_path.write_text(csv_content)
    
    # Load the dataset
    df = load_dataset(str(csv_path))
    
    # Assert index is DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
    
    # Assert timezone is UTC
    assert df.index.tz is not None, "Index should have timezone"
    assert str(df.index.tz) == "UTC", "Index should be in UTC timezone"
    
    # Assert timestamps are CLOSE times (open + 1 hour for 1h candles)
    # First timestamp: 1711929600000 ms = 2024-04-01 00:00:00 UTC (open)
    # After +1h shift: 2024-04-01 01:00:00 UTC (close)
    expected_first_close = pd.Timestamp("2024-04-01 01:00:00", tz="UTC")
    assert df.index[0] == expected_first_close, (
        f"First timestamp should be close time (open+1h): "
        f"expected {expected_first_close}, got {df.index[0]}"
    )
    
    # Second timestamp: 1711933200000 ms = 2024-04-01 01:00:00 UTC (open)
    # After +1h shift: 2024-04-01 02:00:00 UTC (close)
    expected_second_close = pd.Timestamp("2024-04-01 02:00:00", tz="UTC")
    assert df.index[1] == expected_second_close
    
    # Assert required columns are present
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    
    # Assert values are preserved
    assert df.iloc[0]["close"] == pytest.approx(71152.86)
    assert df.iloc[1]["close"] == pytest.approx(70819.25)
    
    # Assert index is monotonic increasing
    assert df.index.is_monotonic_increasing


def test_load_string_datetime_format(tmp_path):
    """Test loading CSV with string datetime format."""
    csv_content = """datetime,open,high,low,close,volume
2024-04-01 01:00:00,71280.00,71288.23,70844.14,71152.86,1116.66
2024-04-01 02:00:00,71152.86,71172.08,70794.12,70819.25,882.39
2024-04-01 03:00:00,70819.26,70902.80,70759.39,70866.98,628.48
"""
    
    csv_path = tmp_path / "test_string_datetime.csv"
    csv_path.write_text(csv_content)
    
    df = load_dataset(str(csv_path))
    
    # Assert index is DatetimeIndex with UTC timezone
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"
    
    # Assert no shift applied (already close times)
    expected_first = pd.Timestamp("2024-04-01 01:00:00", tz="UTC")
    assert df.index[0] == expected_first


def test_load_dataset_schema_validation(tmp_path):
    """Test that schema validation catches missing columns."""
    # Missing 'close' column
    csv_content = """timestamp,open,high,low,volume
1711929600000,71280.00,71288.23,70844.14,1116.66
1711933200000,71152.86,71172.08,70794.12,882.39
"""
    
    csv_path = tmp_path / "test_invalid.csv"
    csv_path.write_text(csv_content)
    
    with pytest.raises(ValueError, match="Missing required columns"):
        load_dataset(str(csv_path))


def test_numeric_index_detection(tmp_path):
    """Test that numeric ms epoch timestamps are correctly detected."""
    # Create DataFrame with numeric index in ms epoch range
    csv_content = """timestamp,open,high,low,close,volume
1711929600000,71280.00,71288.23,70844.14,71152.86,1116.66
1711933200000,71152.86,71172.08,70794.12,70819.25,882.39
"""
    
    df = pd.read_csv(io.StringIO(csv_content), index_col=0)
    
    # Verify it's numeric
    assert pd.api.types.is_numeric_dtype(df.index)
    
    # Verify it's in ms epoch range (>= 10^12)
    assert df.index[0] >= 10**12
    
    # Now test through load_dataset using tmp_path fixture
    csv_path = tmp_path / "test_numeric_ms.csv"
    csv_path.write_text(csv_content)
    
    result_df = load_dataset(str(csv_path))
    
    # Should be converted to DatetimeIndex
    assert isinstance(result_df.index, pd.DatetimeIndex)
    assert result_df.index.tz is not None
