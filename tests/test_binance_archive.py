"""Official archive format and checksum regressions, without network calls."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts.download_binance_archive import fetch_month, parse_archive


def _archive(month, unit, drop_last=False):
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    ts = pd.date_range(start, start + pd.offsets.MonthBegin(1), freq="1h", inclusive="left")
    if drop_last:
        ts = ts[:-1]
    divisor = 1000 if unit == "us" else 1_000_000
    rows = np.column_stack([ts.as_unit("ns").asi8 // divisor] + [np.full(len(ts), value) for value in (100, 101, 99, 100, 10, 0, 0, 0, 0, 0, 0)])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        zipped.writestr(f"BTCUSDT-1h-{month}.csv", pd.DataFrame(rows).to_csv(index=False, header=False))
    return buffer.getvalue()


@pytest.mark.parametrize("month,unit", [("2024-04", "ms"), ("2026-04", "us")])
def test_archive_timestamp_units_and_month_boundaries(month, unit):
    frame = parse_archive(_archive(month, unit), "BTCUSDT", "1h", month)
    assert frame.timestamp.iloc[0] == pd.Timestamp(f"{month}-01", tz="UTC")
    assert frame.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all()


def test_missing_archive_candle_is_rejected():
    with pytest.raises(ValueError, match="missing, duplicate"):
        parse_archive(_archive("2026-04", "us", drop_last=True), "BTCUSDT", "1h", "2026-04")


def test_explicit_research_gap_mode_preserves_gap_without_filling():
    frame = parse_archive(_archive("2026-04", "us", drop_last=True), "BTCUSDT", "1h", "2026-04", allow_gaps=True)
    assert len(frame) == 719
    assert frame.timestamp.iloc[-1] == pd.Timestamp("2026-04-30T22:00Z")


def test_mismatched_archive_checksum_is_rejected(monkeypatch):
    class Response:
        text = "0" * 64 + "  archive.zip"
        content = _archive("2026-04", "us")

        def raise_for_status(self):
            pass

    monkeypatch.setattr("scripts.download_binance_archive.requests.get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_month("BTCUSDT", "1h", "2026-04")
