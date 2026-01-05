import subprocess
import sys

import numpy as np
import pandas as pd
from pathlib import Path

from spot_bot.run_live import CSV_OUTPUT_COLUMNS


def _build_synthetic_ohlcv(rows: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = 20000 + np.linspace(0, 100, rows)
    close = base + np.sin(np.linspace(0, 6.28, rows)) * 50
    open_ = close * 0.999
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = np.full(rows, 1.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_run_live_exports_csv(tmp_path):
    df = _build_synthetic_ohlcv()
    csv_in = tmp_path / "input.csv"
    df.reset_index().rename(columns={"index": "timestamp"}).to_csv(csv_in, index=False)

    csv_out = tmp_path / "out" / "result.csv"
    cmd = [
        sys.executable,
        "-m",
        "spot_bot.run_live",
        "--mode",
        "dryrun",
        "--csv-in",
        str(csv_in),
        "--csv-out",
        str(csv_out),
        "--rv-window",
        "24",
        "--conc-window",
        "64",
        "--psi-window",
        "64",
    ]

    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(cmd, check=True, cwd=repo_root)

    assert csv_out.exists()
    out_df = pd.read_csv(csv_out)
    assert set(CSV_OUTPUT_COLUMNS).issubset(set(out_df.columns))


def test_run_live_exports_feature_table(tmp_path):
    df = _build_synthetic_ohlcv()
    csv_in = tmp_path / "input.csv"
    df.reset_index().rename(columns={"index": "timestamp"}).to_csv(csv_in, index=False)

    csv_out = tmp_path / "out" / "features.csv"
    cmd = [
        sys.executable,
        "-m",
        "spot_bot.run_live",
        "--mode",
        "dryrun",
        "--csv-in",
        str(csv_in),
        "--csv-out",
        str(csv_out),
        "--rv-window",
        "24",
        "--conc-window",
        "64",
        "--psi-window",
        "64",
        "--csv-out-mode",
        "features",
        "--csv-out-tail",
        "50",
    ]

    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(cmd, check=True, cwd=repo_root)

    assert csv_out.exists()
    out_df = pd.read_csv(csv_out)
    assert len(out_df) > 10
    required_cols = set(CSV_OUTPUT_COLUMNS) - {"action"}
    assert required_cols.issubset(out_df.columns)


def test_run_live_exports_complex_cepstrum_features(tmp_path):
    rows = 600
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = 20000 + np.linspace(0, 200, rows)
    modulation = np.sin(np.linspace(0, 4 * np.pi, rows)) + 0.3 * np.sin(np.linspace(0, 8 * np.pi, rows) + 0.2)
    close = base + modulation * 80
    open_ = close * 0.999
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = np.full(rows, 1.0)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)

    csv_in = tmp_path / "input.csv"
    df.reset_index().rename(columns={"index": "timestamp"}).to_csv(csv_in, index=False)

    csv_out = tmp_path / "out" / "features_complex.csv"
    cmd = [
        sys.executable,
        "-m",
        "spot_bot.run_live",
        "--mode",
        "dryrun",
        "--csv-in",
        str(csv_in),
        "--csv-out",
        str(csv_out),
        "--rv-window",
        "24",
        "--conc-window",
        "128",
        "--psi-window",
        "128",
        "--psi-mode",
        "complex_cepstrum",
        "--csv-out-mode",
        "features",
        "--csv-out-tail",
        "200",
    ]

    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(cmd, check=True, cwd=repo_root)

    assert csv_out.exists()
    out_df = pd.read_csv(csv_out)
    debug_cols = ["psi_mode", "psi_n_star", "psi_c_real", "psi_c_imag", "psi_c_abs", "psi_angle_rad"]
    for col in debug_cols:
        assert col in out_df.columns
    psi_vals = out_df["psi"].dropna()
    assert not psi_vals.empty
    assert psi_vals.round(3).nunique() >= 10
    assert out_df["psi_mode"].dropna().iloc[-1] == "complex_cepstrum"
    assert out_df["psi_c_imag"].abs().max() > 0
