"""
Multi-candidate log-phase sweep for BTC/USDT (or any symbol).

Build multiple positive input series x_t from OHLCV, compute log-phase
features, and evaluate their predictive value against future targets.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from btc_log_phase import (
    fetch_ohlcv_binance,
    log_phase,
    phase_embedding,
    risk_filter_backtest,
    rolling_phase_concentration,
    uniformity_test,
)

DEFAULT_CANDIDATES = [
    "price",
    "abslogret",
    "rv",
    "atr",
    "volume",
    "distema",
]

EPS_BUCKET = 1e-12
EPS_LOG = 1e-12
ZERO_STD_REPLACEMENT = 1.0
AUC_TOP_QUANTILE = 0.8  # 80th percentile threshold (top 20%)

# Time-phase presets (hours). Used to build a second circular phase from timestamps.
TIME_PHASE_PRESETS_HOURS = {
    "none": 0.0,
    "day": 24.0,
    "week": 24.0 * 7.0,
    "month": 24.0 * 30.0,
    "lunar": 24.0 * 29.53059,  # synodic month (approx)
}

THETA_COLUMN_NAMES = ["theta_coeff", "theta_coeffs", "theta_energy"]


def _fmt_three(x: float) -> str:
    return f"{x:0.3f}"


def build_candidate_series(df: pd.DataFrame, name: str, args: argparse.Namespace) -> pd.Series:
    """
    Construct candidate positive series x_t from OHLCV data.
    The output aligns with df index and may contain NaNs from rolling windows.
    """
    name = name.lower()
    rv_window = getattr(args, "rv_window", 24)
    atr_window = getattr(args, "atr_window", 14)
    ema_window = getattr(args, "ema_window", 50)
    if name == "price":
        series = df["close"]
    elif name == "abslogret":
        lr = np.log(df["close"] / df["close"].shift(1))
        series = lr.abs()
    elif name == "rv":
        lr = np.log(df["close"] / df["close"].shift(1))
        series = lr.rolling(window=rv_window, min_periods=rv_window).std()
    elif name == "atr":
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        series = tr.rolling(window=atr_window, min_periods=atr_window).mean()
    elif name == "volume":
        series = df["volume"]
        if getattr(args, "volume_roll", 0):
            # Optional rolling sum window for volume
            series = series.rolling(
                window=args.volume_roll, min_periods=args.volume_roll
            ).sum()
    elif name == "distema":
        ema = df["close"].ewm(span=ema_window, adjust=False).mean()
        series = (df["close"] - ema).abs()
    else:
        raise ValueError(f"Unknown candidate: {name}")

    series = series.replace([np.inf, -np.inf], np.nan)
    return series


def build_targets(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Construct strictly future-aligned targets."""
    target_window = getattr(args, "target_window", 24)
    horizon = getattr(args, "horizon", 24)
    lr = np.log(df["close"] / df["close"].shift(1))
    future_lr = lr.shift(-1)
    future_vol = (
        future_lr.rolling(window=target_window, min_periods=target_window)
        .std()
        .shift(-(target_window - 1))
    )
    future_absret = (np.log(df["close"].shift(-horizon) / df["close"])).abs()
    return pd.DataFrame({"y_vol": future_vol, "y_absret": future_absret})


def rolling_torus_concentration(
    cos_s: np.ndarray,
    sin_s: np.ndarray,
    cos_t: np.ndarray,
    sin_t: np.ndarray,
    window: int = 256,
) -> np.ndarray:
    """Rolling mean resultant length on a 2-torus embedded in R^4.

    Embed each sample as v = (cos_s, sin_s, cos_t, sin_t) and compute:
        C = ||mean(v)|| / sqrt(2)
    so that C stays in [0, 1], reaching 1 when both circular phases are perfectly aligned.

    This is a minimal torus extension: scale-phase × time-phase.
    """
    n = len(cos_s)
    out = np.full(n, np.nan, dtype=float)  # NaNs preserve causality until at least `window` samples exist
    norm_scale = 1.0 / math.sqrt(2.0)  # sqrt(2) is the max resultant when both unit circles align
    for i in range(n):
        lo = max(0, i - window + 1)
        m1 = np.mean(cos_s[lo : i + 1])
        m2 = np.mean(sin_s[lo : i + 1])
        m3 = np.mean(cos_t[lo : i + 1])
        m4 = np.mean(sin_t[lo : i + 1])
        out[i] = float(
            np.sqrt(m1 * m1 + m2 * m2 + m3 * m3 + m4 * m4) * norm_scale
        )
    return out


def rolling_internal_concentration(
    cos_phi: np.ndarray,
    sin_phi: np.ndarray,
    cos_psi: np.ndarray,
    sin_psi: np.ndarray,
    window: int = 256,
) -> np.ndarray:
    """Rolling mean resultant length for (phi, psi) on a torus embedded in R^4.

    C_int = ||rolling_mean([cos φ, sin φ, cos ψ, sin ψ])|| / 2
    which stays in [0, 1]. The 1/2 normalization follows the ψ integration
    spec and differs from the time-phase torus scale (which uses √2).
    """
    n = len(cos_phi)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        has_nan = (
            np.isnan(cos_phi[lo : i + 1]).any()
            or np.isnan(sin_phi[lo : i + 1]).any()
            or np.isnan(cos_psi[lo : i + 1]).any()
            or np.isnan(sin_psi[lo : i + 1]).any()
        )
        if has_nan:
            continue
        m1 = np.mean(cos_phi[lo : i + 1])
        m2 = np.mean(sin_phi[lo : i + 1])
        m3 = np.mean(cos_psi[lo : i + 1])
        m4 = np.mean(sin_psi[lo : i + 1])
        out[i] = float(np.sqrt(m1 * m1 + m2 * m2 + m3 * m3 + m4 * m4) / 2.0)
    return out


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    std = std.replace(0.0, np.nan)
    return (series - mean) / std


def _hilbert_phase(series: pd.Series, window: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        window_arr = arr[i - window + 1 : i + 1]
        if np.isnan(window_arr).any():
            continue
        analytic = signal.hilbert(window_arr)
        out[i] = (np.angle(analytic[-1]) / (2 * np.pi)) % 1.0
    return pd.Series(out, index=series.index)


def _rolling_pc1(df: pd.DataFrame, window: int) -> pd.Series:
    cols = [c for c in ["close", "high", "low", "volume"] if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    data = df[cols].astype(float).to_numpy()
    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        window_arr = data[i - window + 1 : i + 1]
        if np.isnan(window_arr).any():
            continue
        mean = window_arr.mean(axis=0)
        std = window_arr.std(axis=0)
        std[std == 0.0] = ZERO_STD_REPLACEMENT  # avoid division by zero; constant features normalize to zero so PCA still runs
        normed = (window_arr - mean) / std
        pca = PCA(n_components=1)
        comp = pca.fit_transform(normed)
        out[i] = comp[-1, 0]
    return pd.Series(out, index=df.index)


def _stable_complex_log(spectrum: np.ndarray) -> np.ndarray:
    """Compute a stable complex logarithm preserving magnitude and phase."""
    return np.log(np.abs(spectrum) + EPS_LOG) + 1j * np.angle(spectrum)


def _extract_cepstrum_params(args: argparse.Namespace) -> tuple[int, float, int | None]:
    min_bin = max(1, int(getattr(args, "cepstrum_min_bin", 2)))
    max_frac = float(getattr(args, "cepstrum_max_frac", 0.25))
    topk_raw = getattr(args, "cepstrum_topk", None)
    if topk_raw is None:
        topk: int | None = None
    else:
        try:
            topk = int(topk_raw)
        except (TypeError, ValueError) as e:
            raise ValueError("--cepstrum-topk/--cepstrum_topk must be an integer or omitted.") from e
        if topk < 1:
            topk = None
    return min_bin, max_frac, topk


def _cepstral_phase(
    series: pd.Series, window: int, min_bin: int = 2, max_frac: float = 0.25, topk: int | None = None
) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    min_bin = max(1, int(min_bin))
    max_frac = float(max_frac)
    for i in range(window - 1, n):
        window_arr = arr[i - window + 1 : i + 1]
        if np.isnan(window_arr).any():
            continue
        spectrum = np.fft.fft(window_arr)
        log_mag = np.log(np.abs(spectrum) + EPS_LOG)
        cepstrum = np.fft.ifft(log_mag)
        candidate_max = min(int(window * max_frac), window // 2, len(cepstrum))
        max_bin = max(candidate_max, min_bin + 1)
        # clamp again after enforcing minimum to avoid overruns when min_bin is large
        max_bin = min(max_bin, len(cepstrum))
        if min_bin >= max_bin:
            continue
        candidate_slice = cepstrum[min_bin:max_bin]
        mags = np.abs(candidate_slice)
        # When topk=1 or None, fall back to the single dominant bin
        if topk is not None and topk >= 2:
            k = min(topk, len(candidate_slice))
            idxs = np.argpartition(mags, -k)[-k:]
            angles = np.angle(candidate_slice[idxs])
            weights = mags[idxs]
            combined = np.sum(weights * np.exp(1j * angles))
            ang = float(np.angle(combined))
        else:
            best_idx = int(np.argmax(mags))
            ang = float(np.angle(candidate_slice[best_idx]))
        out[i] = (ang / (2 * np.pi)) % 1.0
    return pd.Series(out, index=series.index)


def compute_internal_phase(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series | None:
    mode = getattr(args, "psi_mode", None)
    if not mode or str(mode).lower() == "none":
        return None
    mode = str(mode).lower()
    window = int(getattr(args, "psi_window", getattr(args, "conc_window", 256)))
    rv_window = int(getattr(args, "rv_window", 24))
    if window <= 0:
        return None

    if mode == "hilbert_rv":
        lr = np.log(df["close"] / df["close"].shift(1))
        rv = lr.rolling(window=rv_window, min_periods=rv_window).std()
        rv_z = _rolling_zscore(rv, window)
        return _hilbert_phase(rv_z, window)

    if mode == "pca_hilbert":
        pc1 = _rolling_pc1(df, window)
        pc1_z = _rolling_zscore(pc1, window)
        return _hilbert_phase(pc1_z, window)

    if mode == "theta_phase":
        theta_col = next((c for c in THETA_COLUMN_NAMES if c in df.columns), None)
        if theta_col is None:
            return pd.Series(np.nan, index=df.index)
        vals_raw = pd.to_numeric(df[theta_col], errors="coerce")
        vals = np.asarray(vals_raw.to_numpy(), dtype=complex)
        if np.isnan(vals_raw).all():
            return pd.Series(np.nan, index=df.index)
        return pd.Series((np.angle(vals) / (2 * np.pi)) % 1.0, index=df.index)

    if mode == "cepstrum":
        lr = np.log(df["close"] / df["close"].shift(1))
        rv = lr.rolling(window=rv_window, min_periods=rv_window).std()
        log_rv = np.log(np.abs(rv) + EPS_LOG)
        min_bin, max_frac, topk = _extract_cepstrum_params(args)
        return _cepstral_phase(log_rv, window, min_bin=min_bin, max_frac=max_frac, topk=topk)

    raise ValueError(f"Unknown psi mode: {mode}")


def compute_features(
    x: pd.Series, df: pd.DataFrame, args: argparse.Namespace, psi_series: pd.Series | None = None
) -> pd.DataFrame:
    """Compute log-phase derived features for a candidate series.

    If a time-phase is enabled (args.time_phase_hours > 0), also compute a second
    circular phase from real timestamps and a simple torus concentration.
    """
    base = getattr(args, "base", 10.0)
    conc_window = getattr(args, "conc_window", 256)
    phi = log_phase(x.to_numpy(), base=base)
    cos_phi, sin_phi = phase_embedding(phi)
    concentration = rolling_phase_concentration(phi, window=conc_window)

    out = pd.DataFrame(
        {
            "phi": phi,
            "cos_phi": cos_phi,
            "sin_phi": sin_phi,
            "concentration": concentration,
        },
        index=x.index,
    )

    if psi_series is None:
        psi_series = compute_internal_phase(df, args)
    has_psi = psi_series is not None and not psi_series.dropna().empty
    # Optional: add internal phase psi and torus concentration (phi, psi)
    if has_psi:
        psi_vals = psi_series.to_numpy(dtype=float)
        cos_psi, sin_psi = phase_embedding(psi_vals)
        torus_conc = rolling_internal_concentration(cos_phi, sin_phi, cos_psi, sin_psi, conc_window)
        out["psi"] = psi_vals
        out["cos_psi"] = cos_psi
        out["sin_psi"] = sin_psi
        out["c_int"] = torus_conc
    else:
        # Optional: add time-phase and torus concentration
        time_period_hours = float(getattr(args, "time_phase_hours", 0.0) or 0.0)
        if time_period_hours > 0 and "dt" in df.columns:
            dt = pd.to_datetime(df["dt"], utc=True)
            t0 = dt.iloc[0]
            delta_s = (dt - t0).dt.total_seconds().to_numpy(dtype=float)
            period_s = time_period_hours * 3600.0
            phi_time = (delta_s / period_s) % 1.0
            cos_t, sin_t = phase_embedding(phi_time)
            torus_conc = rolling_torus_concentration(
                cos_phi, sin_phi, cos_t, sin_t, window=conc_window
            )
            out["phi_time"] = phi_time
            out["cos_time"] = cos_t
            out["sin_time"] = sin_t
            out["torus_concentration"] = torus_conc

    return out


def _bucket_stats(feature: pd.Series, target: pd.Series, buckets: int = 5) -> Dict[str, object]:
    """Return bucket means, counts, ratio, and monotonicity."""
    out: Dict[str, object] = {"bucket_counts": {}, "bucket_means": {}, "bucket_ratio": math.nan}
    try:
        bucketed = pd.qcut(feature, buckets, labels=False, duplicates="drop")
    except ValueError:
        return out
    means = target.groupby(bucketed).mean()
    counts = bucketed.value_counts().sort_index()
    out["bucket_counts"] = counts.to_dict()
    out["bucket_means"] = means.to_dict()
    clean_means = means.dropna()
    if clean_means.empty or clean_means.min() < EPS_BUCKET:
        out["bucket_ratio"] = math.nan
    else:
        out["bucket_ratio"] = float(clean_means.max() / clean_means.min())
    diffs = clean_means.diff().dropna()
    if diffs.empty:
        out["bucket_monotonic"] = "flat"
    elif (diffs >= 0).all():
        out["bucket_monotonic"] = "increasing"
    elif (diffs <= 0).all():
        out["bucket_monotonic"] = "decreasing"
    else:
        out["bucket_monotonic"] = "mixed"
    return out


def evaluate_candidate(features: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, object]:
    """Compute ICs, uniformity, bucket stats, and optional AUC for a candidate."""
    joined = pd.concat(
        [features[["phi", "cos_phi", "sin_phi", "concentration"]], targets], axis=1
    ).dropna()

    metrics: Dict[str, object] = {
        "ks_stat": math.nan,
        "ks_p": math.nan,
        "conc_median": math.nan,
        "conc_p95": math.nan,
        "ic_conc_y_vol": math.nan,
        "ic_conc_y_absret": math.nan,
        "ic_cos_y_vol": math.nan,
        "ic_sin_y_vol": math.nan,
        "bucket_counts": {},
        "bucket_means": {},
        "bucket_ratio": math.nan,
        "bucket_monotonic": None,
        "auc_conc_y_vol": math.nan,
        # Torus-internal (phi, psi) metrics when psi is enabled
        "c_int_median": math.nan,
        "c_int_p95": math.nan,
        "ic_c_int_y_vol": math.nan,
        "ic_c_int_y_absret": math.nan,
        "c_int_bucket_ratio": math.nan,
        "auc_c_int_y_vol": math.nan,
        # Ensemble score S = mean(rank(C_scale), rank(C_int))
        "ic_s_y_vol": math.nan,
        "s_bucket_counts": {},
        "s_bucket_means": {},
        "s_bucket_ratio": math.nan,
        "auc_s_y_vol": math.nan,
        # Optional torus (scale-phase × time-phase) metrics; present only when time-phase enabled.
        "torus_conc_median": math.nan,
        "torus_conc_p95": math.nan,
        "ic_torus_y_vol": math.nan,
        "ic_torus_y_absret": math.nan,
        "torus_bucket_ratio": math.nan,
        "auc_torus_y_vol": math.nan,
    }

    if joined.empty:
        return metrics

    metrics["ks_stat"], metrics["ks_p"] = uniformity_test(joined["phi"].to_numpy())
    metrics["conc_median"] = float(joined["concentration"].median())
    metrics["conc_p95"] = float(joined["concentration"].quantile(0.95))
    metrics["ic_conc_y_vol"] = joined["concentration"].corr(joined["y_vol"], method="spearman")
    metrics["ic_conc_y_absret"] = joined["concentration"].corr(
        joined["y_absret"], method="spearman"
    )
    metrics["ic_cos_y_vol"] = joined["cos_phi"].corr(joined["y_vol"], method="spearman")
    metrics["ic_sin_y_vol"] = joined["sin_phi"].corr(joined["y_vol"], method="spearman")

    bucket_out = _bucket_stats(joined["concentration"], joined["y_vol"])
    metrics.update(bucket_out)

    # Optional classification: top 20% vol (80th percentile threshold); higher concentration
    # is expected to coincide with higher future volatility.
    try:
        threshold = joined["y_vol"].quantile(AUC_TOP_QUANTILE)
        y_class = (joined["y_vol"] >= threshold).astype(int)
        if y_class.nunique() > 1:
            metrics["auc_conc_y_vol"] = roc_auc_score(y_class, joined["concentration"])
    except (ValueError, TypeError):
        metrics["auc_conc_y_vol"] = math.nan

    if "c_int" in features.columns:
        joined_int = pd.concat([features[["c_int"]], targets], axis=1).dropna()
        if not joined_int.empty:
            metrics["c_int_median"] = float(joined_int["c_int"].median())
            metrics["c_int_p95"] = float(joined_int["c_int"].quantile(0.95))
            metrics["ic_c_int_y_vol"] = joined_int["c_int"].corr(
                joined_int["y_vol"], method="spearman"
            )
            metrics["ic_c_int_y_absret"] = joined_int["c_int"].corr(
                joined_int["y_absret"], method="spearman"
            )
            c_int_bucket = _bucket_stats(joined_int["c_int"], joined_int["y_vol"])
            metrics["c_int_bucket_ratio"] = c_int_bucket.get("bucket_ratio", math.nan)
            try:
                threshold_int = joined_int["y_vol"].quantile(AUC_TOP_QUANTILE)
                y_class_int = (joined_int["y_vol"] >= threshold_int).astype(int)
                if y_class_int.nunique() > 1:
                    metrics["auc_c_int_y_vol"] = roc_auc_score(y_class_int, joined_int["c_int"])
            except (ValueError, TypeError):
                metrics["auc_c_int_y_vol"] = math.nan
    if "c_int" in features.columns and "concentration" in features.columns:
        joined_s = pd.concat(
            [features[["concentration", "c_int"]], targets],
            axis=1,
        ).dropna()
        if not joined_s.empty:
            rank_scale = joined_s["concentration"].rank(pct=True, method="average")
            rank_int = joined_s["c_int"].rank(pct=True, method="average")
            # Intentionally equal-weight blend of ranked scale concentration and internal-phase concentration (c_int)
            s_score = (rank_scale + rank_int) / 2.0
            metrics["ic_s_y_vol"] = s_score.corr(joined_s["y_vol"], method="spearman")
            s_bucket = _bucket_stats(s_score, joined_s["y_vol"])
            metrics["s_bucket_counts"] = s_bucket.get("bucket_counts", {})
            metrics["s_bucket_means"] = s_bucket.get("bucket_means", {})
            metrics["s_bucket_ratio"] = s_bucket.get("bucket_ratio", math.nan)
            try:
                threshold_s = joined_s["y_vol"].quantile(AUC_TOP_QUANTILE)
                y_class_s = (joined_s["y_vol"] >= threshold_s).astype(int)
                if y_class_s.nunique() > 1:
                    metrics["auc_s_y_vol"] = roc_auc_score(y_class_s, s_score)
            except (ValueError, TypeError):
                metrics["auc_s_y_vol"] = math.nan

    # Optional torus concentration evaluation
    if "torus_concentration" in features.columns:
        joined2 = pd.concat([features[["torus_concentration"]], targets], axis=1).dropna()
        if not joined2.empty:
            metrics["torus_conc_median"] = float(joined2["torus_concentration"].median())
            metrics["torus_conc_p95"] = float(joined2["torus_concentration"].quantile(0.95))
            metrics["ic_torus_y_vol"] = joined2["torus_concentration"].corr(
                joined2["y_vol"], method="spearman"
            )
            metrics["ic_torus_y_absret"] = joined2["torus_concentration"].corr(
                joined2["y_absret"], method="spearman"
            )
            torus_bucket = _bucket_stats(joined2["torus_concentration"], joined2["y_vol"])
            metrics["torus_bucket_ratio"] = torus_bucket.get("bucket_ratio", math.nan)
            try:
                threshold2 = joined2["y_vol"].quantile(AUC_TOP_QUANTILE)
                y_class2 = (joined2["y_vol"] >= threshold2).astype(int)
                if y_class2.nunique() > 1:
                    metrics["auc_torus_y_vol"] = roc_auc_score(
                        y_class2, joined2["torus_concentration"]
                    )
            except (ValueError, TypeError):
                metrics["auc_torus_y_vol"] = math.nan

    return metrics


def rank_candidates(results: Iterable[Dict[str, object]]) -> pd.DataFrame:
    """Return a summary DataFrame sorted by absolute IC on y_vol."""
    table = pd.DataFrame(results)
    if table.empty:
        return table
    table = table.copy()
    table["abs_ic_conc_y_vol"] = table["ic_conc_y_vol"].abs()
    abs_cols = [table["abs_ic_conc_y_vol"]]
    if "ic_c_int_y_vol" in table.columns:
        table["abs_ic_c_int_y_vol"] = table["ic_c_int_y_vol"].abs()
        abs_cols.append(table["abs_ic_c_int_y_vol"])
    if "ic_torus_y_vol" in table.columns:
        table["abs_ic_torus_y_vol"] = table["ic_torus_y_vol"].abs()
        abs_cols.append(table["abs_ic_torus_y_vol"])
    abs_df = pd.concat(abs_cols, axis=1, ignore_index=True)
    table["abs_ic_best_y_vol"] = abs_df.max(axis=1)
    return table.sort_values(by=["abs_ic_best_y_vol", "bucket_ratio"], ascending=False)


def _plot_candidate(
    df: pd.DataFrame,
    features: pd.DataFrame,
    candidate: str,
    eq_bh: np.ndarray,
    eq_filtered: np.ndarray,
    save: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    rows = 4 if "psi" in features.columns else 3
    fig, axes = plt.subplots(rows, 1, figsize=(10, 4 * rows), sharex=False)
    axes[0].hist(features["phi"].dropna(), bins=60, color="steelblue", alpha=0.8)
    axes[0].set_title(f"Histogram of phi - {candidate}")
    axes[0].set_ylabel("Count")

    psi_present = "psi" in features.columns
    if psi_present:
        axes[1].plot(df["dt"], features["psi"], color="slateblue", linewidth=1.0)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("psi")
        axes[1].set_title(f"Internal phase ψ - {candidate}")

    conc_ax = axes[1 + int(psi_present)]
    if "c_int" in features.columns or "torus_concentration" in features.columns:
        conc_ax.plot(
            df["dt"],
            features["concentration"],
            color="darkorange",
            linewidth=1.0,
            label="Scale conc",
        )
        if "c_int" in features.columns:
            conc_ax.plot(
                df["dt"],
                features["c_int"],
                color="seagreen",
                linewidth=1.0,
                label="C_int (φ, ψ)",
            )
        if "torus_concentration" in features.columns:
            conc_ax.plot(
                df["dt"],
                features["torus_concentration"],
                color="teal",
                linewidth=1.0,
                label="Torus conc",
            )
        conc_ax.set_title(f"Rolling concentration {candidate} (scale vs torus)")
        conc_ax.legend()
    else:
        conc_ax.plot(df["dt"], features["concentration"], color="darkorange", linewidth=1.2)
        conc_ax.set_title(f"Rolling concentration {candidate}")
    conc_ax.set_ylabel("Concentration")

    eq_ax = axes[2 + int(psi_present)]
    eq_ax.plot(df["dt"], eq_bh, label="Buy & Hold", linewidth=1.2)
    eq_ax.plot(df["dt"], eq_filtered, label="Filtered", linewidth=1.2)
    eq_ax.set_title(f"Equity curves - {candidate}")
    eq_ax.set_ylabel("Equity")
    eq_ax.legend()

    plt.tight_layout()
    if save:
        Path("plots").mkdir(exist_ok=True)
        plt.savefig(Path("plots") / f"{candidate}_log_phase.png", dpi=150)
    else:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log-phase sweep over multiple candidates.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit-total", type=int, default=6000)
    parser.add_argument("--base", type=float, default=10.0)
    parser.add_argument("--conc-window", type=int, default=256)
    parser.add_argument("--thr", type=float, default=0.20)
    parser.add_argument("--rv-window", type=int, default=24)
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument("--ema-window", type=int, default=50)
    parser.add_argument(
        "--split",
        type=float,
        nargs="?",
        const=0.7,
        default=None,
        help=(
            "Optional train fraction for chronological split. "
            "When provided without value defaults to 0.7. Omit to use full sample."
        ),
    )
    parser.add_argument(
        "--embargo",
        type=int,
        default=None,
        help="Gap (in samples) excluded around split boundary; defaults to horizon when omitted.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap resamples on TEST set. 0 disables bootstrap.",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=72,
        help="Block size for contiguous block bootstrap (in samples).",
    )
    parser.add_argument(
        "--psi-mode",
        type=str,
        default="none",
        choices=["none", "hilbert_rv", "pca_hilbert", "theta_phase", "cepstrum"],
        help=(
            "Internal phase ψ method: hilbert_rv (Hilbert of realized vol), "
            "pca_hilbert (PCA then Hilbert), theta_phase (existing theta coefficients), "
            "cepstrum (cepstral analysis), or none."
        ),
    )
    parser.add_argument(
        "--psi-window",
        type=int,
        default=256,
        help="Rolling window for ψ estimation (Hilbert/PCA/Cepstrum).",
    )
    parser.add_argument(
        "--cepstrum-min-bin",
        type=int,
        default=2,
        help="Minimum quefrency bin (exclusive of DC) considered for cepstral ψ.",
    )
    parser.add_argument(
        "--cepstrum-max-frac",
        type=float,
        default=0.25,
        help="Maximum fraction of window length to include in cepstral search (e.g., 0.25 keeps bins up to w/4).",
    )
    parser.add_argument(
        "--cepstrum-topk",
        type=int,
        default=None,
        help="Optional: average phases of top-k cepstral peaks (k>1) to reduce jitter.",
    )
    parser.add_argument("--target-window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument(
        "--time-phase",
        type=str,
        default="none",
        help=(
            "Optional second circular phase derived from timestamps. "
            "Use one of: none, day, week, month, lunar, or a numeric value interpreted as hours (e.g., 168)."
        ),
    )
    parser.add_argument(
        "--volume-roll", type=int, default=0, help="Rolling sum window for volume."
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Comma-separated list of candidates. Default runs all.",
    )
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()
    # Resolve time-phase to hours
    tp = (args.time_phase or "none").strip().lower()
    if tp in TIME_PHASE_PRESETS_HOURS:
        args.time_phase_hours = TIME_PHASE_PRESETS_HOURS[tp]
    else:
        try:
            args.time_phase_hours = float(tp)
        except ValueError as e:
            raise ValueError(
                f"Invalid --time-phase '{tp}'. Use a preset {list(TIME_PHASE_PRESETS_HOURS.keys())} or numeric hours."
            ) from e
    return args


def _candidate_list(args: argparse.Namespace) -> List[str]:
    if args.candidates:
        return [c.strip() for c in args.candidates.split(",") if c.strip()]
    return DEFAULT_CANDIDATES


def main() -> None:
    args = parse_args()
    df = fetch_ohlcv_binance(
        symbol=args.symbol, timeframe=args.timeframe, limit_total=args.limit_total
    )
    psi_series = compute_internal_phase(df, args)
    targets = build_targets(df, args)
    results = []
    # Default fallback of 24 matches parse_args defaults for target_window/horizon
    max_lookahead = max(int(getattr(args, "target_window", 24)), int(getattr(args, "horizon", 24)))
    embargo = int(args.embargo) if args.embargo is not None else int(getattr(args, "horizon", 24))

    for cand in _candidate_list(args):
        x = build_candidate_series(df, cand, args)
        features = compute_features(x, df, args, psi_series=psi_series)

        metrics = evaluate_candidate(features, targets)
        metrics["candidate"] = cand

        bt_df = pd.DataFrame(
            {
                "ret": df["close"] / df["close"].shift(1),
                "concentration": features["concentration"],
            }
        )
        eq_bh, eq_filtered, bt_summary = risk_filter_backtest(bt_df, thr=args.thr)
        metrics["backtest"] = bt_summary
        results.append(metrics)

        if args.split is not None:
            split_idx = int(len(features) * float(args.split))
            train_end = max(0, split_idx - max(embargo, max_lookahead))
            train_feats = features.iloc[:train_end]
            train_targets = targets.iloc[:train_end]
            test_start = min(len(features), split_idx + embargo)
            test_feats = features.iloc[test_start:]
            test_targets = targets.iloc[test_start:]

            train_metrics = (
                evaluate_candidate(train_feats, train_targets) if train_end > 0 else None
            )
            test_metrics = (
                evaluate_candidate(test_feats, test_targets) if len(test_feats) > 0 else None
            )

            def _fmt_section(name: str, m: Dict[str, object]) -> str:
                if m is None:
                    return "insufficient data"

                def _val(key: str) -> float:
                    return float(m.get(key, math.nan))

                parts = [
                    f"{name} C IC={_val('ic_conc_y_vol'):.3f} "
                    f"AUC={_val('auc_conc_y_vol'):.3f}"
                ]
                if not math.isnan(m.get("ic_c_int_y_vol", math.nan)):
                    parts.append(
                        f"C_int IC={_val('ic_c_int_y_vol'):.3f} "
                        f"AUC={_val('auc_c_int_y_vol'):.3f}"
                    )
                if not math.isnan(m.get("ic_s_y_vol", math.nan)):
                    parts.append(
                        f"S IC={_val('ic_s_y_vol'):.3f} "
                        f"AUC={_val('auc_s_y_vol'):.3f}"
                    )
                return " | ".join(parts)

            print(f"[{cand}] TRAIN: {_fmt_section('Train', train_metrics)}")
            print(f"[{cand}] TEST:  {_fmt_section('Test', test_metrics)}")
            if test_metrics is not None and args.bootstrap > 0:
                rng = np.random.default_rng()

                def _sample_indices(n: int, block: int) -> np.ndarray:
                    if n == 0:
                        return np.array([], dtype=int)
                    blk = min(n, max(1, int(block)))
                    idxs: List[int] = []
                    while len(idxs) < n:
                        start = int(rng.integers(0, max(1, n - blk + 1)))
                        idxs.extend(range(start, min(n, start + blk)))
                    return np.array(idxs[:n], dtype=int)

                metric_keys = [
                    ("ic_conc_y_vol", "C IC"),
                    ("auc_conc_y_vol", "C AUC"),
                    ("ic_c_int_y_vol", "C_int IC"),
                    ("auc_c_int_y_vol", "C_int AUC"),
                    ("ic_s_y_vol", "S IC"),
                    ("auc_s_y_vol", "S AUC"),
                ]
                collected: Dict[str, List[float]] = {k: [] for k, _ in metric_keys}
                test_len = len(test_feats)
                for _ in range(int(args.bootstrap)):
                    idx = _sample_indices(test_len, args.block)
                    if idx.size == 0:
                        continue
                    boot_metrics = evaluate_candidate(test_feats.iloc[idx], test_targets.iloc[idx])
                    for k, _ in metric_keys:
                        val = boot_metrics.get(k, math.nan)
                        if not math.isnan(val):
                            collected[k].append(float(val))

                def _summary(vals: List[float]) -> str:
                    if not vals:
                        return "n/a"
                    arr = np.asarray(vals, dtype=float)
                    return (
                        f"median={np.median(arr):.3f} "
                        f"p05={np.percentile(arr, 5):.3f} "
                        f"p95={np.percentile(arr, 95):.3f}"
                    )

                print(f"[{cand}] BOOTSTRAP TEST (n={args.bootstrap}, block={args.block}):")
                for key, label in metric_keys:
                    print(f"  {label}: {_summary(collected[key])}")

        extras: List[str] = []
        if not math.isnan(metrics.get("ic_c_int_y_vol", math.nan)):
            extras.append(
                f" | C_int IC={metrics['ic_c_int_y_vol']:.3f}"
                f" C_int ratio={metrics.get('c_int_bucket_ratio', math.nan):.3f}"
                f" C_int AUC={metrics.get('auc_c_int_y_vol', math.nan):.3f}"
            )
        if not math.isnan(metrics.get("ic_s_y_vol", math.nan)):
            extras.append(
                f" | S IC={metrics['ic_s_y_vol']:.3f}"
                f" S ratio={metrics.get('s_bucket_ratio', math.nan):.3f}"
                f" S AUC={metrics.get('auc_s_y_vol', math.nan):.3f}"
            )
        if not math.isnan(metrics.get("ic_torus_y_vol", math.nan)):
            extras.append(
                f" | Torus IC={metrics['ic_torus_y_vol']:.3f}"
                f" Torus ratio={metrics.get('torus_bucket_ratio', math.nan):.3f}"
                f" Torus AUC={metrics.get('auc_torus_y_vol', math.nan):.3f}"
            )
        extra = "".join(extras)
        print(
            f"[{cand}] KS={metrics['ks_stat']:.3f}/{metrics['ks_p']:.3f} "
            f"IC(C,y_vol)={metrics['ic_conc_y_vol']:.3f} "
            f"IC(C,y_absret)={metrics['ic_conc_y_absret']:.3f} "
            f"Bucket ratio={metrics['bucket_ratio']:.3f} "
            f"AUC={metrics['auc_conc_y_vol']:.3f}{extra}"
        )

        if args.save_plots:
            _plot_candidate(df, features, cand, eq_bh, eq_filtered, save=True)

    ranked = rank_candidates(results)
    if not ranked.empty:
        cols = [
            "candidate",
            "ic_conc_y_vol",
            "ic_conc_y_absret",
            "bucket_ratio",
            "auc_conc_y_vol",
            "conc_median",
            "conc_p95",
        ]
        if "ic_c_int_y_vol" in ranked.columns and not ranked["ic_c_int_y_vol"].isna().all():
            cols += [
                "ic_c_int_y_vol",
                "ic_c_int_y_absret",
                "c_int_bucket_ratio",
                "auc_c_int_y_vol",
                "c_int_median",
                "c_int_p95",
            ]
        if "ic_torus_y_vol" in ranked.columns and not ranked["ic_torus_y_vol"].isna().all():
            cols += [
                "ic_torus_y_vol",
                "ic_torus_y_absret",
                "torus_bucket_ratio",
                "auc_torus_y_vol",
                "torus_conc_median",
                "torus_conc_p95",
            ]
        display_cols = [c for c in cols if c in ranked.columns]
        print("\nRanked summary (by |IC| on y_vol):")
        print(ranked[display_cols].to_string(index=False, float_format=_fmt_three))


if __name__ == "__main__":
    main()
