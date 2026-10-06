"""External features joined by publication availability, never by date alone."""
from __future__ import annotations

import numpy as np
import pandas as pd

MACRO = ("EURUSD", "USDJPY", "NASDAQCOM", "VIXCLS", "DGS10", "DCOILWTICO")
LAG_WINDOWS = ((0, 2), (3, 9), (10, 20))


def distributed_lags(features: pd.DataFrame) -> pd.DataFrame:
    """Past-only calendar-day buckets of signals already available to the bot.

    Publication latency has been applied before this step. These windows model
    a distributed market response, not the release time of an observation.
    Missing values remain missing; each bucket requires its complete history.
    """
    index = features.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None
            or not index.is_unique or not index.is_monotonic_increasing
            or not index.to_series().diff().dropna().eq(pd.Timedelta("1D")).all()):
        raise ValueError("Distributed lags require a complete timezone-aware daily grid")
    buckets = []
    for first, last in LAG_WINDOWS:
        width = last - first + 1
        bucket = features.rolling(width, min_periods=width).mean().shift(first)
        buckets.append(bucket.add_suffix(f"_lag{first}_{last}"))
    return pd.concat(buckets, axis=1)


def validate_signals(records: pd.DataFrame) -> pd.DataFrame:
    required = {"source", "event_time", "available_at", "value"}
    if not required.issubset(records) or records.empty:
        raise ValueError("Nonempty source/event/availability/value records required")
    frame = records[list(required)].copy()
    for col in ("event_time", "available_at"):
        # Do not silently assume the timezone of a supplied naive timestamp.
        parsed = pd.to_datetime(frame[col], errors="raise", format="mixed")
        if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise ValueError("Explicit timezone-aware source timestamps required")
        frame[col] = parsed.dt.tz_convert("UTC")
    if (frame.source.isna().any() or not frame.source.map(lambda s: isinstance(s, str) and bool(s)).all()
            or frame.duplicated(["source", "event_time"]).any()
            or frame[["event_time", "available_at", "value"]].isna().any().any()
            or not np.isfinite(pd.to_numeric(frame.value, errors="raise")).all()
            or (frame.available_at < frame.event_time).any()):
        raise ValueError("Duplicate/invalid source grain or publication before observation")
    frame["value"] = pd.to_numeric(frame.value)
    for code, group in frame.groupby("source"):
        v = group.value
        if ((code == "FEAR_GREED" and not v.between(0, 100).all())
                or (code.startswith("FLOW_") and not v.between(0, 1).all())
                or (code.startswith("FUNDING_") and not v.between(-.5, .5).all())
                or (code in MACRO and code != "DGS10" and (v <= 0).any())):
            raise ValueError("Signal value outside source domain")
    return frame.sort_values(["source", "event_time"]).reset_index(drop=True)


def asof_signal(observations: pd.DataFrame, decisions: pd.DatetimeIndex,
                max_age: pd.Timedelta) -> pd.DataFrame:
    """Join precomputed source features to decisions; stale publications expire."""
    if (not isinstance(decisions, pd.DatetimeIndex) or decisions.tz is None
            or not decisions.is_unique or not decisions.is_monotonic_increasing
            or max_age <= pd.Timedelta(0)):
        raise ValueError("Ordered timezone-aware decisions and positive expiry required")
    if (not observations.event_time.is_unique
            or (observations.available_at < observations.event_time).any()):
        raise ValueError("Unique valid observation times required")
    # Pandas 3 can infer microseconds for one side and nanoseconds for the
    # other. merge_asof requires identical units as well as timezones. Keep
    # nanosecond precision rather than rounding actual settlement jitter.
    observations = observations.copy()
    for column in ("event_time", "available_at"):
        observations[column] = observations[column].dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
    merge_decisions = decisions.tz_convert("UTC").astype("datetime64[ns, UTC]")
    # Several dated observations can become available in the same batch (for
    # example, modeled release times crossing a federal holiday). At that
    # instant the newest observation is known; retain its precomputed features.
    observations = observations.sort_values(["available_at", "event_time"]).drop_duplicates(
        "available_at", keep="last")
    joined = pd.merge_asof(pd.DataFrame({"decision_at": merge_decisions}),
                          observations,
                          left_on="decision_at", right_on="available_at",
                          direction="backward", tolerance=max_age)
    if (joined.available_at > joined.decision_at).any():
        raise AssertionError("Future publication reached a trading decision")
    joined.index = decisions
    return joined


def external_features(records: pd.DataFrame, daily_index: pd.DatetimeIndex, *,
                      extra_delay=pd.Timedelta(0)) -> tuple[dict[str, pd.DataFrame], dict]:
    """Features known at each completed day's following open (caller lags once)."""
    if extra_delay < pd.Timedelta(0):
        raise ValueError("A publication delay cannot be negative")
    records = validate_signals(records)
    decisions = daily_index + pd.Timedelta("1D")
    groups = {name: pd.DataFrame(index=daily_index) for name in ("macro", "sentiment", "crypto")}
    profile = {}
    for code, frame in records.groupby("source", sort=True):
        frame = frame.sort_values("event_time").reset_index(drop=True)
        v = frame.value
        features = frame[["event_time", "available_at"]].copy()
        features["available_at"] += extra_delay
        if code in MACRO:
            group, expiry = "macro", pd.Timedelta("21D" if code == "DCOILWTICO" else "10D")
            if code == "DGS10":
                features[f"{code}_level"] = v / 100
                features[f"{code}_change5"] = v.diff(5) / 100
            else:
                features[f"{code}_r5"] = v.pct_change(5, fill_method=None)
                features[f"{code}_r20"] = v.pct_change(20, fill_method=None)
                if code == "VIXCLS":
                    features[f"{code}_level"] = v / 100
        elif code == "FEAR_GREED":
            group, expiry = "sentiment", pd.Timedelta("3D")
            features["FEAR_GREED_level"] = (v - 50) / 50
            features["FEAR_GREED_change5"] = v.diff(5) / 100
        elif code.startswith("FLOW_") or code.startswith("FUNDING_"):
            group, expiry = "crypto", pd.Timedelta("2D")
            values = 2 * v - 1 if code.startswith("FLOW_") else v
            features[f"{code}_level"] = values
            features[f"{code}_mean5"] = values.rolling(5, min_periods=5).mean()
            features[f"{code}_mean20"] = values.rolling(20, min_periods=20).mean()
        else:
            raise ValueError(f"Unknown external source {code}")
        joined = asof_signal(features, decisions, expiry)
        names = features.columns.drop(["event_time", "available_at"])
        available = pd.DataFrame(joined[names].to_numpy(), columns=names, index=daily_index)
        lagged = distributed_lags(available)
        groups[group] = pd.concat([groups[group], lagged], axis=1)
        profile[code] = {"observations": len(frame), "decisions": len(joined),
                         "shared_availability_observations": int(frame.available_at.duplicated(keep=False).sum()),
                         "available_decisions": int(joined.available_at.notna().sum()),
                         "complete_base_feature_decisions": int(available.notna().all(axis=1).sum()),
                         "complete_feature_decisions": int(lagged.notna().all(axis=1).sum()),
                         "lag_windows_days": LAG_WINDOWS,
                         "feature_count": len(lagged.columns),
                         "max_publication_age_days": float((joined.decision_at - joined.available_at)
                                                              .dt.total_seconds().max() / 86400)}
    return groups, profile
