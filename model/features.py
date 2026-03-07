"""
model/features.py — Feature extraction and target computation.

All functions accept a DataFrame indexed by timestamp (as produced by stitcher.py)
and a config dict (as loaded from config.toml). Time-based rolling is used
throughout so varying data granularity is handled correctly.
"""

import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _circular_std(angles: pd.Series) -> float:
    """Circular standard deviation of wind directions (degrees)."""
    clean = angles.dropna()
    if clean.empty:
        return np.nan
    rad = np.radians(clean)
    R = np.hypot(np.sin(rad).mean(), np.cos(rad).mean())
    return float(np.degrees(np.sqrt(-2 * np.log(np.clip(R, 1e-9, 1)))))


def _circular_range(angles: pd.Series) -> float:
    """Range of circular data (degrees) — max angular deviation from mean."""
    clean = angles.dropna()
    if len(clean) < 2:
        return 0.0
    rad = np.radians(clean.to_numpy())
    mean_angle = np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())
    diff = np.arctan2(np.sin(rad - mean_angle), np.cos(rad - mean_angle))
    return float(np.degrees(diff.max() - diff.min()))


def _trend(series: pd.Series) -> float:
    """Change between last and first non-NaN value in the series."""
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan
    return float(clean.iloc[-1] - clean.iloc[0])


# How far back to look when computing the seasonal baseline.
# 28 days smooths synoptic weather cycles while tracking the seasonal mean.
_ANOMALY_WINDOW = pd.Timedelta("28d")


def _anomaly(series: pd.Series, snap_dt: pd.Timestamp) -> float:
    """
    Current value minus trailing 28-day mean — removes the seasonal baseline
    so the model sees departures from climatology rather than absolute levels.
    Returns NaN if fewer than 3 days of baseline data exist.
    """
    baseline_slice = series[snap_dt - _ANOMALY_WINDOW : snap_dt]
    if baseline_slice.dropna().shape[0] < 3:
        return np.nan
    current = series[:snap_dt].dropna()
    if current.empty:
        return np.nan
    return float(current.iloc[-1] - baseline_slice.mean())


# ---------------------------------------------------------------------------
# Target computation
# ---------------------------------------------------------------------------

def _is_good_instant(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Boolean Series: True where instantaneous conditions are within thresholds.
    Applies the wind-direction consistency check over a rolling time window.
    """
    sc = cfg["sailing"]
    window = f"{sc['consistency_window_hours']}h"

    wind_dir_range = (
        df["wind_direction"]
        .rolling(window, min_periods=2)
        .apply(_circular_range, raw=False)
    )

    return (
        df["wind_speed"].between(sc["wind_speed_min"], sc["wind_speed_max"])
        & (wind_dir_range <= sc["wind_dir_consistency_max"])
    )


def compute_daily_target(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    For each calendar date in df, return the fraction of the sailing window
    (window_start–window_end) where conditions are good.

    Returns a pd.Series indexed by datetime.date.
    """
    sc = cfg["sailing"]
    good = _is_good_instant(df, cfg)
    sailing_good = good.between_time(sc["window_start"], sc["window_end"])
    return sailing_good.groupby(sailing_good.index.date).mean()


def _target_date(snap_dt: pd.Timestamp, window_start: str) -> datetime.date:
    """
    Determine which sailing window date a snapshot is predicting.
    Before the sailing window opens → same calendar day.
    After the sailing window opens  → next calendar day.
    """
    ws_hour = int(window_start.split(":")[0])
    if snap_dt.hour < ws_hour:
        return snap_dt.date()
    return snap_dt.date() + datetime.timedelta(days=1)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_snapshot_features(
    df: pd.DataFrame,
    snap_dt: pd.Timestamp,
) -> Optional[dict]:
    """
    Extract a feature vector from df at a specific snapshot datetime.

    Uses only data up to (and including) snap_dt, so there is no leakage.
    Returns None if there is insufficient data.
    """
    past = df[df.index <= snap_dt]
    if len(past) < 3:
        return None

    current = past.iloc[-1]
    past_3h  = past[past.index >= snap_dt - pd.Timedelta("3h")]
    past_6h  = past[past.index >= snap_dt - pd.Timedelta("6h")]
    past_12h = past[past.index >= snap_dt - pd.Timedelta("12h")]
    past_24h = past[past.index >= snap_dt - pd.Timedelta("24h")]

    # Point counts per window — used both to gate unreliable features and as
    # an explicit feature so the model can learn to trust dense windows more.
    n_3h  = past_3h["wind_direction"].notna().sum()
    n_6h  = past_6h["wind_direction"].notna().sum()
    n_12h = past_12h["wind_direction"].notna().sum()
    n_24h = past_24h["wind_direction"].notna().sum()

    # Minimum points needed before a rolling stat is considered reliable.
    # Below this threshold we return NaN so the value gets median-imputed
    # rather than polluting the model with a single-point artefact.
    # At 5-min resolution a 3h window holds ~36 points; we require at least 3
    # so that even hourly data passes, but a single stray reading does not.
    MIN_3H  = 3
    MIN_6H  = 3
    MIN_12H = 4

    def _guarded_circ_std(series: pd.Series, min_pts: int) -> float:
        return _circular_std(series) if series.notna().sum() >= min_pts else np.nan

    def _guarded_std(series: pd.Series, min_pts: int) -> float:
        return float(series.std()) if series.notna().sum() >= min_pts else np.nan

    wind_dir = current.get("wind_direction", np.nan)

    return {
        # Time context — month dropped; sin/cos_doy kept as a weak seasonal
        # anchor (e.g. nobody sails in January regardless of wind conditions)
        "snapshot_hour": snap_dt.hour,
        "day_of_week":   snap_dt.dayofweek,
        "sin_doy":       np.sin(2 * np.pi * snap_dt.dayofyear / 365),
        "cos_doy":       np.cos(2 * np.pi * snap_dt.dayofyear / 365),
        # Data density: points per hour in the last 24h, normalised so that
        # 5-min resolution (12 pts/hr) → 1.0 and hourly → ~0.083.
        # Lets the model discount features computed from sparse windows.
        "data_density": n_24h / 24 / 12,
        # Anomalies: current value minus 28-day trailing mean.
        # These capture synoptic weather signals (fronts, pressure systems)
        # rather than the seasonal baseline.
        "temperature_anomaly":   _anomaly(df["temperature"], snap_dt),
        "humidity_anomaly":      _anomaly(df["humidity"], snap_dt),
        "pressure_anomaly":      _anomaly(df["pressure_relative"], snap_dt),
        "wind_speed_anomaly":    _anomaly(df["wind_speed"], snap_dt),
        "wind_gust_anomaly":     _anomaly(df["wind_gust"], snap_dt),
        # Wind direction as unit-circle components (no seasonal baseline needed)
        "wind_dir_sin": np.sin(np.radians(wind_dir)),
        "wind_dir_cos": np.cos(np.radians(wind_dir)),
        # Short-term trends — reliable even with 2 points
        "pressure_trend_3h":  _trend(past_3h["pressure_relative"]),
        "temp_trend_3h":      _trend(past_3h["temperature"]),
        "pressure_trend_6h":  _trend(past_6h["pressure_relative"]),
        "pressure_trend_12h": _trend(past_12h["pressure_relative"]),
        # Rolling means — unbiased for any n≥1, so no guard needed
        "wind_speed_mean_3h":  past_3h["wind_speed"].mean(),
        "wind_speed_mean_6h":  past_6h["wind_speed"].mean(),
        "wind_speed_mean_12h": past_12h["wind_speed"].mean(),
        "wind_speed_max_3h":   past_3h["wind_speed"].max(),
        # Variability/consistency stats — gated: a single reading gives a
        # spuriously perfect score (std=0, circ_std=0) which would bias the model
        "wind_speed_std_3h":       _guarded_std(past_3h["wind_speed"], MIN_3H),
        "wind_dir_consistency_3h": _guarded_circ_std(past_3h["wind_direction"], MIN_3H),
        "wind_dir_consistency_6h": _guarded_circ_std(past_6h["wind_direction"], MIN_6H),
        "wind_dir_consistency_12h": _guarded_circ_std(past_12h["wind_direction"], MIN_12H),
    }


# ---------------------------------------------------------------------------
# Training pair construction
# ---------------------------------------------------------------------------

def build_training_pairs(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) training pairs from historical data.

    For each (date, snapshot_time), extract features at that snapshot and
    pair them with the sailing quality of the corresponding target date.
    Snapshots before the sailing window → predict same day.
    Snapshots after  the sailing window → predict next day.

    Returns (X, y) where y is binary (1 = good day).
    """
    daily_quality = compute_daily_target(df, cfg)
    if len(daily_quality) < 2:
        return pd.DataFrame(), pd.Series(dtype=int)

    min_good = cfg["prediction"]["min_good_fraction"]
    window_start = cfg["sailing"]["window_start"]

    rows: list[dict] = []
    for snap_date in daily_quality.index:
        for snap_str in cfg["prediction"]["snapshots"]:
            h, m = map(int, snap_str.split(":"))
            snap_dt = pd.Timestamp(
                year=snap_date.year, month=snap_date.month, day=snap_date.day,
                hour=h, minute=m,
            )
            tgt_date = _target_date(snap_dt, window_start)

            if tgt_date not in daily_quality.index:
                continue

            features = extract_snapshot_features(df, snap_dt)
            if features is None:
                continue

            features["_target"] = int(daily_quality[tgt_date] >= min_good)
            rows.append(features)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int)

    Xdf = pd.DataFrame(rows)
    y = Xdf.pop("_target")
    return Xdf, y
