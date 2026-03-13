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
    Per-reading boolean: True where wind speed (and optionally temperature)
    are within thresholds.  Direction consistency is NOT checked here —
    it is evaluated once per day in compute_daily_target so that the check
    works regardless of data resolution (hourly, 4-hourly, or 5-minute).
    """
    sc = cfg["sailing"]
    good = df["wind_speed"].between(sc["wind_speed_min"], sc["wind_speed_max"])

    min_temp = sc.get("min_temperature")
    if min_temp is not None and "temperature" in df.columns:
        good = good & (df["temperature"] >= min_temp)

    return good


def compute_daily_target(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    For each calendar date in df, return the fraction of the sailing window
    (window_start–window_end) where conditions are good.

    Speed and temperature are checked per reading.  Direction consistency is
    checked once per day using all in-range readings within the sailing window,
    so the check works correctly regardless of data resolution (hourly, 4 h,
    5 min).  Days with fewer than 3 in-range direction readings are skipped.

    Returns a pd.Series indexed by datetime.date.
    """
    sc = cfg["sailing"]

    sailing = df.between_time(sc["window_start"], sc["window_end"])
    speed_temp_ok = _is_good_instant(sailing, cfg)

    results: dict = {}
    for date, day_df in sailing.groupby(sailing.index.date):
        day_ok = speed_temp_ok.reindex(day_df.index).fillna(False)
        frac = float(day_ok.mean())

        # Direction consistency: circular std over speed-qualified readings only.
        # Using speed-qualified readings avoids calm-period noise (many stations
        # report direction = 0° when wind is calm, which is not a real direction).
        valid_dirs = day_df.loc[day_ok, "wind_direction"].dropna()
        if len(valid_dirs) < 3:
            # Not enough in-range readings to assess direction — mark as poor.
            results[date] = 0.0
            continue

        if _circular_std(valid_dirs) > sc["wind_dir_consistency_max"]:
            results[date] = 0.0  # direction too variable
        else:
            results[date] = frac

    return pd.Series(results)


def _target_date(snap_dt: pd.Timestamp, window_end: str) -> datetime.date:
    """
    Determine which sailing window date a snapshot is predicting.
    Before the sailing window closes → same calendar day.
    After the sailing window closes  → next calendar day.
    """
    we_hour = int(window_end.split(":")[0])
    if snap_dt.hour < we_hour:
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
    past_18h = past[past.index >= snap_dt - pd.Timedelta("18h")]
    past_24h = past[past.index >= snap_dt - pd.Timedelta("24h")]

    # Point counts per window — used only to gate unreliable features.
    # NOTE: do NOT include data density as a model feature; it correlates
    # with how recently data was collected (recent = 5-min resolution = dense),
    # which would teach the model to predict based on data age, not weather.
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
        # Time context
        "snapshot_hour": snap_dt.hour,
        "day_of_week":   snap_dt.dayofweek,
        "sin_doy":       np.sin(2 * np.pi * snap_dt.dayofyear / 365),
        "cos_doy":       np.cos(2 * np.pi * snap_dt.dayofyear / 365),
        # Anomalies: departure from 28-day trailing mean.
        # Captures synoptic signals (fronts, pressure systems).
        # Wind-speed anomaly removed: thermal wind hasn't formed at the morning
        # snapshot, so wind_speed ≈ 0 on BOTH good and bad days — the anomaly
        # is near-zero noise that actively misleads the model.
        "temperature_anomaly": _anomaly(df["temperature"], snap_dt),
        "humidity_anomaly":    _anomaly(df["humidity"], snap_dt),
        "pressure_anomaly":    _anomaly(df["pressure_relative"], snap_dt),
        # Wind direction as unit-circle components (no seasonal baseline needed)
        "wind_dir_sin": np.sin(np.radians(wind_dir)),
        "wind_dir_cos": np.cos(np.radians(wind_dir)),
        # Short-term trends — reliable even with 2 points
        "pressure_trend_3h":  _trend(past_3h["pressure_relative"]),
        "temp_trend_3h":      _trend(past_3h["temperature"]),
        "pressure_trend_6h":  _trend(past_6h["pressure_relative"]),
        "pressure_trend_12h": _trend(past_12h["pressure_relative"]),
        # Rolling means and maxima.
        # At a 06:00 morning snapshot, past_18h reaches back to yesterday 12:00,
        # which covers the peak thermal window (≈10:00–16:00) of the previous day.
        # wind_speed_max_18h / _24h therefore capture "did yesterday have thermals?"
        # — the single strongest predictor of today's thermal wind at a lake site.
        "wind_speed_mean_3h":  past_3h["wind_speed"].mean(),
        "wind_speed_mean_6h":  past_6h["wind_speed"].mean(),
        "wind_speed_mean_12h": past_12h["wind_speed"].mean(),
        "wind_speed_max_3h":   past_3h["wind_speed"].max(),
        "wind_speed_max_18h":  past_18h["wind_speed"].max(),
        "wind_speed_max_24h":  past_24h["wind_speed"].max(),
        # Variability/consistency stats — gated: a single reading gives a
        # spuriously perfect score (std=0, circ_std=0) which would bias the model
        "wind_speed_std_3h":       _guarded_std(past_3h["wind_speed"], MIN_3H),
        "wind_dir_consistency_3h": _guarded_circ_std(past_3h["wind_direction"], MIN_3H),
        "wind_dir_consistency_6h": _guarded_circ_std(past_6h["wind_direction"], MIN_6H),
        "wind_dir_consistency_12h": _guarded_circ_std(past_12h["wind_direction"], MIN_12H),
        # ── Absolute levels ───────────────────────────────────────────────────
        # The 28-day anomaly features above capture synoptic departures (fronts,
        # pressure systems) but systematically neutralise the seasonal signal:
        # a sustained June heatwave has near-zero temperature *anomaly* yet
        # maximum thermal-wind potential.  Keep both anomaly (for weather-system
        # signals) and absolute level (for the seasonal/thermal baseline).
        "temperature_mean_24h": float(past_24h["temperature"].mean()),
        "temperature_max_24h":  float(past_24h["temperature"].max()),
        "pressure_mean_24h":    float(past_24h["pressure_relative"].mean()),
        # ── Lake–air temperature gradient ─────────────────────────────────────
        # The primary driver of thermal lake wind: cold lake + hot land → strong
        # convective flow toward the lake.  water_temperature changes on a
        # timescale of days–weeks, so it encodes the accumulated lake heat budget
        # and distinguishes early-season (cold lake, big gradient) from late-season
        # (warm lake, reduced gradient) conditions.
        # At the morning snapshot (06:00) air is cooler than the daytime peak but
        # the lake temperature is stable — the difference is a reliable proxy for
        # expected afternoon thermal strength.
        "water_temperature": (
            float(current["water_temperature"])
            if "water_temperature" in df.columns
            and pd.notna(current.get("water_temperature"))
            else np.nan
        ),
        "air_water_temp_diff": (
            float(current["temperature"] - current["water_temperature"])
            if "water_temperature" in df.columns
            and pd.notna(current.get("temperature"))
            and pd.notna(current.get("water_temperature"))
            else np.nan
        ),
        # ── Diurnal heating cycle ─────────────────────────────────────────────
        # How large was the day/night temperature swing in the last 24h?
        # A big swing → clear skies → strong daytime heating → good thermals.
        "diurnal_temp_range_24h": (
            float(past_24h["temperature"].max() - past_24h["temperature"].min())
            if n_24h >= 4 else np.nan
        ),
        # How far into the heating cycle are we at snapshot time?
        # Zero at the overnight minimum (morning snapshot), peaks at ~14:00.
        "temp_above_daily_min": (
            float(current["temperature"] - past_24h["temperature"].min())
            if n_24h >= 4 and pd.notna(current.get("temperature")) else np.nan
        ),
        # Direct solar radiation — instantaneous heating rate.
        # Zero at night is a valid and meaningful value, not missing data.
        "solar_mean_3h": (
            float(past_3h["solar"].mean())
            if "solar" in df.columns and n_3h >= MIN_3H else np.nan
        ),
        # Dew-point depression: larger = drier air = clearer sky = more solar heating.
        "dew_point_depression": (
            float(current["temperature"] - current["dew_point"])
            if "dew_point" in df.columns
            and pd.notna(current.get("temperature"))
            and pd.notna(current.get("dew_point"))
            else np.nan
        ),
        # Low overnight humidity → clear skies overnight → strong daytime heating.
        "humidity_min_24h": (
            float(past_24h["humidity"].min()) if n_24h >= 4 else np.nan
        ),
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
    Snapshots before window_end → predict same day (incl. live in-window data).
    Snapshots at/after window_end → predict next day.

    Snapshot times come from ``cfg["prediction"]["snapshots"]`` if present,
    otherwise a built-in default set covering pre-window, in-window, and
    post-window hours is used (matching the GitHub Actions schedule).

    Returns (X, y) where y is binary (1 = good day).
    """
    # Default snapshot hours mirror the GitHub Actions cron schedule (CET):
    # 05:00, 07:00  pre-window | 10:00, 13:00  in-window | 18:00, 22:00  post-window
    _DEFAULT_SNAPSHOTS = ["05:00", "07:00", "10:00", "13:00", "18:00", "22:00"]

    daily_quality = compute_daily_target(df, cfg)
    if len(daily_quality) < 2:
        return pd.DataFrame(), pd.Series(dtype=int)

    min_good = cfg["prediction"]["min_good_fraction"]
    window_end = cfg["sailing"]["window_end"]
    snapshots = cfg["prediction"].get("snapshots", _DEFAULT_SNAPSHOTS)

    rows: list[dict] = []
    for snap_date in daily_quality.index:
        for snap_str in snapshots:
            h, m = map(int, snap_str.split(":"))
            snap_dt = pd.Timestamp(
                year=snap_date.year, month=snap_date.month, day=snap_date.day,
                hour=h, minute=m,
            )
            tgt_date = _target_date(snap_dt, window_end)

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
