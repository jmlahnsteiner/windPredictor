"""
model/features_sequence.py — Raw time-series sequences for the GRU model.

Produces:
  build_sequence(df, snap_dt)               → (T=24, 7) float32 array
  build_nwp_context(nwp_df, snap_dt, cfg)   → (12,) float32 array
  build_sequence_training_pairs(df, cfg, nwp_df=None) → (seqs, contexts, labels)
"""

import datetime
from typing import Optional

import numpy as np
import pandas as pd

# Station features per timestep (7 channels)
_SEQ_COLS = [
    "wind_speed",
    "wind_dir_sin",   # derived from wind_direction
    "wind_dir_cos",
    "temperature",
    "pressure_relative",
    "humidity",
    "solar",
]
SEQ_FEATURES = len(_SEQ_COLS)    # 7
CONTEXT_FEATURES = 12             # 9 NWP + 3 time


def build_sequence(
    df: pd.DataFrame,
    snap_dt: pd.Timestamp,
    hours: int = 24,
) -> np.ndarray:
    """
    Extract the last `hours` of hourly-resampled station data before snap_dt.

    Returns float32 array of shape (hours, SEQ_FEATURES).
    Missing timesteps and NaN values are zero-filled.
    """
    source_cols = [c for c in ["wind_speed", "wind_direction",
                                "temperature", "pressure_relative",
                                "humidity", "solar"]
                   if c in df.columns]
    resampled = df[source_cols].resample("1h").mean()

    snap_floor = snap_dt.floor("h")
    start_dt = snap_floor - pd.Timedelta(hours=hours - 1)
    window = resampled.loc[start_dt:snap_floor]

    # Dense hourly index — ensures shape is always (hours, F)
    full_idx = pd.date_range(start=start_dt, end=snap_floor, freq="1h")
    window = window.reindex(full_idx)

    wd = window["wind_direction"].fillna(0.0) if "wind_direction" in window.columns else pd.Series(0.0, index=full_idx)
    rad = np.radians(wd.to_numpy())
    sin_wd = np.sin(rad)
    cos_wd = np.cos(rad)

    def _col(name):
        if name in window.columns:
            return window[name].fillna(0.0).to_numpy()
        return np.zeros(len(window))

    arr = np.column_stack([
        _col("wind_speed"),
        sin_wd,
        cos_wd,
        _col("temperature"),
        _col("pressure_relative"),
        _col("humidity"),
        _col("solar"),
    ])

    return arr.astype(np.float32)


def _extract_nwp_window_stats(
    nwp_df: pd.DataFrame,
    target_date: datetime.date,
    cfg: dict,
) -> np.ndarray:
    """
    Extract the 9 NWP sailing-window features for target_date as a float32 array.
    Returns zeros (not NaN) — GRU context vector must always be finite.
    """
    from model.features import _circular_std, _NWP_KEYS

    sc = cfg["sailing"]
    mask = np.array([t.date() == target_date for t in nwp_df.index])
    day_nwp = nwp_df.iloc[mask]

    result = np.zeros(len(_NWP_KEYS), dtype=np.float32)
    if day_nwp.empty:
        return result

    window_nwp = day_nwp.between_time(sc["window_start"], sc["window_end"])
    if len(window_nwp) < 1:
        return result

    ws  = window_nwp["wind_speed"].dropna()
    wg  = window_nwp["wind_gust"].dropna()
    wd  = window_nwp["wind_direction"].dropna()
    cc  = window_nwp["cloud_cover"].dropna()
    blh = window_nwp["blh"].dropna()
    dr  = window_nwp["direct_radiation"].dropna()

    vals = {}
    if not ws.empty:
        vals["nwp_wind_speed_mean"] = float(ws.mean())
        vals["nwp_wind_speed_max"]  = float(ws.max())
    if not wg.empty:
        vals["nwp_wind_gust_max"] = float(wg.max())
    if not wd.empty:
        rad = np.radians(wd.to_numpy())
        sin_m = float(np.sin(rad).mean())
        cos_m = float(np.cos(rad).mean())
        mag = max(float(np.hypot(sin_m, cos_m)), 1e-9)
        vals["nwp_wind_dir_sin"] = sin_m / mag
        vals["nwp_wind_dir_cos"] = cos_m / mag
        vals["nwp_dir_consistency"] = float(_circular_std(wd)) if len(wd) >= 2 else 0.0
    if not cc.empty:
        vals["nwp_cloud_cover_mean"] = float(cc.mean())
    if not blh.empty:
        vals["nwp_blh_mean"] = float(blh.mean())
    if not dr.empty:
        vals["nwp_direct_radiation_mean"] = float(dr.mean())

    for i, k in enumerate(_NWP_KEYS):
        result[i] = vals.get(k, 0.0)
    return result


def build_nwp_context(
    nwp_df: Optional[pd.DataFrame],
    snap_dt: pd.Timestamp,
    cfg: dict,
) -> np.ndarray:
    """
    Build a (CONTEXT_FEATURES,) float32 context vector for the GRU.

    Components:
      - 9 NWP sailing-window features for the target day (zeros if unavailable)
      - 3 time features: sin_doy, cos_doy, snapshot_hour/23
    """
    from model.features import _target_date

    nwp_vals = np.zeros(9, dtype=np.float32)

    if nwp_df is not None and not nwp_df.empty:
        sc = cfg["sailing"]
        target_date = _target_date(snap_dt, sc["window_end"])
        nwp_vals = _extract_nwp_window_stats(nwp_df, target_date, cfg)

    time_ctx = np.array([
        np.sin(2 * np.pi * snap_dt.dayofyear / 365),
        np.cos(2 * np.pi * snap_dt.dayofyear / 365),
        snap_dt.hour / 23.0,
    ], dtype=np.float32)

    return np.concatenate([nwp_vals, time_ctx])


def build_sequence_training_pairs(
    df: pd.DataFrame,
    cfg: dict,
    nwp_df: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (sequences, contexts, labels) arrays for GRU training.

    Returns:
      sequences : (N, 24, SEQ_FEATURES) float32
      contexts  : (N, CONTEXT_FEATURES) float32
      labels    : (N,) int32 (0 or 1)
    """
    from model.features import compute_daily_target, _target_date

    _DEFAULT_SNAPSHOTS = ["05:00", "07:00", "10:00", "13:00", "18:00", "22:00"]

    daily_quality = compute_daily_target(df, cfg)
    if len(daily_quality) < 2:
        empty = np.zeros((0, 24, SEQ_FEATURES), dtype=np.float32)
        return empty, np.zeros((0, CONTEXT_FEATURES), dtype=np.float32), np.zeros(0, dtype=np.int32)

    min_good   = cfg["prediction"]["min_good_fraction"]
    window_end = cfg["sailing"]["window_end"]
    snapshots  = cfg["prediction"].get("snapshots", _DEFAULT_SNAPSHOTS)

    seqs, ctxs, labs = [], [], []

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

            seq = build_sequence(df, snap_dt)
            ctx = build_nwp_context(nwp_df, snap_dt, cfg)
            label = int(daily_quality[tgt_date] >= min_good)

            seqs.append(seq)
            ctxs.append(ctx)
            labs.append(label)

    if not seqs:
        empty = np.zeros((0, 24, SEQ_FEATURES), dtype=np.float32)
        return empty, np.zeros((0, CONTEXT_FEATURES), dtype=np.float32), np.zeros(0, dtype=np.int32)

    T = 24
    padded = np.zeros((len(seqs), T, SEQ_FEATURES), dtype=np.float32)
    for i, s in enumerate(seqs):
        t = min(s.shape[0], T)
        padded[i, -t:, :] = s[-t:]   # right-align: most recent hour is last

    return padded, np.array(ctxs, dtype=np.float32), np.array(labs, dtype=np.int32)
