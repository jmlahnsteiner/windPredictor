"""
input/open_meteo.py — Fetch NWP forecast data from Open-Meteo (no API key needed).

Provides hourly forecast variables relevant to thermal / sea-breeze wind:
temperature, wind speed/direction/gusts, cloud cover, boundary layer height,
and direct solar radiation.

API reference: https://open-meteo.com/en/docs
"""

import datetime
import math

import pandas as pd
import requests

_BASE_URL = "https://api.open-meteo.com/v1/forecast"

_HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "boundary_layer_height",
    "direct_radiation",
]

# Friendly column names for the returned DataFrame
_COL_NAMES = {
    "temperature_2m":        "temperature",
    "wind_speed_10m":        "wind_speed",
    "wind_direction_10m":    "wind_direction",
    "wind_gusts_10m":        "wind_gust",
    "cloud_cover":           "cloud_cover",
    "boundary_layer_height": "blh",
    "direct_radiation":      "direct_radiation",
}


def fetch_forecast(lat: float, lon: float, forecast_days: int = 7) -> pd.DataFrame:
    """
    Fetch hourly NWP forecast from Open-Meteo for the given location.

    Returns a DataFrame indexed by local timestamp (tz-aware, matching the
    station timezone via timezone=auto).  Returns an empty DataFrame on failure.
    """
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          ",".join(_HOURLY_VARS),
        "wind_speed_unit": "kn",
        "forecast_days":   forecast_days,
        "timezone":        "auto",
    }

    try:
        resp = requests.get(_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(f"  [!] Open-Meteo fetch failed: {exc}")
        return pd.DataFrame()

    hourly = data.get("hourly", {})
    times = pd.to_datetime(hourly.pop("time", []))
    df = pd.DataFrame(hourly, index=times)
    df.index.name = "timestamp"
    df = df.rename(columns=_COL_NAMES)
    return df


def sailing_window_stats(
    nwp_df: pd.DataFrame,
    target_date: datetime.date,
    window_start: str,
    window_end: str,
) -> dict:
    """
    Extract NWP statistics for the sailing window on target_date.

    Returns a dict of aggregated values, or {} when data is unavailable.
    """
    if nwp_df.empty:
        return {}

    # Filter to target date then to sailing window hours.
    # Use date comparison on the index to avoid tz issues.
    mask = pd.Series(nwp_df.index).apply(lambda t: t.date() == target_date).values
    day_df = nwp_df.iloc[mask]
    if day_df.empty:
        return {}

    # between_time works on the DatetimeIndex (tz-aware is fine).
    window = day_df.between_time(window_start, window_end)
    if len(window) < 2:
        return {}

    ws  = window["wind_speed"].dropna()
    wg  = window["wind_gust"].dropna()
    wd  = window["wind_direction"].dropna()
    cc  = window["cloud_cover"].dropna()
    blh = window["blh"].dropna()

    # Circular mean wind direction
    mean_dir: float | None = None
    if not wd.empty:
        rad = [math.radians(d) for d in wd]
        sin_m = sum(math.sin(r) for r in rad) / len(rad)
        cos_m = sum(math.cos(r) for r in rad) / len(rad)
        mean_dir = round(math.degrees(math.atan2(sin_m, cos_m)) % 360)

    return {
        "mean_wind_kn":    round(float(ws.mean()), 1) if not ws.empty else None,
        "max_gust_kn":     round(float(wg.max()),  1) if not wg.empty else None,
        "cloud_cover_pct": round(float(cc.mean()))    if not cc.empty else None,
        "blh_m":           round(float(blh.mean()))   if not blh.empty else None,
        "mean_dir_deg":    mean_dir,
        "source":          "open-meteo",
    }
