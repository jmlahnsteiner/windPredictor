"""
input/open_meteo_historical.py — Fetch ERA5 historical NWP from Open-Meteo archive API.

Fetches the same variables as open_meteo.py (fetch_forecast) so historical NWP
and live NWP are on identical scales and column names.

Usage:
    python input/open_meteo_historical.py --start 2024-01-01 --end 2026-04-05
"""

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the variable list and column renaming from the forecast module.
from input.open_meteo import _HOURLY_VARS, _COL_NAMES

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_CHUNK_DAYS = 90


def fetch_historical_chunk(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetch hourly ERA5 data for a single date range (<=90 days recommended).

    Returns a tz-aware DatetimeIndex DataFrame with the same column names as
    fetch_forecast() — empty DataFrame on failure.
    """
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          ",".join(_HOURLY_VARS),
        "wind_speed_unit": "kn",
        "start_date":      start_date.isoformat(),
        "end_date":        end_date.isoformat(),
        "timezone":        "auto",
    }

    try:
        resp = requests.get(_ARCHIVE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [!] Open-Meteo archive fetch failed ({start_date}–{end_date}): {exc}")
        return pd.DataFrame()

    hourly = data.get("hourly", {})
    times = pd.to_datetime(hourly.pop("time", []))

    # Use the IANA timezone name from the response (e.g. "Europe/Vienna").
    # This correctly handles DST transitions — unlike using the fixed
    # utc_offset_seconds scalar which breaks when the range spans a DST change.
    tz_name = data.get("timezone", "UTC")
    try:
        times = times.tz_localize(tz_name)
    except Exception:
        times = times.tz_localize("UTC")

    df = pd.DataFrame(hourly, index=times)
    df.index.name = "timestamp"
    df = df.rename(columns=_COL_NAMES)
    return df


def fetch_historical_range(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetch ERA5 data for an arbitrary date range, chunked into <=90-day requests.

    Returns a combined tz-aware DataFrame, or empty DataFrame on complete failure.
    """
    chunks = []
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=_CHUNK_DAYS - 1), end_date)
        print(f"  Fetching NWP history: {current} → {chunk_end} …", flush=True)
        chunk = fetch_historical_chunk(lat, lon, current, chunk_end)
        if not chunk.empty:
            chunks.append(chunk)
        current = chunk_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill NWP readings from ERA5")
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end",   required=True, metavar="YYYY-MM-DD")
    args = parser.parse_args()

    from utils.config import load_config
    _HERE = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(_HERE, "..", "config.toml"))

    loc = cfg.get("location", {})
    lat, lon = loc.get("lat"), loc.get("lon")
    if lat is None or lon is None:
        print("ERROR: [location] lat/lon not set in config.toml")
        sys.exit(1)

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    print(f"Fetching ERA5 NWP for {lat},{lon}  {start} → {end}")

    df = fetch_historical_range(lat, lon, start, end)
    if df.empty:
        print("ERROR: no data returned")
        sys.exit(1)

    from input.nwp_store import upsert_nwp_readings
    n = upsert_nwp_readings(df)
    print(f"Upserted {n} NWP rows to nwp_readings")
