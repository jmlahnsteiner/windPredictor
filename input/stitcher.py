"""
stitcher.py — Parse daily Ecowitt xlsx files and upsert rows into weather_readings DB.

Re-running is safe: existing rows are replaced (upsert by timestamp).

Usage:
    python stitcher.py                # defaults
    python stitcher.py downloaded_files/  # explicit input dir
"""

import glob
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Column mapping: flattened xlsx header → clean name
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "Time": "timestamp",
    "Außen_Temperature(℃)": "temperature",
    "Außen_Feels Like(℃)": "feels_like",
    "Außen_Dew Point(℃)": "dew_point",
    "Außen_Humidity(%)": "humidity",
    "Solar und UVI_Solar(W/m²)": "solar",
    "Solar und UVI_UVI": "uvi",
    "Regenfall_Rain Rate(mm/hr)": "rain_rate",
    "Regenfall_Daily(mm)": "daily_rain",
    "Regenfall_Event(mm)": "event_rain",
    "Regenfall_Hourly(mm)": "hourly_rain",
    "Regenfall_Weekly(mm)": "weekly_rain",
    "Regenfall_Monthly(mm)": "monthly_rain",
    "Regenfall_Yearly(mm)": "yearly_rain",
    "Luftdruck_Relative(hPa)": "pressure_relative",
    "Luftdruck_Absolute(hPa)": "pressure_absolute",
    "Wassertemperatur_Temperature(℃)": "water_temperature",
    "Wind_Wind Speed(knots)": "wind_speed",
    "Wind_Wind Gust(knots)": "wind_gust",
    "Wind_Wind Direction(º)": "wind_direction",
}

NUMERIC_COLS = [c for c in COLUMN_MAP.values() if c != "timestamp"]

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(_HERE, "downloaded_files")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-level Excel headers into 'Level0_Level1' strings."""
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            l0, l1 = str(col[0]), str(col[1])
            l0_empty = not l0 or "Unnamed" in l0
            l1_empty = not l1 or "Unnamed" in l1
            if l0_empty:
                new_cols.append(l1)
            elif l1_empty:
                new_cols.append(l0)
            else:
                new_cols.append(f"{l0}_{l1}")
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_xlsx(path: str) -> pd.DataFrame | None:
    """
    Parse a single Ecowitt xlsx file into a clean, typed DataFrame.
    Index is a DatetimeTZNaive timestamp. Returns None on failure.
    """
    try:
        df = pd.read_excel(path, header=[0, 1])
    except Exception as e:
        print(f"  [!] Cannot read {os.path.basename(path)}: {e}")
        return None

    df = _flatten_columns(df)
    df = df.rename(columns=COLUMN_MAP)

    if "timestamp" not in df.columns:
        print(f"  [!] No timestamp column in {os.path.basename(path)}, skipping")
        return None

    df = df.assign(timestamp=pd.to_datetime(df["timestamp"], errors="coerce"))
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    df = df.assign(**{c: pd.to_numeric(df[c], errors="coerce") for c in numeric_present})

    # Keep only mapped columns so the schema stays consistent across files
    known = set(NUMERIC_COLS)
    df = df[[c for c in df.columns if c in known]]

    return df


def stitch_to_db(
    input_dir: str = DEFAULT_INPUT_DIR,
    db_path: str | None = None,
    since: "date | None" = None,
) -> int:
    """
    Parse xlsx files in input_dir and upsert rows into weather_readings.

    If `since` is given, only files whose name contains a date >= since are
    processed (filename pattern: Wetterstation_YYYY-MM-DD.xlsx).
    Re-running is safe: existing rows are replaced (upsert by timestamp).
    Returns number of rows upserted.
    """
    import sys as _sys
    from datetime import date as _date
    import re as _re
    _sys.path.insert(0, os.path.dirname(_HERE))
    from input.weather_store import upsert_readings

    all_paths = sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
    if since is not None:
        _pat = _re.compile(r"(\d{4}-\d{2}-\d{2})")
        def _keep(p: str) -> bool:
            m = _pat.search(os.path.basename(p))
            if not m:
                return True  # can't tell — include it
            try:
                return _date.fromisoformat(m.group(1)) >= since
            except ValueError:
                return True
        paths = [p for p in all_paths if _keep(p)]
        skipped = len(all_paths) - len(paths)
        if skipped:
            print(f"  (skipping {skipped} file(s) before {since})")
    else:
        paths = all_paths
    if not paths:
        print(f"No xlsx files found in {input_dir}")
        return 0

    total = 0
    for path in paths:
        df = parse_xlsx(path)
        if df is not None and not df.empty:
            n = upsert_readings(df, db_path=db_path) if db_path else upsert_readings(df)
            total += n
            print(f"  [+] {os.path.basename(path)}: {n} rows upserted")
        else:
            print(f"  [-] {os.path.basename(path)}: skipped")

    print(f"\nTotal: {total} rows upserted")
    return total


if __name__ == "__main__":
    args = sys.argv[1:]
    input_dir = args[0] if args else DEFAULT_INPUT_DIR
    print(f"Input: {input_dir}\n")
    stitch_to_db(input_dir)
