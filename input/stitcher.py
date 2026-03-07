"""
stitcher.py — Combine daily Ecowitt xlsx files into a single Parquet time series.

The output Parquet file is the canonical dataset for downstream analysis.
Re-running is safe: existing data is merged and de-duplicated.

Usage:
    python stitcher.py                                   # defaults
    python stitcher.py downloaded_files/ ../data.parquet # explicit paths
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
DEFAULT_OUTPUT = os.path.join(_HERE, "..", "data.parquet")


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


def stitch(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_path: str = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """
    Load all xlsx files from input_dir, merge with any existing Parquet,
    deduplicate, sort, and write back to output_path.

    Returns the combined DataFrame.
    """
    paths = sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
    if not paths:
        print(f"No xlsx files found in {input_dir}")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    # Carry over existing data so re-runs are purely additive
    if os.path.exists(output_path):
        existing = pd.read_parquet(output_path)
        frames.append(existing)
        print(f"  [~] existing parquet: {len(existing)} rows")

    for path in paths:
        df = parse_xlsx(path)
        if df is not None and not df.empty:
            frames.append(df)
            print(f"  [+] {os.path.basename(path)}: {len(df)} rows")
        else:
            print(f"  [-] {os.path.basename(path)}: skipped")

    if not frames:
        print("No valid data to stitch.")
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    combined.to_parquet(output_path, engine="pyarrow", compression="snappy")

    span = f"{combined.index.min().date()} → {combined.index.max().date()}"
    print(f"\nStitched {len(frames)} source(s) → {len(combined)} rows  ({span})")
    print(f"Saved: {os.path.abspath(output_path)}")
    return combined


if __name__ == "__main__":
    args = sys.argv[1:]
    input_dir = args[0] if len(args) > 0 else DEFAULT_INPUT_DIR
    output_path = args[1] if len(args) > 1 else DEFAULT_OUTPUT
    print(f"Input : {input_dir}")
    print(f"Output: {output_path}\n")
    stitch(input_dir, output_path)
