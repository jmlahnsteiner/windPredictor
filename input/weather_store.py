"""
input/weather_store.py — Load and upsert weather station readings.

Backed by Supabase (PostgreSQL) when SUPABASE_DB_URL is set,
or a local SQLite file otherwise.
"""
import os
import sqlite3
from datetime import date
from typing import Optional

import pandas as pd

from utils.db import DEFAULT_SQLITE, backend, get_connection, placeholder

# All sensor columns stored (excluding the 'timestamp' primary key)
_COLUMNS = [
    "temperature", "feels_like", "dew_point", "humidity",
    "solar", "uvi", "rain_rate", "daily_rain",
    "pressure_relative", "pressure_absolute",
    "water_temperature", "wind_speed", "wind_gust", "wind_direction",
]

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS weather_readings (
    timestamp           TEXT PRIMARY KEY,
    temperature         REAL,
    feels_like          REAL,
    dew_point           REAL,
    humidity            REAL,
    solar               REAL,
    uvi                 REAL,
    rain_rate           REAL,
    daily_rain          REAL,
    pressure_relative   REAL,
    pressure_absolute   REAL,
    water_temperature   REAL,
    wind_speed          REAL,
    wind_gust           REAL,
    wind_direction      REAL
);
CREATE INDEX IF NOT EXISTS idx_wr_timestamp ON weather_readings(timestamp);
"""


def _ensure_schema(con, bk: str) -> None:
    if bk == "sqlite":
        con.executescript(_SQLITE_SCHEMA)
        con.commit()
    else:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS weather_readings (
                timestamp TEXT PRIMARY KEY,
                temperature REAL, feels_like REAL, dew_point REAL,
                humidity REAL, solar REAL, uvi REAL,
                rain_rate REAL, daily_rain REAL,
                pressure_relative REAL, pressure_absolute REAL,
                water_temperature REAL, wind_speed REAL,
                wind_gust REAL, wind_direction REAL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_wr_timestamp ON weather_readings(timestamp)"
        )
        con.commit()


def upsert_readings(df: pd.DataFrame, db_path: str = DEFAULT_SQLITE) -> int:
    """
    Upsert DataFrame rows into weather_readings.

    df must have a DatetimeIndex. Missing columns are stored as NULL.
    Returns number of rows upserted.
    """
    if df.empty:
        return 0

    rows = []
    for ts, row in df.iterrows():
        vals = [ts.isoformat()]
        for col in _COLUMNS:
            v = row[col] if col in row.index else None
            vals.append(float(v) if v is not None and pd.notna(v) else None)
        rows.append(tuple(vals))

    con, bk = get_connection(db_path)
    try:
        _ensure_schema(con, bk)
        ph = placeholder(bk)

        col_names = ", ".join(["timestamp"] + _COLUMNS)
        placeholders = ", ".join([ph] * (len(_COLUMNS) + 1))

        if bk == "postgres":
            update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in _COLUMNS)
            sql = (
                f"INSERT INTO weather_readings ({col_names}) VALUES ({placeholders}) "
                f"ON CONFLICT (timestamp) DO UPDATE SET {update_clause}"
            )
        else:
            sql = (
                f"INSERT OR REPLACE INTO weather_readings ({col_names}) "
                f"VALUES ({placeholders})"
            )

        cur = con.cursor()
        cur.executemany(sql, rows)
        con.commit()
    finally:
        con.close()

    return len(rows)


def load_weather_readings(
    start: Optional[date] = None,
    end: Optional[date] = None,
    db_path: str = DEFAULT_SQLITE,
) -> pd.DataFrame:
    """
    Load weather readings as a DatetimeTZNaive-indexed DataFrame.

    start / end are inclusive calendar-date bounds (None = no bound).
    Returns an empty DataFrame when no data is available.
    """
    bk = backend()

    if bk == "sqlite":
        if not os.path.exists(db_path):
            return pd.DataFrame()
        con = sqlite3.connect(db_path)
        con.executescript(_SQLITE_SCHEMA)
        con.commit()
        bk_local = "sqlite"
    else:
        con, bk_local = get_connection(db_path)
        _ensure_schema(con, bk_local)

    ph = placeholder(bk_local)
    where_clauses: list[str] = []
    params: list = []

    if start is not None:
        where_clauses.append(f"timestamp >= {ph}")
        params.append(start.isoformat() + "T00:00:00")
    if end is not None:
        where_clauses.append(f"timestamp <= {ph}")
        params.append(end.isoformat() + "T23:59:59")

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sql = f"SELECT * FROM weather_readings {where} ORDER BY timestamp"

    try:
        cur = con.cursor()
        cur.execute(sql, params) if params else cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    finally:
        con.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)
    df = df.assign(timestamp=pd.to_datetime(df["timestamp"]))
    df = df.set_index("timestamp").sort_index()
    numeric_updates = {
        col: pd.to_numeric(df[col], errors="coerce")
        for col in _COLUMNS
        if col in df.columns
    }
    df = df.assign(**numeric_updates)

    return df
