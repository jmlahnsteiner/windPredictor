# NWP-enriched RF + GRU Experiment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed Open-Meteo NWP forecast features into the Random Forest (so the model sees synoptic context it currently ignores), and build a GRU sequence model as a parallel experiment.

**Architecture:** Phase 1 adds a `nwp_readings` DB table (ERA5 historical backfill + live forecast), extracts 9 NWP sailing-window features inside `extract_snapshot_features`, and wires them through train/predict. Phase 2 builds a PyTorch GRU that ingests raw 24h station sequences + the NWP context vector, evaluated against the NWP-enriched RF on identical temporal folds.

**Tech Stack:** Python, scikit-learn, PyTorch, Open-Meteo archive API, SQLite/Supabase

**Spec:** `docs/superpowers/specs/2026-04-05-nwp-rf-gru-design.md`

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `input/nwp_store.py` | upsert/load `nwp_readings` table (mirrors `weather_store.py`) |
| Create | `input/open_meteo_historical.py` | fetch ERA5 historical NWP in 90-day chunks |
| Create | `supabase/schema_nwp.sql` | DDL for `nwp_readings` table |
| Modify | `model/features.py` | add `nwp_df`/`cfg` params + 9 NWP feature keys |
| Modify | `model/train.py` | load NWP, pass to `build_training_pairs`, switch to `TimeSeriesSplit` |
| Modify | `model/predict.py` | fetch NWP once in `predict_now`, pass to `predict_snapshot` |
| Modify | `deploy.py` | add `--backfill-nwp` flag |
| Modify | `requirements.txt` | add `torch` |
| Create | `model/features_sequence.py` | hourly resample → (T=24, F=7) sequence arrays |
| Create | `model/gru_model.py` | PyTorch GRU architecture |
| Create | `model/train_gru.py` | GRU training + temporal CV + comparison report |
| Create | `tests/input/test_nwp_store.py` | store roundtrip tests |
| Create | `tests/input/test_open_meteo_historical.py` | historical fetch tests (mocked HTTP) |
| Create | `tests/model/__init__.py` | test package init |
| Create | `tests/model/test_features_nwp.py` | NWP feature extraction tests |
| Create | `tests/model/test_features_sequence.py` | sequence prep tests |
| Create | `tests/model/test_gru_model.py` | GRU forward pass tests |

---

## Chunk 1: Phase 1 — NWP-enriched Random Forest

### Task 1: nwp_readings table + nwp_store.py

**Files:**
- Create: `supabase/schema_nwp.sql`
- Create: `input/nwp_store.py`
- Create: `tests/input/test_nwp_store.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/input/test_nwp_store.py`:

```python
import os
import tempfile
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest


def _make_nwp_df(n=8):
    """NWP DataFrame with tz-aware index (like fetch_forecast returns)."""
    idx = pd.date_range(
        "2026-04-05 08:00", periods=n, freq="1h",
        tz="Europe/Vienna",
    )
    return pd.DataFrame({
        "temperature":    [15.0] * n,
        "wind_speed":     [5.0]  * n,
        "wind_direction": [180.0] * n,
        "wind_gust":      [7.0]  * n,
        "cloud_cover":    [30.0] * n,
        "blh":            [800.0] * n,
        "direct_radiation": [400.0] * n,
    }, index=idx)


def _no_supabase():
    return {k: v for k, v in os.environ.items() if k != "SUPABASE_DB_URL"}


def test_upsert_and_load_roundtrip():
    df = _make_nwp_df()
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase(), clear=True):
            from input.nwp_store import upsert_nwp_readings, load_nwp_readings
            n = upsert_nwp_readings(df, db_path=db)
            assert n == 8
            result = load_nwp_readings(db_path=db)
            assert len(result) == 8
            assert isinstance(result.index, pd.DatetimeIndex)
            assert result["wind_speed"].iloc[0] == pytest.approx(5.0)


def test_upsert_idempotent():
    df = _make_nwp_df()
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase(), clear=True):
            from input.nwp_store import upsert_nwp_readings, load_nwp_readings
            upsert_nwp_readings(df, db_path=db)
            upsert_nwp_readings(df, db_path=db)
            assert len(load_nwp_readings(db_path=db)) == 8


def test_date_range_filter():
    df = _make_nwp_df()
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase(), clear=True):
            from input.nwp_store import upsert_nwp_readings, load_nwp_readings
            upsert_nwp_readings(df, db_path=db)
            result = load_nwp_readings(start=date(2026, 4, 6), db_path=db)
            assert result.empty


def test_load_empty_db():
    with patch.dict(os.environ, _no_supabase(), clear=True):
        from input.nwp_store import load_nwp_readings
        result = load_nwp_readings(db_path="/tmp/nonexistent_nwp_test.db")
        assert result.empty
        assert isinstance(result, pd.DataFrame)


def test_index_is_tz_aware():
    """Timestamps stored with tz offset round-trip back as tz-aware."""
    df = _make_nwp_df()
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase(), clear=True):
            from input.nwp_store import upsert_nwp_readings, load_nwp_readings
            upsert_nwp_readings(df, db_path=db)
            result = load_nwp_readings(db_path=db)
            assert result.index.tz is not None
```

- [ ] **Step 1.2: Run tests — expect import failure**

```bash
cd /path/to/windPredictor
python -m pytest tests/input/test_nwp_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'input.nwp_store'`

- [ ] **Step 1.3: Create `supabase/schema_nwp.sql`**

```sql
-- nwp_readings: hourly Open-Meteo NWP data (ERA5 historical + live forecast)
CREATE TABLE IF NOT EXISTS nwp_readings (
    timestamp         TEXT PRIMARY KEY,
    temperature       REAL,
    wind_speed        REAL,
    wind_direction    REAL,
    wind_gust         REAL,
    cloud_cover       REAL,
    blh               REAL,
    direct_radiation  REAL
);
CREATE INDEX IF NOT EXISTS idx_nwp_timestamp ON nwp_readings(timestamp);
```

- [ ] **Step 1.4: Create `input/nwp_store.py`**

Model this exactly on `input/weather_store.py`. Key difference: timestamps from Open-Meteo are tz-aware (ISO 8601 with offset), so `load_nwp_readings` must preserve timezone on parse.

```python
"""
input/nwp_store.py — Store and load Open-Meteo NWP readings.

Backed by Supabase (PostgreSQL) or local SQLite (same as weather_store).
"""
import os
import sqlite3
from datetime import date
from typing import Optional

import pandas as pd

from utils.db import DEFAULT_SQLITE, backend, get_connection, placeholder

# Columns mirror the _COL_NAMES mapping in open_meteo.py
_COLUMNS = [
    "temperature", "wind_speed", "wind_direction", "wind_gust",
    "cloud_cover", "blh", "direct_radiation",
]

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS nwp_readings (
    timestamp        TEXT PRIMARY KEY,
    temperature      REAL,
    wind_speed       REAL,
    wind_direction   REAL,
    wind_gust        REAL,
    cloud_cover      REAL,
    blh              REAL,
    direct_radiation REAL
);
CREATE INDEX IF NOT EXISTS idx_nwp_timestamp ON nwp_readings(timestamp);
"""


def _ensure_schema(con, bk: str) -> None:
    if bk == "sqlite":
        con.executescript(_SQLITE_SCHEMA)
        con.commit()
    else:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nwp_readings (
                timestamp TEXT PRIMARY KEY,
                temperature REAL, wind_speed REAL, wind_direction REAL,
                wind_gust REAL, cloud_cover REAL, blh REAL,
                direct_radiation REAL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_nwp_timestamp ON nwp_readings(timestamp)"
        )
        con.commit()


def upsert_nwp_readings(df: pd.DataFrame, db_path: str = DEFAULT_SQLITE) -> int:
    """
    Upsert NWP DataFrame rows into nwp_readings.

    df must have a DatetimeIndex (tz-aware). Missing columns stored as NULL.
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
                f"INSERT INTO nwp_readings ({col_names}) VALUES ({placeholders}) "
                f"ON CONFLICT (timestamp) DO UPDATE SET {update_clause}"
            )
        else:
            sql = (
                f"INSERT OR REPLACE INTO nwp_readings ({col_names}) "
                f"VALUES ({placeholders})"
            )

        cur = con.cursor()
        cur.executemany(sql, rows)
        con.commit()
    finally:
        con.close()

    return len(rows)


def load_nwp_readings(
    start: Optional[date] = None,
    end: Optional[date] = None,
    db_path: str = DEFAULT_SQLITE,
) -> pd.DataFrame:
    """
    Load NWP readings as a tz-aware DatetimeIndex DataFrame.

    start / end are inclusive calendar-date bounds (None = no bound).
    Returns empty DataFrame when no data is available.
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
    sql = f"SELECT * FROM nwp_readings {where} ORDER BY timestamp"

    try:
        cur = con.cursor()
        cur.execute(sql, params) if params else cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows_data = cur.fetchall()
    finally:
        con.close()

    if not rows_data:
        return pd.DataFrame()

    df = pd.DataFrame(rows_data, columns=cols)
    # utc=True handles mixed/offset-aware ISO strings → tz-aware UTC, then convert
    df = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))
    df = df.set_index("timestamp").sort_index()
    numeric_updates = {
        col: pd.to_numeric(df[col], errors="coerce")
        for col in _COLUMNS
        if col in df.columns
    }
    df = df.assign(**numeric_updates)
    return df
```

- [ ] **Step 1.5: Run tests — expect all pass**

```bash
python -m pytest tests/input/test_nwp_store.py -v
```

Expected: 5 passed

- [ ] **Step 1.6: Commit**

```bash
git add supabase/schema_nwp.sql input/nwp_store.py tests/input/test_nwp_store.py
git commit -m "feat: add nwp_readings table and nwp_store upsert/load"
```

---

### Task 2: open_meteo_historical.py (ERA5 backfill)

**Files:**
- Create: `input/open_meteo_historical.py`
- Create: `tests/input/test_open_meteo_historical.py`

- [ ] **Step 2.1: Write the failing tests**

Create `tests/input/test_open_meteo_historical.py`:

```python
import json
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _mock_response(start_date: str, n_hours: int = 4):
    """Build a fake Open-Meteo archive API response."""
    import pandas as _pd
    times = _pd.date_range(start_date + " 00:00", periods=n_hours, freq="1h")
    data = {
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "temperature_2m":        [15.0] * n_hours,
            "wind_speed_10m":        [5.0]  * n_hours,
            "wind_direction_10m":    [180.0] * n_hours,
            "wind_gusts_10m":        [7.0]  * n_hours,
            "cloud_cover":           [30.0] * n_hours,
            "boundary_layer_height": [800.0] * n_hours,
            "direct_radiation":      [400.0] * n_hours,
        },
        "utc_offset_seconds": 3600,
    }
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = data
    return mock


def test_fetch_historical_chunk_returns_dataframe():
    with patch("requests.get", return_value=_mock_response("2026-04-05")) as mock_get:
        from input.open_meteo_historical import fetch_historical_chunk
        df = fetch_historical_chunk(47.8, 13.7, date(2026, 4, 5), date(2026, 4, 5))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "wind_speed" in df.columns
    assert "blh" in df.columns


def test_fetch_historical_chunk_uses_kn_and_timezone():
    """Must pass wind_speed_unit=kn and timezone=auto to the API."""
    with patch("requests.get", return_value=_mock_response("2026-04-05")) as mock_get:
        from input.open_meteo_historical import fetch_historical_chunk
        fetch_historical_chunk(47.8, 13.7, date(2026, 4, 5), date(2026, 4, 5))
    call_params = mock_get.call_args[1]["params"]
    assert call_params.get("wind_speed_unit") == "kn"
    assert call_params.get("timezone") == "auto"


def test_fetch_historical_chunk_returns_empty_on_error():
    mock = MagicMock()
    mock.raise_for_status.side_effect = Exception("network error")
    with patch("requests.get", return_value=mock):
        from input.open_meteo_historical import fetch_historical_chunk
        df = fetch_historical_chunk(47.8, 13.7, date(2026, 4, 5), date(2026, 4, 5))
    assert df.empty


def test_fetch_historical_range_chunks_90_days():
    """A range > 90 days should result in multiple API calls."""
    with patch("input.open_meteo_historical.fetch_historical_chunk",
               return_value=pd.DataFrame()) as mock_chunk:
        from input.open_meteo_historical import fetch_historical_range
        fetch_historical_range(47.8, 13.7, date(2024, 1, 1), date(2024, 6, 1))
    # 5 months > 90 days → at least 2 chunks
    assert mock_chunk.call_count >= 2
```

- [ ] **Step 2.2: Run tests — expect import failure**

```bash
python -m pytest tests/input/test_open_meteo_historical.py -v
```

Expected: `ModuleNotFoundError: No module named 'input.open_meteo_historical'`

- [ ] **Step 2.3: Create `input/open_meteo_historical.py`**

```python
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
    Fetch hourly ERA5 data for a single date range (≤90 days recommended).

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
    Fetch ERA5 data for an arbitrary date range, chunked into ≤90-day requests.

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
```

- [ ] **Step 2.4: Run tests — expect all pass**

```bash
python -m pytest tests/input/test_open_meteo_historical.py -v
```

Expected: 4 passed

- [ ] **Step 2.5: Commit**

```bash
git add input/open_meteo_historical.py tests/input/test_open_meteo_historical.py
git commit -m "feat: add ERA5 historical NWP fetch in 90-day chunks"
```

---

### Task 3: NWP features in features.py

**Files:**
- Modify: `model/features.py`
- Create: `tests/model/__init__.py`
- Create: `tests/model/test_features_nwp.py`

- [ ] **Step 3.1: Create test package and write failing tests**

```bash
touch tests/model/__init__.py
```

Create `tests/model/test_features_nwp.py`:

```python
import numpy as np
import pandas as pd
import pytest


def _make_station_df():
    """Minimal station df — enough for extract_snapshot_features to return non-None."""
    idx = pd.date_range("2026-04-05 00:00", periods=48, freq="30min")
    n = len(idx)
    return pd.DataFrame({
        "wind_speed":        [5.0] * n,
        "wind_direction":    [180.0] * n,
        "wind_gust":         [7.0] * n,
        "temperature":       [15.0] * n,
        "humidity":          [60.0] * n,
        "pressure_relative": [1013.0] * n,
    }, index=idx)


def _make_nwp_df(target_date="2026-04-05"):
    """NWP df covering the sailing window of target_date (tz-aware)."""
    # 08:00–16:00 local → 8 hours
    idx = pd.date_range(
        f"{target_date} 08:00", periods=8, freq="1h", tz="Europe/Vienna"
    )
    return pd.DataFrame({
        "temperature":    [18.0] * 8,
        "wind_speed":     [6.0]  * 8,
        "wind_direction": [200.0] * 8,
        "wind_gust":      [9.0]  * 8,
        "cloud_cover":    [20.0] * 8,
        "blh":            [1000.0] * 8,
        "direct_radiation": [500.0] * 8,
    }, index=idx)


def _cfg():
    return {
        "sailing": {
            "window_start": "08:00",
            "window_end":   "16:00",
        }
    }


def test_nwp_features_populated_when_provided():
    """When nwp_df and cfg are provided, all nwp_* features should be finite floats."""
    from model.features import extract_snapshot_features

    station_df = _make_station_df()
    nwp_df = _make_nwp_df("2026-04-05")
    cfg = _cfg()
    # Before window_end (16:00) → target is same day
    snap_dt = pd.Timestamp("2026-04-05 06:00")

    feats = extract_snapshot_features(station_df, snap_dt, nwp_df=nwp_df, cfg=cfg)
    assert feats is not None

    assert feats["nwp_wind_speed_mean"] == pytest.approx(6.0)
    assert feats["nwp_wind_speed_max"]  == pytest.approx(6.0)
    assert feats["nwp_wind_gust_max"]   == pytest.approx(9.0)
    assert feats["nwp_cloud_cover_mean"] == pytest.approx(20.0)
    assert feats["nwp_blh_mean"]         == pytest.approx(1000.0)
    assert feats["nwp_direct_radiation_mean"] == pytest.approx(500.0)
    # Direction sin/cos should be finite
    assert np.isfinite(feats["nwp_wind_dir_sin"])
    assert np.isfinite(feats["nwp_wind_dir_cos"])
    # Consistency for uniform direction → near 0
    assert feats["nwp_dir_consistency"] == pytest.approx(0.0, abs=1e-3)


def test_nwp_features_nan_when_not_provided():
    """Without nwp_df, all nwp_* features must be NaN."""
    from model.features import extract_snapshot_features

    station_df = _make_station_df()
    snap_dt = pd.Timestamp("2026-04-05 06:00")

    feats = extract_snapshot_features(station_df, snap_dt)
    assert feats is not None

    nwp_keys = [
        "nwp_wind_speed_mean", "nwp_wind_speed_max", "nwp_wind_gust_max",
        "nwp_wind_dir_sin", "nwp_wind_dir_cos", "nwp_dir_consistency",
        "nwp_cloud_cover_mean", "nwp_blh_mean", "nwp_direct_radiation_mean",
    ]
    for k in nwp_keys:
        assert k in feats, f"Missing key: {k}"
        assert np.isnan(feats[k]), f"Expected NaN for {k}, got {feats[k]}"


def test_nwp_target_date_after_window_end():
    """After window_end (16:00), target is next day — NWP for next day is used."""
    from model.features import extract_snapshot_features

    station_df = _make_station_df()
    # NWP for next day (2026-04-06)
    nwp_df = _make_nwp_df("2026-04-06")
    cfg = _cfg()
    snap_dt = pd.Timestamp("2026-04-05 18:00")  # after 16:00 → target is 2026-04-06

    feats = extract_snapshot_features(station_df, snap_dt, nwp_df=nwp_df, cfg=cfg)
    assert feats is not None
    assert feats["nwp_wind_speed_mean"] == pytest.approx(6.0)


def test_existing_features_unchanged():
    """Adding NWP params must not break any of the original 28 features."""
    from model.features import extract_snapshot_features

    station_df = _make_station_df()
    snap_dt = pd.Timestamp("2026-04-05 06:00")

    feats_before = extract_snapshot_features(station_df, snap_dt)
    feats_after  = extract_snapshot_features(station_df, snap_dt, nwp_df=None, cfg=None)

    original_keys = [k for k in feats_before if not k.startswith("nwp_")]
    for k in original_keys:
        v_before = feats_before[k]
        v_after  = feats_after[k]
        if v_before is None or (isinstance(v_before, float) and np.isnan(v_before)):
            assert v_after is None or np.isnan(v_after)
        else:
            assert v_after == pytest.approx(v_before, rel=1e-6), f"Key {k} changed"
```

- [ ] **Step 3.2: Run tests — expect failure**

```bash
python -m pytest tests/model/test_features_nwp.py -v
```

Expected: FAILED — `extract_snapshot_features() got unexpected keyword argument 'nwp_df'`

- [ ] **Step 3.3: Modify `model/features.py`**

Change `extract_snapshot_features` signature and add NWP feature block. The function currently ends with `return { ... large dict ... }`. Refactor to assign the dict to `feats`, append NWP keys, then return.

**Signature change** (line ~145):
```python
def extract_snapshot_features(
    df: pd.DataFrame,
    snap_dt: pd.Timestamp,
    nwp_df: Optional[pd.DataFrame] = None,
    cfg: Optional[dict] = None,
) -> Optional[dict]:
```

**At the end of the function**, replace the `return { ... }` with:

```python
    feats = {
        # Time context
        "snapshot_hour": snap_dt.hour,
        "day_of_week":   snap_dt.dayofweek,
        "sin_doy":       np.sin(2 * np.pi * snap_dt.dayofyear / 365),
        "cos_doy":       np.cos(2 * np.pi * snap_dt.dayofyear / 365),
        # ... all existing keys unchanged ...
        "humidity_min_24h": (
            float(past_24h["humidity"].min()) if n_24h >= 4 else np.nan
        ),
    }

    # ── NWP features ─────────────────────────────────────────────────────────
    # Synoptic-scale forecast for the target day's sailing window.
    # NaN when NWP data is unavailable — RF imputes with training medians.
    _NWP_KEYS = (
        "nwp_wind_speed_mean", "nwp_wind_speed_max", "nwp_wind_gust_max",
        "nwp_wind_dir_sin", "nwp_wind_dir_cos", "nwp_dir_consistency",
        "nwp_cloud_cover_mean", "nwp_blh_mean", "nwp_direct_radiation_mean",
    )

    nwp_feats = {k: np.nan for k in _NWP_KEYS}

    if nwp_df is not None and not nwp_df.empty and cfg is not None:
        sc = cfg["sailing"]
        target_date = _target_date(snap_dt, sc["window_end"])

        # Slice to target date (tz-aware index: .date() returns local date)
        mask = pd.Series(nwp_df.index, dtype=object).apply(
            lambda t: t.date() == target_date
        ).values
        day_nwp = nwp_df.iloc[mask]

        if not day_nwp.empty:
            window_nwp = day_nwp.between_time(sc["window_start"], sc["window_end"])
            if len(window_nwp) >= 1:
                ws  = window_nwp["wind_speed"].dropna()
                wg  = window_nwp["wind_gust"].dropna()
                wd  = window_nwp["wind_direction"].dropna()
                cc  = window_nwp["cloud_cover"].dropna()
                blh = window_nwp["blh"].dropna()
                dr  = window_nwp["direct_radiation"].dropna()

                if not wd.empty:
                    rad = np.radians(wd.to_numpy())
                    sin_m = float(np.sin(rad).mean())
                    cos_m = float(np.cos(rad).mean())
                    mag = max(np.hypot(sin_m, cos_m), 1e-9)
                    nwp_feats["nwp_wind_dir_sin"] = sin_m / mag
                    nwp_feats["nwp_wind_dir_cos"] = cos_m / mag
                    nwp_feats["nwp_dir_consistency"] = (
                        _circular_std(wd) if len(wd) >= 2 else np.nan
                    )

                if not ws.empty:
                    nwp_feats["nwp_wind_speed_mean"] = float(ws.mean())
                    nwp_feats["nwp_wind_speed_max"]  = float(ws.max())
                if not wg.empty:
                    nwp_feats["nwp_wind_gust_max"] = float(wg.max())
                if not cc.empty:
                    nwp_feats["nwp_cloud_cover_mean"] = float(cc.mean())
                if not blh.empty:
                    nwp_feats["nwp_blh_mean"] = float(blh.mean())
                if not dr.empty:
                    nwp_feats["nwp_direct_radiation_mean"] = float(dr.mean())

    feats.update(nwp_feats)
    return feats
```

**Also update `build_training_pairs`** signature and the call to `extract_snapshot_features` on line 343:

```python
def build_training_pairs(
    df: pd.DataFrame,
    cfg: dict,
    nwp_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    ...
    # Change the call inside the loop from:
    features = extract_snapshot_features(df, snap_dt)
    # To:
    features = extract_snapshot_features(df, snap_dt, nwp_df=nwp_df, cfg=cfg)
```

- [ ] **Step 3.4: Run tests — expect all pass**

```bash
python -m pytest tests/model/test_features_nwp.py -v
```

Expected: 4 passed

- [ ] **Step 3.5: Run full test suite — no regressions**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 3.6: Commit**

```bash
git add model/features.py tests/model/__init__.py tests/model/test_features_nwp.py
git commit -m "feat: add 9 NWP features to extract_snapshot_features"
```

---

### Task 4: Wire NWP into train.py + switch to TimeSeriesSplit

**Files:**
- Modify: `model/train.py`

There are no new test files for this task — the change is wiring already-tested components.
Run the existing test suite as the verification check.

- [ ] **Step 4.1: Modify `model/train.py`**

**Add import** at the top (after existing imports):
```python
from sklearn.model_selection import TimeSeriesSplit
```

**Add NWP load** after `df = load_weather_readings()`:
```python
    print("Loading NWP readings from database …")
    from input.nwp_store import load_nwp_readings
    nwp_df = load_nwp_readings()
    if nwp_df.empty:
        print("  [!] No NWP data found — run open_meteo_historical.py to backfill.")
        print("  Proceeding with station features only (nwp_* features will be NaN).")
        nwp_df = None
    else:
        print(f"  NWP readings: {len(nwp_df):,} rows  "
              f"({nwp_df.index.min().date()} → {nwp_df.index.max().date()})")
```

**Update `build_training_pairs` call** to pass `nwp_df`:
```python
    X, y = build_training_pairs(df, cfg, nwp_df=nwp_df)
```

**Replace `StratifiedKFold` cross-validation** with `TimeSeriesSplit`:
```python
    # Cross-validation only when we have enough samples
    if len(X) >= 10 and y.nunique() > 1:
        n_splits = min(5, y.value_counts().min())
        cv = TimeSeriesSplit(n_splits=n_splits)   # temporal: no shuffle, no future leakage
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        print(f"\nCV ROC-AUC : {scores.mean():.3f} ± {scores.std():.3f}  (k={n_splits}, temporal)")
```

Also **remove the unused `StratifiedKFold` import** from the import line.

- [ ] **Step 4.2: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all pass

- [ ] **Step 4.3: Commit**

```bash
git add model/train.py
git commit -m "feat: feed NWP features into RF training; switch to TimeSeriesSplit CV"
```

---

### Task 5: Wire NWP into predict.py

**Files:**
- Modify: `model/predict.py`

- [ ] **Step 5.1: Refactor `predict.py` to fetch NWP once and pass to feature extraction**

**Add a new helper** `_fetch_nwp_df(cfg)` just before `predict_now`:

```python
def _fetch_nwp_df(cfg: dict) -> "pd.DataFrame":
    """
    Fetch live NWP forecast for the configured location.
    Returns empty DataFrame when location is not configured or fetch fails.
    """
    loc = cfg.get("location", {})
    lat, lon = loc.get("lat"), loc.get("lon")
    if lat is None or lon is None:
        return pd.DataFrame()

    try:
        from input.open_meteo import fetch_forecast
    except ImportError:
        return pd.DataFrame()

    print("Fetching NWP forecast from Open-Meteo …", flush=True)
    return fetch_forecast(lat, lon)
```

**Update `predict_snapshot` signature** to accept `nwp_df`:
```python
def predict_snapshot(
    df: pd.DataFrame,
    snap_dt: pd.Timestamp,
    bundle: dict,
    cfg: dict,
    nwp_df: Optional["pd.DataFrame"] = None,
) -> dict:
```

**Update the `extract_snapshot_features` call** inside `predict_snapshot` (line ~131):
```python
    features = extract_snapshot_features(df, snap_dt, nwp_df=nwp_df, cfg=cfg)
```

**Update `predict_now`** to fetch NWP once and pass it down:

At the top of `predict_now`, after `bundle = joblib.load(model_path)`:
```python
    nwp_df = _fetch_nwp_df(cfg)
```

Update the `predict_snapshot` call to pass `nwp_df`:
```python
    result = predict_snapshot(df, snap_dt, bundle, cfg, nwp_df=nwp_df)
```

**Remove the old body of `_enrich_with_nwp`** (the entire block starting with `loc = cfg.get("location", {})` through the `fetch_forecast` call and the for-loop) and replace it with the new function below:
```python
def _enrich_with_nwp(
    results: list[dict],
    cfg: dict,
    nwp_df: Optional["pd.DataFrame"] = None,
) -> None:
    """
    Attach nwp_forecast display stats to each result. Mutates results in-place.
    Accepts a pre-fetched nwp_df; fetches fresh if not provided.
    """
    if nwp_df is None or (hasattr(nwp_df, "empty") and nwp_df.empty):
        # Fallback: try to fetch (for callers that don't pre-fetch)
        nwp_df = _fetch_nwp_df(cfg)

    if nwp_df is None or (hasattr(nwp_df, "empty") and nwp_df.empty):
        return

    try:
        from input.open_meteo import sailing_window_stats
    except ImportError:
        return

    sc = cfg["sailing"]
    for result in results:
        if "error" in result or "predicting_date" not in result:
            continue
        tgt_date = date.fromisoformat(result["predicting_date"])
        stats = sailing_window_stats(nwp_df, tgt_date, sc["window_start"], sc["window_end"])
        if stats:
            result["nwp_forecast"] = stats
```

Update the `_enrich_with_nwp` call at the bottom of `predict_now`:
```python
    _enrich_with_nwp(results, cfg, nwp_df=nwp_df)
```

**Remove the old NWP fetch block inside `_enrich_with_nwp`** (the entire `loc = cfg.get(...)` + `fetch_forecast` block that was there before).

- [ ] **Step 5.2: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all pass

- [ ] **Step 5.3: Commit**

```bash
git add model/predict.py
git commit -m "feat: pass live NWP forecast into predict_snapshot feature extraction"
```

---

### Task 6: deploy.py --backfill-nwp + retrain

**Files:**
- Modify: `deploy.py`
- Modify: `requirements.txt`

- [ ] **Step 6.1: Add `torch` to requirements.txt**

Add at the end of `requirements.txt`:
```
torch>=2.0.0
```

- [ ] **Step 6.2: Add `--backfill-nwp` flag to deploy.py**

Add a new step function before `main()`:
```python
def step_backfill_nwp(start: str, end: str) -> None:
    _banner(f"[NWP] Backfilling ERA5 NWP data  ({start} → {end})")
    from input.open_meteo_historical import fetch_historical_range
    from input.nwp_store import upsert_nwp_readings
    from utils.config import load_config
    import datetime

    cfg = _load_config()
    loc = cfg.get("location", {})
    lat, lon = loc.get("lat"), loc.get("lon")
    if lat is None or lon is None:
        raise RuntimeError("[location] lat/lon not set in config.toml")

    df = fetch_historical_range(
        lat, lon,
        datetime.date.fromisoformat(start),
        datetime.date.fromisoformat(end),
    )
    if df.empty:
        raise RuntimeError("No NWP data returned from ERA5 API")
    n = upsert_nwp_readings(df)
    print(f"Upserted {n} NWP rows", flush=True)
```

Add `--backfill-nwp` argument to the parser:
```python
    parser.add_argument(
        "--backfill-nwp",
        nargs=2, metavar=("START", "END"),
        help="Fetch ERA5 NWP history for date range (YYYY-MM-DD YYYY-MM-DD)",
    )
```

Handle it in `main()` before `step_download`:
```python
    if args.backfill_nwp:
        step_backfill_nwp(args.backfill_nwp[0], args.backfill_nwp[1])
        return
```

- [ ] **Step 6.3: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all pass

- [ ] **Step 6.4: Commit**

```bash
git add deploy.py requirements.txt
git commit -m "feat: add --backfill-nwp flag to deploy.py; add torch dependency"
```

- [ ] **Step 6.5: Backfill 2 years of historical NWP data**

```bash
python deploy.py --backfill-nwp 2024-01-01 2026-04-05
```

Expected output: fetches ~9 chunks of 90 days, upserts ~17,520 hourly rows.

- [ ] **Step 6.6: Retrain the RF model with NWP features**

```bash
python model/train.py
```

Expected output:
- "NWP readings: N rows …"
- Feature list now includes `nwp_wind_speed_mean`, etc.
- CV ROC-AUC printed
- "Saved: model/weights.joblib"

- [ ] **Step 6.7: Smoke-test predict**

```bash
python model/predict.py
```

Expected: no errors, prints forecast entries with valid probabilities.

- [ ] **Step 6.8: Commit updated weights**

```bash
git add model/weights.joblib
git commit -m "chore: retrain RF with NWP features (TimeSeriesSplit CV)"
```

---

## Chunk 2: Phase 2 — GRU Sequence Experiment

### Task 7: features_sequence.py

**Files:**
- Create: `model/features_sequence.py`
- Create: `tests/model/test_features_sequence.py`

- [ ] **Step 7.1: Write the failing tests**

Create `tests/model/test_features_sequence.py`:

```python
import numpy as np
import pandas as pd
import pytest


def _make_station_df(hours=48, freq="5min"):
    """Station df at given resolution."""
    n = hours * (60 // int(freq.rstrip("min")))
    idx = pd.date_range("2026-04-04 00:00", periods=n, freq=freq)
    return pd.DataFrame({
        "wind_speed":        np.random.uniform(3, 8, n),
        "wind_direction":    [180.0] * n,
        "temperature":       [15.0]  * n,
        "pressure_relative": [1013.0] * n,
        "humidity":          [60.0]  * n,
        "solar":             [200.0] * n,
    }, index=idx)


def _make_nwp_df(target_date="2026-04-05"):
    idx = pd.date_range(
        f"{target_date} 08:00", periods=8, freq="1h", tz="Europe/Vienna"
    )
    return pd.DataFrame({
        "temperature": [18.0] * 8, "wind_speed": [6.0] * 8,
        "wind_direction": [200.0] * 8, "wind_gust": [9.0] * 8,
        "cloud_cover": [20.0] * 8, "blh": [1000.0] * 8,
        "direct_radiation": [500.0] * 8,
    }, index=idx)


def _cfg():
    return {
        "sailing": {"window_start": "08:00", "window_end": "16:00"},
        "prediction": {"min_good_fraction": 0.25},
    }


def test_build_sequence_shape():
    """build_sequence returns (T, 7) array with T≤24."""
    from model.features_sequence import build_sequence
    df = _make_station_df()
    snap_dt = pd.Timestamp("2026-04-05 06:00")
    seq = build_sequence(df, snap_dt)
    assert seq.ndim == 2
    assert seq.shape[1] == 7   # 7 features per timestep
    assert seq.shape[0] <= 24  # at most 24 hours


def test_build_sequence_no_nans():
    """Sequence must not contain NaN (missing → 0.0)."""
    from model.features_sequence import build_sequence
    df = _make_station_df()
    snap_dt = pd.Timestamp("2026-04-05 06:00")
    seq = build_sequence(df, snap_dt)
    assert not np.any(np.isnan(seq))


def test_build_sequence_sparse_data():
    """Works even with sparse / short station data."""
    from model.features_sequence import build_sequence
    df = _make_station_df(hours=6)  # only 6 hours of data
    snap_dt = pd.Timestamp("2026-04-04 06:00")
    seq = build_sequence(df, snap_dt)
    assert seq.shape[1] == 7


def test_build_nwp_context_shape():
    """build_nwp_context returns (12,) vector."""
    from model.features_sequence import build_nwp_context
    nwp_df = _make_nwp_df("2026-04-05")
    cfg = _cfg()
    snap_dt = pd.Timestamp("2026-04-05 06:00")
    ctx = build_nwp_context(nwp_df, snap_dt, cfg)
    assert ctx.shape == (12,)


def test_build_nwp_context_populated_when_data_present():
    """Context vector has no NaN and non-zero NWP values when data is provided."""
    from model.features_sequence import build_nwp_context
    nwp_df = _make_nwp_df("2026-04-05")
    cfg = _cfg()
    snap_dt = pd.Timestamp("2026-04-05 06:00")
    ctx = build_nwp_context(nwp_df, snap_dt, cfg)
    assert not np.any(np.isnan(ctx))
    # NWP wind speed mean (index 0) should be 6.0 — not zero
    assert ctx[0] != 0.0, "NWP features were not extracted — got zeros"


def test_build_sequence_training_pairs_shapes():
    """build_sequence_training_pairs returns correct array shapes."""
    from model.features_sequence import build_sequence_training_pairs
    from model.features import compute_daily_target

    df = _make_station_df(hours=5 * 24)  # 5 days
    cfg = _cfg()
    nwp_df = None

    sequences, contexts, labels = build_sequence_training_pairs(df, cfg, nwp_df=None)
    assert sequences.ndim == 3        # (N, T, 7)
    assert contexts.ndim == 2         # (N, 12)
    assert labels.ndim == 1           # (N,)
    assert sequences.shape[0] == contexts.shape[0] == labels.shape[0]
    assert sequences.shape[2] == 7
    assert contexts.shape[1] == 12
```

- [ ] **Step 7.2: Run tests — expect import failure**

```bash
python -m pytest tests/model/test_features_sequence.py -v
```

Expected: `ModuleNotFoundError: No module named 'model.features_sequence'`

- [ ] **Step 7.3: Create `model/features_sequence.py`**

```python
"""
model/features_sequence.py — Raw time-series sequences for the GRU model.

Produces:
  build_sequence(df, snap_dt)               → (T, 7) float array
  build_nwp_context(nwp_df, snap_dt, cfg)   → (12,) float array
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

    Returns float32 array of shape (T, SEQ_FEATURES) where T ≤ hours.
    Missing timesteps and NaN values are zero-filled.
    """
    # Resample to 1h mean, keeping only columns we need
    source_cols = [c for c in ["wind_speed", "wind_direction",
                                "temperature", "pressure_relative",
                                "humidity", "solar"]
                   if c in df.columns]
    resampled = df[source_cols].resample("1h").mean()

    # Window: last `hours` ending at snap_dt (inclusive, floored to hour)
    snap_floor = snap_dt.floor("h")
    start_dt = snap_floor - pd.Timedelta(hours=hours - 1)
    window = resampled.loc[start_dt:snap_floor]

    # Build dense hourly index — ensures shape is always (hours, F)
    full_idx = pd.date_range(start=start_dt, end=snap_floor, freq="1h")
    window = window.reindex(full_idx)

    # Derive sin/cos of wind direction
    wd = window["wind_direction"].fillna(0.0)
    rad = np.radians(wd.to_numpy())
    sin_wd = np.sin(rad)
    cos_wd = np.cos(rad)

    arr = np.column_stack([
        window["wind_speed"].fillna(0.0).to_numpy(),
        sin_wd,
        cos_wd,
        window["temperature"].fillna(0.0).to_numpy()         if "temperature"       in window.columns else np.zeros(len(window)),
        window["pressure_relative"].fillna(0.0).to_numpy()   if "pressure_relative" in window.columns else np.zeros(len(window)),
        window["humidity"].fillna(0.0).to_numpy()            if "humidity"          in window.columns else np.zeros(len(window)),
        window["solar"].fillna(0.0).to_numpy()               if "solar"             in window.columns else np.zeros(len(window)),
    ])

    return arr.astype(np.float32)


def _extract_nwp_window_stats(
    nwp_df: pd.DataFrame,
    target_date: "datetime.date",
    cfg: dict,
) -> np.ndarray:
    """
    Extract the 9 NWP sailing-window features for target_date as a float32 array.
    Returns zeros (not NaN) so the GRU context vector is always finite.
    """
    from model.features import _circular_std, _NWP_KEYS
    import datetime

    sc = cfg["sailing"]
    mask = pd.Series(nwp_df.index, dtype=object).apply(
        lambda t: t.date() == target_date
    ).values
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

    Mirrors the snapshot/target logic of build_training_pairs in features.py:
    6 snapshot times per day, same target-date rules.

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

    min_good  = cfg["prediction"]["min_good_fraction"]
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

    # Pad sequences to uniform length (24 timesteps)
    T = 24
    padded = np.zeros((len(seqs), T, SEQ_FEATURES), dtype=np.float32)
    for i, s in enumerate(seqs):
        t = min(s.shape[0], T)
        padded[i, -t:, :] = s[-t:]   # right-align: most recent hour is last

    return padded, np.array(ctxs, dtype=np.float32), np.array(labs, dtype=np.int32)
```

**Note:** `_NWP_KEYS` needs to be exported from `model/features.py`. Add this line near where `_NWP_KEYS` is defined in `features.py`:

```python
# (make importable by features_sequence.py)
_NWP_KEYS = (
    "nwp_wind_speed_mean", "nwp_wind_speed_max", "nwp_wind_gust_max",
    "nwp_wind_dir_sin", "nwp_wind_dir_cos", "nwp_dir_consistency",
    "nwp_cloud_cover_mean", "nwp_blh_mean", "nwp_direct_radiation_mean",
)
```

Move the `_NWP_KEYS` tuple to module level in `features.py` (currently it's defined inline inside `extract_snapshot_features`). The tuple inside the function can then reference the module-level name.

- [ ] **Step 7.4: Run tests — expect all pass**

```bash
python -m pytest tests/model/test_features_sequence.py -v
```

Expected: 6 passed

- [ ] **Step 7.5: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all pass

- [ ] **Step 7.6: Commit**

```bash
git add model/features_sequence.py model/features.py tests/model/test_features_sequence.py
git commit -m "feat: add sequence feature builder and NWP context vector for GRU"
```

---

### Task 8: gru_model.py

**Files:**
- Create: `model/gru_model.py`
- Create: `tests/model/test_gru_model.py`

- [ ] **Step 8.1: Write the failing tests**

Create `tests/model/test_gru_model.py`:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")   # skip gracefully if torch not installed


def test_forward_pass_shape():
    """Output should be (B, 1) probabilities."""
    from model.gru_model import SailingGRU
    model = SailingGRU()
    B, T, F = 4, 24, 7
    seq = torch.randn(B, T, F)
    ctx = torch.randn(B, 12)
    out = model(seq, ctx)
    assert out.shape == (B, 1)


def test_output_in_range():
    """Sigmoid output must be in [0, 1]."""
    from model.gru_model import SailingGRU
    model = SailingGRU()
    seq = torch.randn(16, 24, 7)
    ctx = torch.randn(16, 12)
    out = model(seq, ctx)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_single_sample():
    """Batch size 1 should work."""
    from model.gru_model import SailingGRU
    model = SailingGRU()
    out = model(torch.randn(1, 24, 7), torch.randn(1, 12))
    assert out.shape == (1, 1)


def test_loss_is_scalar():
    """BCELoss on model output should be a scalar."""
    from model.gru_model import SailingGRU
    model = SailingGRU()
    seq = torch.randn(8, 24, 7)
    ctx = torch.randn(8, 12)
    labels = torch.randint(0, 2, (8,)).float()
    out = model(seq, ctx).squeeze(1)
    loss = torch.nn.BCELoss()(out, labels)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0
```

- [ ] **Step 8.2: Run tests — expect import failure**

```bash
python -m pytest tests/model/test_gru_model.py -v
```

Expected: FAILED — `ModuleNotFoundError: No module named 'model.gru_model'`

- [ ] **Step 8.3: Create `model/gru_model.py`**

```python
"""
model/gru_model.py — PyTorch GRU for sailing condition forecasting.

Architecture:
  Sequence input : (B, T=24, seq_features=7)  — hourly station readings
  Context input  : (B, context_features=12)   — NWP + time context

  GRU(seq_features → hidden_size, num_layers=2, dropout)
  → last hidden state (B, hidden_size)
  → concat context   (B, hidden_size + context_features)
  → Linear → ReLU → Dropout → Linear → Sigmoid
  → (B, 1)  probability
"""

import torch
import torch.nn as nn

from model.features_sequence import CONTEXT_FEATURES, SEQ_FEATURES


class SailingGRU(nn.Module):
    def __init__(
        self,
        seq_features: int = SEQ_FEATURES,
        context_features: int = CONTEXT_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        combined = hidden_size + context_features
        self.head = nn.Sequential(
            nn.Linear(combined, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        seq : (B, T, seq_features)
        ctx : (B, context_features)
        returns (B, 1) probabilities
        """
        _, hidden = self.gru(seq)          # hidden: (num_layers, B, hidden_size)
        last_hidden = hidden[-1]           # (B, hidden_size)
        combined = torch.cat([last_hidden, ctx], dim=1)   # (B, hidden+ctx)
        return self.head(combined)
```

- [ ] **Step 8.4: Run tests — expect all pass**

```bash
python -m pytest tests/model/test_gru_model.py -v
```

Expected: 4 passed (or skipped if torch not installed)

- [ ] **Step 8.5: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all pass

- [ ] **Step 8.6: Commit**

```bash
git add model/gru_model.py tests/model/test_gru_model.py
git commit -m "feat: add PyTorch GRU architecture for sequence-based sailing forecast"
```

---

### Task 9: train_gru.py + comparison report

**Files:**
- Create: `model/train_gru.py`

This script trains the GRU, compares it against the NWP-enriched RF, and writes a report.
No unit tests — it's a training pipeline script, validated by running successfully.

- [ ] **Step 9.1: Create `model/train_gru.py`**

```python
"""
model/train_gru.py — Train and evaluate the GRU sailing forecast model.

Compares GRU against the NWP-enriched RF on identical temporal folds.
Writes evaluation results to docs/gru_eval.md.

Usage:
    python model/train_gru.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from input.nwp_store import load_nwp_readings
from input.weather_store import load_weather_readings
from model.features import build_training_pairs
from model.features_sequence import build_sequence_training_pairs
from model.gru_model import SailingGRU
from utils.config import load_config

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "..", "config.toml")
WEIGHTS_PATH = os.path.join(_HERE, "weights_gru.pt")
REPORT_PATH  = os.path.join(_HERE, "..", "docs", "gru_eval.md")


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    """Balanced class weights as a 1-element tensor for BCELoss pos_weight."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def train_fold(
    seqs_tr: np.ndarray,
    ctxs_tr: np.ndarray,
    labs_tr: np.ndarray,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> SailingGRU:
    """Train a fresh GRU on one fold with early stopping."""
    model = SailingGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = _class_weights(labs_tr)

    seq_t = torch.tensor(seqs_tr, dtype=torch.float32)
    ctx_t = torch.tensor(ctxs_tr, dtype=torch.float32)
    lab_t = torch.tensor(labs_tr, dtype=torch.float32)

    n = len(seq_t)
    best_loss = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            out = model(seq_t[idx], ctx_t[idx]).squeeze(1)
            # Manual pos_weight scaling
            weights = torch.where(lab_t[idx] == 1, pos_weight, torch.ones(1))
            loss = (nn.BCELoss(reduction="none")(out, lab_t[idx]) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        epoch_loss /= n
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model: SailingGRU, seqs: np.ndarray, ctxs: np.ndarray,
             labs: np.ndarray) -> dict:
    """Compute metrics on a held-out set."""
    model.eval()
    with torch.no_grad():
        probs = model(
            torch.tensor(seqs, dtype=torch.float32),
            torch.tensor(ctxs, dtype=torch.float32),
        ).squeeze(1).numpy()

    preds = (probs >= 0.5).astype(int)
    metrics = {"roc_auc": roc_auc_score(labs, probs) if len(set(labs)) > 1 else float("nan")}
    for name, fn in [("precision", precision_score), ("recall", recall_score),
                     ("f1", f1_score)]:
        try:
            metrics[name] = fn(labs, preds, zero_division=0)
        except Exception:
            metrics[name] = float("nan")
    return metrics


def evaluate_rf(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """Cross-validate the NWP-enriched RF with temporal folds."""
    from sklearn.ensemble import RandomForestClassifier

    cfg = load_config(DEFAULT_CONFIG)
    mc = cfg["model"]
    clf = RandomForestClassifier(
        n_estimators=mc["n_estimators"],
        max_depth=mc["max_depth"],
        min_samples_leaf=mc["min_samples_leaf"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    X_filled = X.fillna(X.median(numeric_only=True))
    cv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    for tr, va in cv.split(X_filled):
        clf.fit(X_filled.iloc[tr], y.iloc[tr])
        probs = clf.predict_proba(X_filled.iloc[va])[:, 1]
        preds = (probs >= 0.5).astype(int)
        y_va = y.iloc[va].to_numpy()
        m = {"roc_auc": roc_auc_score(y_va, probs) if len(set(y_va)) > 1 else float("nan")}
        for name, fn in [("precision", precision_score), ("recall", recall_score),
                         ("f1", f1_score)]:
            m[name] = fn(y_va, preds, zero_division=0)
        fold_metrics.append(m)

    return {k: float(np.nanmean([m[k] for m in fold_metrics]))
            for k in fold_metrics[0]}


def write_report(gru_metrics: dict, rf_metrics: dict, n_samples: int) -> None:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = [
        "# GRU vs RF Evaluation Report\n",
        f"N training samples: {n_samples}  |  CV: 5-fold TimeSeriesSplit\n\n",
        "| Metric | RF (NWP-enriched) | GRU |",
        "|--------|-------------------|-----|",
    ]
    for k in ["roc_auc", "precision", "recall", "f1"]:
        rf_v  = rf_metrics.get(k, float("nan"))
        gru_v = gru_metrics.get(k, float("nan"))
        lines.append(f"| {k} | {rf_v:.3f} | {gru_v:.3f} |")

    verdict = (
        "**GRU shows ≥3% ROC-AUC improvement — candidate for production promotion.**"
        if gru_metrics.get("roc_auc", 0) - rf_metrics.get("roc_auc", 0) >= 0.03
        else "**RF remains the recommended production model.**"
    )
    lines += ["", verdict, ""]
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")


def main():
    cfg = load_config(DEFAULT_CONFIG)

    print("Loading station data …")
    df = load_weather_readings()
    print(f"  {len(df):,} rows  ({df.index.min().date()} → {df.index.max().date()})")

    print("Loading NWP data …")
    nwp_df = load_nwp_readings()
    if nwp_df.empty:
        print("  [!] No NWP data — context vector will be zeros")
        nwp_df = None
    else:
        print(f"  {len(nwp_df):,} NWP rows")

    # ── Build GRU training data ───────────────────────────────────────────────
    print("\nBuilding sequence training pairs …")
    sequences, contexts, labels = build_sequence_training_pairs(df, cfg, nwp_df=nwp_df)
    print(f"  Sequences: {sequences.shape}  Labels: {labels.shape}  "
          f"Good: {labels.sum()}/{len(labels)}")

    if len(sequences) < 10:
        print("ERROR: not enough training pairs")
        sys.exit(1)

    # ── GRU temporal CV ──────────────────────────────────────────────────────
    n_splits = min(5, int(labels.sum()))
    cv = TimeSeriesSplit(n_splits=n_splits)
    print(f"\nGRU temporal CV ({n_splits} folds) …")
    gru_fold_metrics = []
    for fold, (tr, va) in enumerate(cv.split(sequences)):
        print(f"  Fold {fold+1}/{n_splits}  train={len(tr)}  val={len(va)}", flush=True)
        model = train_fold(sequences[tr], contexts[tr], labels[tr])
        m = evaluate(model, sequences[va], contexts[va], labels[va])
        gru_fold_metrics.append(m)
        print(f"    ROC-AUC={m['roc_auc']:.3f}  F1={m['f1']:.3f}")

    gru_metrics = {k: float(np.nanmean([m[k] for m in gru_fold_metrics]))
                   for k in gru_fold_metrics[0]}
    print(f"\nGRU mean ROC-AUC: {gru_metrics['roc_auc']:.3f}")

    # ── RF baseline (same folds) ──────────────────────────────────────────────
    print("\nBuilding RF training pairs for comparison …")
    X, y = build_training_pairs(df, cfg, nwp_df=nwp_df)
    print(f"  RF pairs: {len(X)}")
    rf_metrics = evaluate_rf(X, y, n_splits=n_splits)
    print(f"RF mean ROC-AUC: {rf_metrics['roc_auc']:.3f}")

    # ── Train final GRU on all data + save weights ────────────────────────────
    print("\nTraining final GRU on all data …")
    final_model = train_fold(sequences, contexts, labels, epochs=200, patience=20)
    torch.save(final_model.state_dict(), WEIGHTS_PATH)
    print(f"Saved GRU weights: {WEIGHTS_PATH}")

    write_report(gru_metrics, rf_metrics, n_samples=len(sequences))

    print("\nSummary:")
    for k in ["roc_auc", "precision", "recall", "f1"]:
        print(f"  {k:12s}  RF={rf_metrics[k]:.3f}  GRU={gru_metrics[k]:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.2: Add `model/weights_gru.pt` to .gitignore**

```bash
echo "model/weights_gru.pt" >> .gitignore
```

- [ ] **Step 9.3: Run full test suite — no regressions**

```bash
python -m pytest tests/ -v
```

Expected: all pass

- [ ] **Step 9.4: Commit**

```bash
git add model/train_gru.py .gitignore
git commit -m "feat: add GRU training script with temporal CV and RF comparison report"
```

- [ ] **Step 9.5: Run GRU training experiment**

```bash
python model/train_gru.py
```

Expected:
- Loads station + NWP data
- Runs 5-fold temporal CV for both GRU and RF
- Prints side-by-side ROC-AUC comparison
- Saves `model/weights_gru.pt`
- Writes `docs/gru_eval.md`

- [ ] **Step 9.6: Commit report**

```bash
git add docs/gru_eval.md
git commit -m "docs: add GRU vs RF evaluation report"
```

---

## Final verification

- [ ] **Run full test suite one last time**

```bash
python -m pytest tests/ -v
```

Expected: all pass, no regressions.

- [ ] **Smoke-test the full local pipeline**

```bash
python deploy.py --no-download --no-stitch
```

Expected: predicts + renders without errors.
