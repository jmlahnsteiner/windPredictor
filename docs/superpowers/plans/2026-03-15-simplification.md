# windPredictor Simplification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all mutable state (weather readings, forecast snapshots, prediction history) to Supabase, deploy GitHub Pages via artifact instead of git commits, and clean up code duplication — eliminating all binary/conflict-prone files from the git repo.

**Architecture:** A shared `utils/` layer (config, db connection, circular maths) is introduced first. Then each pipeline stage (input → model → render → CI) is migrated to use the DB-backed stores. The git commit step in CI is replaced by `actions/deploy-pages`; `deploy.py` drops its publish step so local runs never push.

**Tech Stack:** Python 3.12+, pandas, scikit-learn/joblib, psycopg2-binary, python-dotenv, pytest, GitHub Actions (`actions/upload-pages-artifact@v3`, `actions/deploy-pages@v4`)

**Spec:** `docs/superpowers/specs/2026-03-15-simplification-design.md`

---

## Chunk 1: Foundation — utils layer + tests scaffold

### Task 1: Create tests scaffold and install python-dotenv

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/utils/__init__.py`
- Create: `tests/input/__init__.py`
- Create: `tests/model/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1.1: Add python-dotenv to requirements.txt**

Open `requirements.txt` and append:
```
python-dotenv==1.0.1
pytest==8.3.5
```

- [ ] **Step 1.2: Install new dependencies**

```bash
pip install python-dotenv==1.0.1 pytest==8.3.5
```
Expected: installs without error.

- [ ] **Step 1.3: Create test directory structure**

```bash
mkdir -p tests/utils tests/input tests/model
touch tests/__init__.py tests/utils/__init__.py tests/input/__init__.py tests/model/__init__.py
```

- [ ] **Step 1.4: Commit scaffold**

```bash
git add requirements.txt tests/
git commit -m "chore: add pytest + python-dotenv, create tests scaffold"
```

---

### Task 2: utils/circular.py

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/circular.py`
- Create: `tests/utils/test_circular.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/utils/test_circular.py`:
```python
import math
import numpy as np
import pandas as pd
import pytest
from utils.circular import circular_std


def test_perfectly_steady_direction():
    # All readings from the same direction → std ≈ 0
    angles = pd.Series([180.0] * 10)
    assert circular_std(angles) < 1.0


def test_opposite_directions_high_std():
    # N and S alternating → very high circular std
    angles = pd.Series([0.0, 180.0] * 5)
    assert circular_std(angles) > 80.0


def test_known_value():
    # 0°, 90°, 180°, 270° → mean resultant length R ≈ 0, std ≈ high
    angles = pd.Series([0.0, 90.0, 180.0, 270.0])
    result = circular_std(angles)
    assert result > 100.0


def test_accepts_list():
    result = circular_std([10.0, 20.0, 30.0])
    assert isinstance(result, float)
    assert result < 15.0


def test_single_value_returns_zero():
    result = circular_std([45.0])
    assert result == pytest.approx(0.0, abs=1e-6)


def test_empty_series_returns_nan():
    result = circular_std(pd.Series([], dtype=float))
    assert math.isnan(result)


def test_series_with_nans_ignored():
    angles = pd.Series([90.0, float('nan'), 90.0, 90.0])
    result = circular_std(angles)
    assert result < 1.0
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
cd /Users/macstudio/Code/windPredictor && python -m pytest tests/utils/test_circular.py -v
```
Expected: `ModuleNotFoundError: No module named 'utils'`

- [ ] **Step 2.3: Create utils/__init__.py and utils/circular.py**

`utils/__init__.py` — empty file.

`utils/circular.py`:
```python
"""utils/circular.py — Circular statistics for wind direction data."""
import math

import numpy as np
import pandas as pd


def circular_std(angles) -> float:
    """
    Circular standard deviation of wind directions (degrees).

    Accepts a pd.Series or list of degree values. NaN values are dropped.
    Returns NaN if no valid values remain.
    """
    if not isinstance(angles, pd.Series):
        angles = pd.Series(angles)
    clean = angles.dropna()
    if clean.empty:
        return float("nan")
    if len(clean) == 1:
        return 0.0
    rad = np.radians(clean)
    R = np.hypot(np.sin(rad).mean(), np.cos(rad).mean())
    return float(np.degrees(np.sqrt(-2 * np.log(np.clip(R, 1e-9, 1)))))
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
python -m pytest tests/utils/test_circular.py -v
```
Expected: 7 tests PASSED.

- [ ] **Step 2.5: Commit**

```bash
git add utils/__init__.py utils/circular.py tests/utils/test_circular.py
git commit -m "feat: add utils/circular.py with circular_std"
```

---

### Task 3: utils/config.py

**Files:**
- Create: `utils/config.py`
- Create: `tests/utils/test_config.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/utils/test_config.py`:
```python
import os
import tempfile
import textwrap
from pathlib import Path
from utils.config import load_config


def _write_toml(content: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False)
    f.write(textwrap.dedent(content))
    f.flush()
    return f.name


def test_loads_toml_values():
    path = _write_toml("""
        [sailing]
        window_start = "08:00"
        window_end   = "16:00"
        wind_speed_min = 2.0
    """)
    cfg = load_config(path)
    assert cfg["sailing"]["window_start"] == "08:00"
    assert cfg["sailing"]["wind_speed_min"] == 2.0


def test_returns_dict():
    path = _write_toml("[model]\nn_estimators = 300\n")
    cfg = load_config(path)
    assert isinstance(cfg, dict)


def test_loads_project_config():
    """Smoke test: project config.toml loads without error."""
    cfg = load_config()
    assert "sailing" in cfg
    assert "prediction" in cfg
```

- [ ] **Step 3.2: Run test to verify it fails**

```bash
python -m pytest tests/utils/test_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'utils.config'`

- [ ] **Step 3.3: Implement utils/config.py**

```python
"""utils/config.py — Single config loader for the whole project."""
import tomllib
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def load_config(path: str | None = None) -> dict:
    """
    Load config.toml. Also loads .env from the project root if python-dotenv
    is installed (silently skipped if not).

    Parameters
    ----------
    path : str | None
        Explicit path to a .toml file. Defaults to <project_root>/config.toml.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    if path is None:
        path = str(_ROOT / "config.toml")

    with open(path, "rb") as f:
        return tomllib.load(f)
```

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
python -m pytest tests/utils/test_config.py -v
```
Expected: 3 tests PASSED.

- [ ] **Step 3.5: Commit**

```bash
git add utils/config.py tests/utils/test_config.py
git commit -m "feat: add utils/config.py with load_config"
```

---

### Task 4: utils/db.py

**Files:**
- Create: `utils/db.py`
- Create: `tests/utils/test_db.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/utils/test_db.py`:
```python
import os
import tempfile
import sqlite3
import pytest
from unittest.mock import patch


def test_backend_returns_sqlite_when_no_env():
    with patch.dict(os.environ, {}, clear=True):
        # Remove SUPABASE_DB_URL if present
        env = {k: v for k, v in os.environ.items() if k != "SUPABASE_DB_URL"}
        with patch.dict(os.environ, env, clear=True):
            from utils.db import backend
            assert backend() == "sqlite"


def test_backend_returns_postgres_when_env_set():
    with patch.dict(os.environ, {"SUPABASE_DB_URL": "postgresql://fake"}):
        from utils import db
        import importlib
        importlib.reload(db)  # re-evaluate module-level
        from utils.db import backend
        assert backend() == "postgres"


def test_placeholder_sqlite():
    from utils.db import placeholder
    assert placeholder("sqlite") == "?"


def test_placeholder_postgres():
    from utils.db import placeholder
    assert placeholder("postgres") == "%s"


def test_get_connection_sqlite_creates_file():
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test.db")
        with patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                      if k != "SUPABASE_DB_URL"}, clear=True):
            from utils.db import get_connection
            con, bk = get_connection(db_path)
            assert bk == "sqlite"
            assert isinstance(con, sqlite3.Connection)
            con.close()
        assert os.path.exists(db_path)
```

- [ ] **Step 4.2: Run test to verify it fails**

```bash
python -m pytest tests/utils/test_db.py -v
```
Expected: import error or attribute error.

- [ ] **Step 4.3: Implement utils/db.py**

```python
"""utils/db.py — Shared database connection helpers (Supabase / SQLite fallback)."""
import os
import sqlite3
from pathlib import Path

_ROOT = Path(__file__).parent.parent
DEFAULT_SQLITE = str(_ROOT / "local.db")


def backend() -> str:
    """Return 'postgres' if SUPABASE_DB_URL is set, else 'sqlite'."""
    return "postgres" if os.environ.get("SUPABASE_DB_URL") else "sqlite"


def placeholder(bk: str) -> str:
    """SQL parameter placeholder for the given backend."""
    return "%s" if bk == "postgres" else "?"


def get_connection(db_path: str = DEFAULT_SQLITE):
    """
    Return (connection, backend_string).

    db_path is only used for the SQLite backend; Postgres reads from
    the SUPABASE_DB_URL environment variable.
    """
    bk = backend()
    if bk == "postgres":
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "psycopg2-binary is required for the Supabase backend. "
                "Run: pip install psycopg2-binary"
            ) from e
        return psycopg2.connect(os.environ["SUPABASE_DB_URL"]), "postgres"

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    return sqlite3.connect(db_path), "sqlite"
```

- [ ] **Step 4.4: Run tests to verify they pass**

```bash
python -m pytest tests/utils/test_db.py -v
```
Expected: 5 tests PASSED.

- [ ] **Step 4.5: Commit**

```bash
git add utils/db.py tests/utils/test_db.py
git commit -m "feat: add utils/db.py with shared connection helpers"
```

---

## Chunk 2: Supabase schemas + weather_store

### Task 5: Add new Supabase schemas

**Files:**
- Create: `supabase/schema_additions.sql`
- Modify: `supabase/schema.sql` (append the new tables for reference)

- [ ] **Step 5.1: Create supabase/schema_additions.sql**

```sql
-- windPredictor v2 — run once in Supabase SQL Editor after schema.sql

CREATE TABLE IF NOT EXISTS weather_readings (
    timestamp           TEXT PRIMARY KEY,  -- ISO-8601 local naive, e.g. "2026-03-15T08:05:00"
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

CREATE TABLE IF NOT EXISTS forecast_snapshots (
    snapshot        TEXT NOT NULL,      -- ISO-8601 datetime of prediction run
    predicting_date TEXT NOT NULL,      -- YYYY-MM-DD target date
    payload         TEXT NOT NULL,      -- JSON blob (the full prediction dict)
    PRIMARY KEY (snapshot, predicting_date)
);
CREATE INDEX IF NOT EXISTS idx_fs_date ON forecast_snapshots(predicting_date);
```

- [ ] **Step 5.2: Run schema_additions.sql in Supabase**

Open the Supabase dashboard → SQL Editor → New query, paste the contents of `supabase/schema_additions.sql`, and run it. Verify both tables are created.

- [ ] **Step 5.3: Commit**

```bash
git add supabase/schema_additions.sql
git commit -m "feat: add weather_readings + forecast_snapshots Supabase schema"
```

---

### Task 6: input/weather_store.py

**Files:**
- Create: `input/weather_store.py`
- Create: `tests/input/test_weather_store.py`

- [ ] **Step 6.1: Write the failing tests**

Create `tests/input/test_weather_store.py`:
```python
import os
import tempfile
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest


def _make_df(n=5) -> pd.DataFrame:
    """Create a minimal weather DataFrame matching the parquet schema."""
    import numpy as np
    idx = pd.date_range("2026-03-10 08:00", periods=n, freq="5min")
    return pd.DataFrame({
        "wind_speed":       [3.0, 4.0, 5.0, 4.5, 3.5][:n],
        "wind_direction":   [180.0] * n,
        "wind_gust":        [5.0] * n,
        "temperature":      [15.0] * n,
        "humidity":         [60.0] * n,
        "pressure_relative":[1013.0] * n,
        "water_temperature":[12.0] * n,
    }, index=idx)


def _no_supabase_env():
    return {k: v for k, v in os.environ.items() if k != "SUPABASE_DB_URL"}


def test_upsert_and_load_roundtrip():
    """Upserted rows come back with correct types and index."""
    df = _make_df()
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase_env(), clear=True):
            from input.weather_store import upsert_readings, load_weather_readings
            n = upsert_readings(df, db_path=db)
            assert n == 5

            result = load_weather_readings(db_path=db)
            assert len(result) == 5
            assert isinstance(result.index, pd.DatetimeIndex)
            assert result["wind_speed"].iloc[0] == pytest.approx(3.0)


def test_upsert_is_idempotent():
    """Upserting the same rows twice doesn't create duplicates."""
    df = _make_df()
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase_env(), clear=True):
            from input.weather_store import upsert_readings, load_weather_readings
            upsert_readings(df, db_path=db)
            upsert_readings(df, db_path=db)
            result = load_weather_readings(db_path=db)
            assert len(result) == 5


def test_date_range_filtering():
    """load_weather_readings respects start/end date bounds."""
    df = _make_df(5)
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase_env(), clear=True):
            from input.weather_store import upsert_readings, load_weather_readings
            upsert_readings(df, db_path=db)
            result = load_weather_readings(
                start=date(2026, 3, 11),   # after all data
                db_path=db,
            )
            assert result.empty


def test_empty_df_returns_empty():
    """Loading from a nonexistent DB returns empty DataFrame."""
    with patch.dict(os.environ, _no_supabase_env(), clear=True):
        from input.weather_store import load_weather_readings
        result = load_weather_readings(db_path="/tmp/nonexistent_test.db")
        assert result.empty
        assert isinstance(result, pd.DataFrame)


def test_load_all_when_no_bounds():
    """None bounds load all available rows."""
    df = _make_df(5)
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase_env(), clear=True):
            from input.weather_store import upsert_readings, load_weather_readings
            upsert_readings(df, db_path=db)
            result = load_weather_readings(db_path=db)
            assert len(result) == 5
```

- [ ] **Step 6.2: Run tests to verify they fail**

```bash
python -m pytest tests/input/test_weather_store.py -v
```
Expected: `ModuleNotFoundError: No module named 'input.weather_store'`

- [ ] **Step 6.3: Implement input/weather_store.py**

```python
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

    try:
        con, bk = get_connection(db_path)
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
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    for col in _COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
```

- [ ] **Step 6.4: Run tests to verify they pass**

```bash
python -m pytest tests/input/test_weather_store.py -v
```
Expected: 5 tests PASSED.

- [ ] **Step 6.5: Commit**

```bash
git add input/weather_store.py tests/input/test_weather_store.py supabase/schema_additions.sql
git commit -m "feat: add input/weather_store.py with upsert/load for weather_readings"
```

---

### Task 7: supabase/migrate_parquet.py

**Files:**
- Create: `supabase/migrate_parquet.py`

No automated test (one-time migration script). A dry-run check is built in.

- [ ] **Step 7.1: Create supabase/migrate_parquet.py**

```python
"""
supabase/migrate_parquet.py — One-time migration of data.parquet → weather_readings.

Prerequisites:
  1. Run supabase/schema_additions.sql in Supabase
  2. Set SUPABASE_DB_URL in your environment (or .env file)

Usage:
    python supabase/migrate_parquet.py
    python supabase/migrate_parquet.py --dry-run   # count rows only
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

PARQUET_PATH = os.path.join(_ROOT, "data.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate data.parquet → weather_readings")
    parser.add_argument("--dry-run", action="store_true", help="Count rows only, do not write")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_ROOT, ".env"))
    except ImportError:
        pass

    if not os.path.exists(PARQUET_PATH):
        print(f"ERROR: {PARQUET_PATH} not found.")
        sys.exit(1)

    import pandas as pd
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df):,} rows  ({df.index.min().date()} → {df.index.max().date()})")

    if args.dry_run:
        print(f"[dry-run] Would upsert {len(df):,} rows into weather_readings. Exiting.")
        return

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        print("ERROR: SUPABASE_DB_URL is not set. Add it to .env or export it.")
        sys.exit(1)

    from input.weather_store import upsert_readings
    print("Upserting rows into weather_readings (this may take a minute) …")
    n = upsert_readings(df)
    print(f"Done: {n:,} rows upserted.")
    print("\nNext steps:")
    print("  git rm --cached data.parquet")
    print("  echo 'data.parquet' >> .gitignore")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Dry-run to verify it loads the parquet**

```bash
python supabase/migrate_parquet.py --dry-run
```
Expected output: `Loaded 40,xxx rows  (YYYY-MM-DD → YYYY-MM-DD)` then `[dry-run] Would upsert ...`

- [ ] **Step 7.3: Run the actual migration**

```bash
python supabase/migrate_parquet.py
```
Expected: `Done: 40,xxx rows upserted.`

- [ ] **Step 7.4: Verify in Supabase dashboard**

Open Supabase → Table Editor → `weather_readings`. Confirm row count matches.

- [ ] **Step 7.5: Commit**

```bash
git add supabase/migrate_parquet.py
git commit -m "feat: add migrate_parquet.py for one-time data migration"
```

---

## Chunk 3: Input layer — config, scraper, stitcher

### Task 8: Fix config.toml

**Files:**
- Modify: `config.toml`

- [ ] **Step 8.1: Fix [location] section and add [ecowitt]**

Replace the bottom of `config.toml` (the broken location block) with:

```toml
[location]
# Coordinates for NWP forecast enrichment via Open-Meteo (no API key needed).
name = "Traunsee"
lat  = 47.8071    # decimal degrees, positive = North
lon  = 13.7790    # decimal degrees, positive = East

[ecowitt]
# Ecowitt weather station credentials.
# Override with ECOWITT_DEVICE_ID / ECOWITT_AUTHORIZE env vars (or .env).
device_id = "cHF5MGpPMzZFeDEvbFAvL2J6QjBWdz09"
authorize = "E6K45F"
```

- [ ] **Step 8.2: Verify config loads without error**

```bash
python -c "from utils.config import load_config; c = load_config(); print(c.get('location')); print(c.get('ecowitt'))"
```
Expected: both dicts printed with values (not None).

- [ ] **Step 8.3: Commit**

```bash
git add config.toml
git commit -m "fix: restore [location] section and add [ecowitt] section to config.toml"
```

---

### Task 9: Create .env.example and .env

**Files:**
- Create: `.env.example`
- Create: `.env` (gitignored, not committed)
- Modify: `.gitignore`

- [ ] **Step 9.1: Create .env.example**

```bash
cat > .env.example << 'EOF'
# Copy this file to .env and fill in your values.
# .env is gitignored — never commit real credentials.

SUPABASE_DB_URL=postgresql://postgres.xxxx:PASSWORD@aws-0-eu-west-1.pooler.supabase.com:5432/postgres
ECOWITT_DEVICE_ID=your_device_id_here
ECOWITT_AUTHORIZE=your_authorize_code_here
EOF
```

- [ ] **Step 9.2: Add .env to .gitignore**

Open `.gitignore` and append:
```
.env
local.db
index.html
```

- [ ] **Step 9.3: Commit**

```bash
git add .env.example .gitignore
git commit -m "chore: add .env.example and update .gitignore"
```

---

### Task 10: input/scraper.py — remove hardcoded credentials

**Files:**
- Modify: `input/scraper.py`

- [ ] **Step 10.1: Replace the entire contents of input/scraper.py**

Replace the full file (keeping `download_range`, `last_week`, and `__main__` unchanged but replacing everything above them):

```python
"""
scraper.py — Download daily Ecowitt weather xlsx files.

Usage:
    python scraper.py                          # last 7 days
    python scraper.py 2026-03-01              # single date
    python scraper.py 2026-02-01 2026-03-06   # date range

Credentials are read from (in priority order):
  1. Environment variables: ECOWITT_DEVICE_ID, ECOWITT_AUTHORIZE
  2. config.toml [ecowitt] section
  3. .env file (loaded automatically via utils/config.py)
"""

import os
import sys
import time
from datetime import date, datetime, timedelta

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from utils.config import load_config

_SORT_LIST = "0|1|2|49|4|5|32"   # station-specific sensor selection, not a credential

DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "downloaded_files")


def _credentials() -> tuple[str, str]:
    """Return (device_id, authorize) from env vars or config.toml."""
    cfg = load_config()
    ecowitt = cfg.get("ecowitt", {})
    device_id = os.environ.get("ECOWITT_DEVICE_ID") or ecowitt.get("device_id", "")
    authorize = os.environ.get("ECOWITT_AUTHORIZE") or ecowitt.get("authorize", "")
    if not device_id or not authorize:
        raise RuntimeError(
            "Ecowitt credentials not found. Set ECOWITT_DEVICE_ID and "
            "ECOWITT_AUTHORIZE in .env or config.toml [ecowitt]."
        )
    return device_id, authorize


def _make_headers(device_id: str, authorize: str) -> dict:
    referer = (
        f"https://www.ecowitt.net/home/share?authorize={authorize}"
        f"&device_id={device_id}&units=1,3,7,12,16,24"
    )
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
        ),
        "Referer": referer,
        "Content-Type": "application/x-www-form-urlencoded",
    }


def download_date(
    date_str: str,
    session: requests.Session,
    output_dir: str,
    edate: str | None = None,
) -> bool:
    """Download weather data for a single date (YYYY-MM-DD). Returns True on success."""
    edate = edate or "23:59"
    device_id, authorize = _credentials()
    headers = _make_headers(device_id, authorize)

    ts = int(time.time() * 1000)
    resp = session.post(
        f"https://www.ecowitt.net/index/export_excel?time={ts}",
        data={
            "device_id": device_id,
            "authorize": authorize,
            "mode": "0",
            "sdate": f"{date_str} 00:00",
            "edate": f"{date_str} {edate}",
            "sortList": _SORT_LIST,
            "hideList": "",
        },
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()

    try:
        data = resp.json()
        xlsx_url = data["url"]
    except Exception:
        date_compact = date_str.replace("-", "")
        edate_compact = edate.replace(":", "")
        xlsx_url = (
            f"https://www.ecowitt.net/uploads/156707/"
            f"Wetterstation%28{date_compact}0000-{date_compact}{edate_compact}%29.xlsx"
        )

    resp = session.get(xlsx_url, headers=headers, timeout=15)
    resp.raise_for_status()

    if resp.content[:2] != b"PK":
        print(f"  [!] {date_str}: not a valid xlsx ({len(resp.content)} bytes)")
        return False

    out_path = os.path.join(output_dir, f"Wetterstation_{date_str}.xlsx")
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"  [+] {date_str}: {len(resp.content):,} bytes → {out_path}")
    return True


def download_range(
    start: date,
    end: date,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    skip_existing: bool = True,
    force_dates: set | None = None,
    delay: float = 1.0,
) -> dict[str, bool]:
    """
    Download one xlsx per day for [start, end] inclusive.
    Skips dates that already have a file unless they appear in force_dates.
    Returns {date_str: success} for every date in the range.
    """
    os.makedirs(output_dir, exist_ok=True)
    force_dates = force_dates or set()
    results: dict[str, bool] = {}
    session = requests.Session()

    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        out_path = os.path.join(output_dir, f"Wetterstation_{date_str}.xlsx")
        forced = current in force_dates

        if skip_existing and not forced and os.path.exists(out_path):
            print(f"  [=] {date_str}: already exists, skipping")
            results[date_str] = True
            current += timedelta(days=1)
            continue

        try:
            edate = datetime.now().strftime("%H:%M") if forced else None
            results[date_str] = download_date(date_str, session, output_dir, edate=edate)
        except requests.exceptions.RequestException as e:
            print(f"  [!] {date_str}: {e}")
            results[date_str] = False

        time.sleep(delay)
        current += timedelta(days=1)

    return results


def last_week() -> tuple[date, date]:
    """Return (start, end) for the 7 complete days ending yesterday."""
    yesterday = date.today() - timedelta(days=1)
    return yesterday - timedelta(days=6), yesterday


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        start, end = last_week()
    elif len(args) == 1:
        start = end = datetime.strptime(args[0], "%Y-%m-%d").date()
    elif len(args) == 2:
        start = datetime.strptime(args[0], "%Y-%m-%d").date()
        end   = datetime.strptime(args[1], "%Y-%m-%d").date()
    else:
        print("Usage: python scraper.py [start_date [end_date]]")
        sys.exit(1)

    print(f"Downloading {start} → {end}  ({(end - start).days + 1} day(s))")
    results = download_range(start, end)
    n_ok = sum(results.values())
    print(f"\nDone: {n_ok}/{len(results)} downloaded successfully")
```

- [ ] **Step 10.2: Verify config loads credentials**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from input.scraper import _credentials
d, a = _credentials()
print('device_id:', d[:8] + '...')
print('authorize:', a)
"
```
Expected: credentials printed (not empty).

- [ ] **Step 10.3: Commit**

```bash
git add input/scraper.py
git commit -m "refactor: move Ecowitt credentials from scraper.py to config.toml/env"
```

---

### Task 11: input/stitcher.py — write to DB instead of parquet

**Files:**
- Modify: `input/stitcher.py`

- [ ] **Step 11.1: Replace parquet write with DB upsert**

Replace the `stitch()` function with `stitch_to_db()`. Keep `parse_xlsx()` and `_flatten_columns()` unchanged. Replace the `__main__` block to call `stitch_to_db()`.

The full new `stitch_to_db()` function (replaces `stitch()`):

```python
def stitch_to_db(
    input_dir: str = DEFAULT_INPUT_DIR,
    db_path: str | None = None,
) -> int:
    """
    Parse all xlsx files in input_dir and upsert rows into weather_readings.

    Re-running is safe: existing rows are replaced (upsert by timestamp).
    Returns number of rows upserted.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(_HERE))
    from input.weather_store import upsert_readings

    paths = sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
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
```

Also add the import at the top of the file if missing:
```python
import glob
import os
import sys
import pandas as pd
```

Replace the `__main__` block:
```python
if __name__ == "__main__":
    args = sys.argv[1:]
    input_dir = args[0] if args else DEFAULT_INPUT_DIR
    print(f"Input: {input_dir}\n")
    stitch_to_db(input_dir)
```

- [ ] **Step 11.2: Verify stitcher still parses xlsx (smoke test)**

```bash
python -c "
from input.stitcher import parse_xlsx
import glob, os
files = glob.glob('input/downloaded_files/*.xlsx')
if files:
    df = parse_xlsx(files[0])
    print('Parsed:', len(df), 'rows' if df is not None else 'None')
else:
    print('No xlsx files to test')
"
```

- [ ] **Step 11.3: Commit**

```bash
git add input/stitcher.py
git commit -m "refactor: stitcher writes to weather_readings DB instead of parquet"
```

---

## Chunk 4: Model layer — history, train, predict

### Task 12: model/history.py — delegate to utils/db.py

**Files:**
- Modify: `model/history.py`

- [ ] **Step 12.1: Replace internal connection helpers with utils/db.py**

Replace lines 38–97 (the `_backend`, `_SQLITE_SCHEMA`, `_connect_sqlite`, `_connect_postgres`, `_connect`, `_ph` functions) with:

```python
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from utils.db import DEFAULT_SQLITE, backend as _backend_fn, get_connection, placeholder

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts          TEXT    NOT NULL,
    snapshot_dt     TEXT    NOT NULL,
    predicting_date TEXT    NOT NULL,
    probability     REAL    NOT NULL,
    good            INTEGER NOT NULL,
    threshold       REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    predicting_date TEXT    PRIMARY KEY,
    actual_good     INTEGER,
    actual_frac     REAL
);
CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(predicting_date);
CREATE INDEX IF NOT EXISTS idx_run_ts    ON predictions(run_ts);
"""


def _backend() -> str:
    return _backend_fn()


def _connect(db_path: str = DEFAULT_SQLITE):
    """Return (connection, backend). Creates SQLite schema if needed."""
    con, bk = get_connection(db_path)
    if bk == "sqlite":
        con.executescript(_SQLITE_SCHEMA)
        con.commit()
    return con, bk


def _ph(bk: str) -> str:
    return placeholder(bk)
```

Then update the `db_path` default in every public function signature from `"predictions.db"` to `DEFAULT_SQLITE` so that the SQLite fallback path writes to `local.db`:

```python
def record_predictions(results: list[dict], db_path: str = DEFAULT_SQLITE) -> int:
def record_outcome(predicting_date, actual_good, actual_frac, db_path: str = DEFAULT_SQLITE):
def backfill_outcomes(daily_quality, db_path: str = DEFAULT_SQLITE) -> int:
def load_history(db_path: str = DEFAULT_SQLITE, days=30, snapshot_hour=None):
def accuracy_summary(db_path: str = DEFAULT_SQLITE, days: int = 30) -> dict:
```

The function bodies remain unchanged — they all call `_connect(db_path)` and `_ph(backend)` which are preserved.

- [ ] **Step 12.2: Smoke-test history.py still works with SQLite**

```bash
python -c "
import os, tempfile
os.environ.pop('SUPABASE_DB_URL', None)
from model.history import record_predictions, accuracy_summary
results = [{'snapshot': '2026-01-01T08:00', 'predicting_date': '2026-01-01', 'probability': 0.7, 'good': True, 'threshold': 0.3}]
with tempfile.TemporaryDirectory() as d:
    db = d + '/test.db'
    n = record_predictions(results, db)
    print('Recorded:', n)
    s = accuracy_summary(db)
    print('Summary:', s)
"
```
Expected: `Recorded: 1`, `Summary: {}` (no outcomes yet).

- [ ] **Step 12.3: Commit**

```bash
git add model/history.py
git commit -m "refactor: model/history.py delegates connection helpers to utils/db"
```

---

### Task 13: model/train.py — use weather_store + utils/config

**Files:**
- Modify: `model/train.py`

- [ ] **Step 13.1: Replace parquet load and load_config**

At the top of `model/train.py`, replace the existing `load_config` function and `_HERE`/path setup with:

```python
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.features import build_training_pairs
from utils.config import load_config

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "..", "config.toml")
```

Then in `train()`, replace:
```python
    parquet_path = os.path.join(root, cfg["paths"]["data_parquet"])
    df = pd.read_parquet(parquet_path)
```
with:
```python
    from input.weather_store import load_weather_readings
    print("Loading all weather readings from database …")
    df = load_weather_readings()
    if df.empty:
        print("ERROR: No weather data found. Run the scraper and stitch first.")
        return
```

Remove the now-unused `load_config` function defined in the file (leave only the `train` and `__main__` logic).

- [ ] **Step 13.2: Smoke-test train.py import (don't actually train)**

```bash
python -c "import model.train; print('train.py imports OK')"
```

- [ ] **Step 13.3: Commit**

```bash
git add model/train.py
git commit -m "refactor: model/train.py loads weather data from DB instead of parquet"
```

---

### Task 14: model/predict.py — weather_store + Supabase snapshots + utils

**Files:**
- Modify: `model/predict.py`

This is the most significant model change. Four things change:
1. `load_config` → `utils/config.py`
2. `_circular_std` (used inside `_condition_rating`) → `utils/circular.py`
3. `pd.read_parquet` → `load_weather_readings`
4. `merge_predictions` (file-based) → `save_forecast_snapshots` + `load_forecast_snapshots` (DB-based)

The new snapshot schema needs to be ensured on first run (SQLite only; Postgres expects it from `schema_additions.sql`).

- [ ] **Step 14.1: Update imports at the top of model/predict.py**

Replace the existing import block with:

```python
import json
import math
import os
import sys
from datetime import date, datetime, timedelta
from typing import Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.features import extract_snapshot_features, _target_date
from utils.config import load_config
from utils.circular import circular_std
from utils.db import DEFAULT_SQLITE, backend, get_connection, placeholder
```

- [ ] **Step 14.2: Update _condition_rating to use utils/circular.py**

In `_condition_rating`, find:
```python
        if len(dirs) >= 3:
            rad = [math.radians(d) for d in dirs]
            R   = math.hypot(
                sum(math.sin(r) for r in rad) / len(rad),
                sum(math.cos(r) for r in rad) / len(rad),
            )
            circ_std   = math.degrees(math.sqrt(-2 * math.log(max(R, 1e-9))))
            dir_factor = max(0.0, 1.0 - circ_std / dir_max)
```
Replace with:
```python
        if len(dirs) >= 3:
            circ_std   = circular_std(dirs)
            dir_factor = max(0.0, 1.0 - circ_std / dir_max)
```

- [ ] **Step 14.3: Replace load_config (the local copy) with the import**

Remove the local `load_config` function definition (lines ~119–122 in the original). The `from utils.config import load_config` import at the top takes over. Update `DEFAULT_CONFIG`:
```python
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "..", "config.toml")
```

- [ ] **Step 14.4: Replace pd.read_parquet in __main__ with weather_store**

In the `if __name__ == "__main__":` block, replace:
```python
    df = pd.read_parquet(os.path.join(root, cfg["paths"]["data_parquet"]))
```
with:
```python
    from input.weather_store import load_weather_readings
    df = load_weather_readings(
        start=date.today() - timedelta(days=35),
        end=date.today(),
    )
    if df.empty:
        print("ERROR: No weather data in DB. Run scraper + stitch first.")
        sys.exit(1)
```

- [ ] **Step 14.5: Add snapshot DB helpers and replace merge_predictions**

Add these two functions **before** `merge_predictions` (or replace it entirely):

```python
_SNAPSHOT_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_snapshots (
    snapshot        TEXT NOT NULL,
    predicting_date TEXT NOT NULL,
    payload         TEXT NOT NULL,
    PRIMARY KEY (snapshot, predicting_date)
);
CREATE INDEX IF NOT EXISTS idx_fs_date ON forecast_snapshots(predicting_date);
"""


def _ensure_snapshot_schema(con, bk: str) -> None:
    if bk == "sqlite":
        con.executescript(_SNAPSHOT_SQLITE_SCHEMA)
        con.commit()
    else:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forecast_snapshots (
                snapshot TEXT NOT NULL, predicting_date TEXT NOT NULL,
                payload TEXT NOT NULL, PRIMARY KEY (snapshot, predicting_date)
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fs_date ON forecast_snapshots(predicting_date)"
        )
        con.commit()


def save_forecast_snapshots(
    results: list[dict],
    days_to_keep: int = 7,
    db_path: str = DEFAULT_SQLITE,
) -> None:
    """
    Upsert new forecast entries into forecast_snapshots and prune old ones.
    Replaces the file-based merge_predictions().
    """
    if not results:
        return

    con, bk = get_connection(db_path)
    _ensure_snapshot_schema(con, bk)
    ph = placeholder(bk)

    if bk == "postgres":
        upsert_sql = (
            f"INSERT INTO forecast_snapshots (snapshot, predicting_date, payload) "
            f"VALUES ({ph}, {ph}, {ph}) "
            f"ON CONFLICT (snapshot, predicting_date) DO UPDATE SET payload = EXCLUDED.payload"
        )
    else:
        upsert_sql = (
            f"INSERT OR REPLACE INTO forecast_snapshots (snapshot, predicting_date, payload) "
            f"VALUES ({ph}, {ph}, {ph})"
        )

    cutoff = (date.today() - timedelta(days=days_to_keep)).isoformat()

    try:
        cur = con.cursor()
        rows = [
            (r.get("snapshot", ""), r["predicting_date"], json.dumps(r))
            for r in results
            if "predicting_date" in r
        ]
        cur.executemany(upsert_sql, rows)
        cur.execute(f"DELETE FROM forecast_snapshots WHERE predicting_date < {ph}", (cutoff,))
        con.commit()
    finally:
        con.close()


def load_forecast_snapshots(
    days: int = 7,
    db_path: str = DEFAULT_SQLITE,
) -> list[dict]:
    """
    Load the rolling forecast window from the DB.
    Returns list[dict] — same type build_html() accepts.
    """
    bk = backend()

    if bk == "sqlite":
        import sqlite3 as _sqlite3
        if not os.path.exists(db_path):
            return []
        con = _sqlite3.connect(db_path)
        con.executescript(_SNAPSHOT_SQLITE_SCHEMA)
        con.commit()
    else:
        con, _ = get_connection(db_path)
        _ensure_snapshot_schema(con, "postgres")

    ph = placeholder(bk)
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    try:
        cur = con.cursor()
        cur.execute(
            f"SELECT payload FROM forecast_snapshots WHERE predicting_date >= {ph} "
            f"ORDER BY predicting_date, snapshot",
            (cutoff,),
        )
        rows = cur.fetchall()
    finally:
        con.close()

    return [json.loads(row[0]) for row in rows]
```

- [ ] **Step 14.6: Update __main__ to use save_forecast_snapshots instead of merge_predictions**

Replace:
```python
    merged = merge_predictions(results, pred_path)
    n_days = len({r["predicting_date"] for r in merged if "predicting_date" in r})
    print(f"Saved: {pred_path}  ({len(merged)} entries, {n_days} day(s))", flush=True)
```
with:
```python
    save_forecast_snapshots(results)
    snapshots = load_forecast_snapshots()
    n_days = len({r["predicting_date"] for r in snapshots if "predicting_date" in r})
    print(f"Snapshots saved to DB  ({len(snapshots)} entries, {n_days} day(s))", flush=True)
```

Also remove the `cfg["paths"]["predictions_file"]` / `pred_path` lines that are no longer needed.

- [ ] **Step 14.7: Smoke-test predict.py imports**

```bash
python -c "import model.predict; print('predict.py imports OK')"
```

- [ ] **Step 14.8: Commit**

```bash
git add model/predict.py
git commit -m "refactor: predict.py reads DB for weather data and writes forecast snapshots to DB"
```

---

## Chunk 5: Render layer — split render_html.py

### Task 15: Extract render/charts.py and render/data.py

**Files:**
- Create: `render/__init__.py`
- Create: `render/charts.py`
- Create: `render/data.py`
- Modify: `render_html.py`

- [ ] **Step 15.1: Create render/__init__.py (empty)**

```bash
mkdir -p render && touch render/__init__.py
```

- [ ] **Step 15.2: Create render/charts.py**

Extract `_prob_trend_svg` and `_wind_svg` from `render_html.py` into `render/charts.py`. Neither function uses `circular_std` — do not import it here.

```python
"""render/charts.py — SVG chart generators for the forecast page."""
import math
from datetime import datetime


def prob_trend_svg(snaps: list[dict]) -> str:
    """
    Compact sparkline showing how sailing probability evolved across snapshots.
    Returns '' when fewer than 2 snapshots exist.
    (Previously _prob_trend_svg in render_html.py)
    """
    # [paste the full body of _prob_trend_svg here, unchanged except function name]
    ...


def wind_svg(window_wind: dict, cfg: dict) -> str:
    """
    Two-panel SVG: wind speed time series + compass rose.
    Returns '' when window data is unavailable.
    (Previously _wind_svg in render_html.py)
    """
    # [paste the full body of _wind_svg here, unchanged except function name]
    ...
```

Full implementations: copy the bodies of `_prob_trend_svg` and `_wind_svg` from `render_html.py` verbatim, renaming them `prob_trend_svg` and `wind_svg`.

- [ ] **Step 15.3: Create render/data.py**

Extract `_window_stats`, `_expected_wind_chips`, `_stats_html`, `_history_html` into `render/data.py`. Also **move** `_score_to_hex` from `render_html.py` into this module — `history_html` calls it, so it must be co-located.

```python
"""render/data.py — Data helpers and history rendering for the forecast page."""
import math
import os

import numpy as np

from utils.circular import circular_std


def score_to_hex(score: int) -> str:
    """Map a 0-100 condition score to a hex colour. (Moved from render_html.py)"""
    if score < 15:   return "#94a3b8"
    if score < 30:   return "#7dd3fc"
    if score < 45:   return "#fbbf24"
    if score < 60:   return "#86efac"
    if score < 75:   return "#22c55e"
    if score < 90:   return "#10b981"
    return "#06b6d4"


def window_stats(window_wind: dict, cfg: dict) -> dict:
    """(Previously _window_stats in render_html.py)"""
    # [paste body of _window_stats verbatim]
    ...


def expected_wind_chips(headline: dict, cfg: dict) -> str:
    """(Previously _expected_wind_chips in render_html.py)"""
    # [paste body verbatim]
    ...


def stats_html(headline: dict, cfg: dict) -> str:
    """(Previously _stats_html in render_html.py)"""
    # [paste body verbatim]
    ...


def history_html(db_path: str, days: int = 60) -> str:
    """
    Render prediction accuracy history section.
    (Previously _history_html in render_html.py)
    Uses utils/db.backend() instead of importing model.history._backend.
    """
    from utils.db import backend, DEFAULT_SQLITE
    from model.history import load_history, accuracy_summary

    bk = backend()
    actual_path = db_path or DEFAULT_SQLITE
    if bk == "sqlite" and not os.path.exists(actual_path):
        return ""

    # [paste remainder of _history_html body, with these substitutions:]
    # - replace `from model.history import _backend, load_history, accuracy_summary` → removed (done above)
    # - replace `_backend() == "sqlite"` → `bk == "sqlite"`
    # - replace `_score_to_hex(...)` → `score_to_hex(...)`
    # - replace `if _backend() == "sqlite" and not os.path.exists(db_path):` → done above via `actual_path`
    ...
```

Full implementations: copy verbatim from `render_html.py`, applying the substitutions noted above.

- [ ] **Step 15.4: Update render_html.py to use extracted modules**

In `render_html.py`:
- Remove the bodies of `_prob_trend_svg`, `_wind_svg`, `_window_stats`, `_expected_wind_chips`, `_stats_html`, `_history_html`, **and `_score_to_hex`** (moved to `render/data.py`)
- Add imports at the top:
  ```python
  from render.charts import prob_trend_svg, wind_svg
  from render.data import score_to_hex, window_stats, expected_wind_chips, stats_html, history_html
  ```
- Update all internal call sites:
  - `_prob_trend_svg(snaps)` → `prob_trend_svg(snaps)`
  - `_wind_svg(...)` → `wind_svg(...)`
  - `_window_stats(...)` → `window_stats(...)`
  - `_expected_wind_chips(...)` → `expected_wind_chips(...)`
  - `_stats_html(...)` → `stats_html(...)`
  - `_history_html(db_path)` → `history_html(db_path)`
  - `_score_to_hex(...)` → `score_to_hex(...)` (two call sites inside `build_html`)
- Update `build_html` signature default for `db_path`:
  ```python
  from utils.db import DEFAULT_SQLITE
  def build_html(predictions: list[dict], cfg: dict, db_path: str | None = None) -> str:
      if db_path is None:
          db_path = DEFAULT_SQLITE
  ```

- [ ] **Step 15.5: Update render_html.py main() to read from DB**

Replace the `main()` function:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Render forecast snapshots → index.html")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional JSON file override (default: read from DB)",
    )
    parser.add_argument("--config", default=os.path.join(_HERE, "config.toml"))
    parser.add_argument("--out",    default=os.path.join(_HERE, "index.html"))
    args = parser.parse_args()

    if args.predictions:
        with open(args.predictions) as f:
            predictions = json.load(f)
    else:
        from model.predict import load_forecast_snapshots
        predictions = load_forecast_snapshots()

    cfg = load_config(args.config)
    html = build_html(predictions, cfg)

    with open(args.out, "w") as f:
        f.write(html)

    print(f"Written: {args.out}  ({len(predictions)} predictions, "
          f"{len(set(p['predicting_date'] for p in predictions if 'predicting_date' in p))} days)")
```

Also update `load_config` import at top of `render_html.py`:
```python
from utils.config import load_config
```
and remove the local `load_config` function definition.

- [ ] **Step 15.6: Smoke-test the render pipeline end-to-end**

```bash
python render_html.py --out /tmp/test_index.html && echo "Render OK" && head -5 /tmp/test_index.html
```
Expected: `Render OK` and `<!DOCTYPE html>` at the start.

- [ ] **Step 15.7: Commit**

```bash
git add render/__init__.py render/charts.py render/data.py render_html.py
git commit -m "refactor: split render_html.py into render/charts.py and render/data.py"
```

---

### Task 16: deploy.py — remove publish step, update predict/render steps

**Files:**
- Modify: `deploy.py`

- [ ] **Step 16.1: Update deploy.py**

Replace the full `deploy.py` with this updated version (key changes: remove step_publish, update step_predict to not duplicate history recording, update step_render to read from DB, remove parquet staging):

```python
#!/usr/bin/env python3
"""
deploy.py — Local pipeline: download → stitch → predict → render.

Generates index.html for local preview. Does NOT push to git.
CI (GitHub Actions) handles deployment via artifact-based Pages.

Usage:
    python deploy.py                      # last 2 days, predict for today
    python deploy.py --date 2026-03-08   # predict for a specific date
    python deploy.py --days 7            # download last 7 days of data
    python deploy.py --no-download       # skip download, use existing xlsx files
    python deploy.py --preview           # open index.html in browser after render
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def _banner(msg: str) -> None:
    print(f"\n{'─' * 60}\n{msg}\n{'─' * 60}", flush=True)


def _load_config() -> dict:
    from utils.config import load_config
    return load_config(os.path.join(_ROOT, "config.toml"))


def step_download(days: int) -> None:
    _banner(f"[1/4] Downloading last {days} day(s) of weather data")
    from input.scraper import download_range

    today = date.today()
    end   = today
    start = end - timedelta(days=days - 1)
    print(f"Range: {start} → {end}  (today re-fetched for latest readings)", flush=True)

    results = download_range(start, end, force_dates={today})
    n_ok = sum(results.values())
    print(f"\nDownloaded: {n_ok}/{len(results)} day(s)", flush=True)
    if n_ok == 0:
        raise RuntimeError("No files downloaded — check network / credentials.")


def step_stitch() -> None:
    _banner("[2/4] Stitching xlsx → database")
    from input.stitcher import stitch_to_db
    stitch_to_db(input_dir=os.path.join(_ROOT, "input", "downloaded_files"))


def step_predict(ref_date: date | None) -> None:
    label = str(ref_date or date.today())
    _banner(f"[3/4] Running predictions  (ref date: {label})")

    import pandas as pd
    from input.weather_store import load_weather_readings
    from model.predict import predict_now, save_forecast_snapshots

    df = load_weather_readings(
        start=date.today() - timedelta(days=35),
        end=date.today(),
    )
    if df.empty:
        raise RuntimeError("No weather data found. Run step_stitch first.")

    config_path = os.path.join(_ROOT, "config.toml")

    snap_dt = None
    if ref_date is not None:
        now = datetime.now()
        snap_dt = pd.Timestamp(
            year=ref_date.year, month=ref_date.month, day=ref_date.day,
            hour=now.hour, minute=0,
        )

    results = predict_now(df, config_path, snap_dt=snap_dt)

    # Save forecast snapshots to DB (replaces predictions.json)
    save_forecast_snapshots(results)
    print(f"Forecast snapshots saved to DB  ({len(results)} entries)", flush=True)

    # History recording is handled inside predict_now/__main__ when run via CLI.
    # When called from deploy.py we do it explicitly here:
    from model.history import record_predictions, backfill_outcomes
    from model.features import compute_daily_target
    from utils.config import load_config

    cfg = load_config(config_path)
    direct = [r for r in results
              if not r.get("is_extended_forecast") and not r.get("window_observed_only")]
    n_written = record_predictions(direct)
    print(f"History: wrote {n_written} row(s)", flush=True)

    daily_quality = compute_daily_target(df, cfg)
    n_outcomes = backfill_outcomes(daily_quality)
    print(f"History: upserted {n_outcomes} outcome(s)", flush=True)


def step_render() -> None:
    _banner("[4/4] Rendering index.html")
    from render_html import build_html
    from model.predict import load_forecast_snapshots
    from utils.config import load_config

    predictions = load_forecast_snapshots()
    cfg = load_config(os.path.join(_ROOT, "config.toml"))

    html = build_html(predictions, cfg)

    out_path = os.path.join(_ROOT, "index.html")
    with open(out_path, "w") as f:
        f.write(html)

    n_days = len({p["predicting_date"] for p in predictions if "predicting_date" in p})
    print(f"Written: {out_path}  ({len(predictions)} predictions, {n_days} day(s))", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wind-predictor local pipeline (no git push)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--date",        metavar="YYYY-MM-DD")
    parser.add_argument("--days",        type=int, default=2)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-stitch",   action="store_true")
    parser.add_argument("--preview",     action="store_true", help="Open index.html in browser")
    args = parser.parse_args()

    ref_date: date | None = None
    if args.date:
        ref_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    try:
        if not args.no_download:
            step_download(args.days)
        if not args.no_stitch:
            step_stitch()
        step_predict(ref_date)
        step_render()
        print("\nAll done.", flush=True)

        if args.preview:
            import webbrowser
            webbrowser.open(os.path.join(_ROOT, "index.html"))

    except Exception as exc:
        print(f"\nFailed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 16.2: Smoke-test deploy.py --help**

```bash
python deploy.py --help
```
Expected: help text without import errors.

- [ ] **Step 16.3: Commit**

```bash
git add deploy.py
git commit -m "refactor: deploy.py removes publish step and uses DB-backed pipeline"
```

---

## Chunk 6: CI + cleanup

### Task 17: Update forecast.yml for artifact Pages deployment

**Files:**
- Modify: `.github/workflows/forecast.yml`

- [ ] **Step 17.1: Replace forecast.yml**

```yaml
name: Wind Forecast

on:
  schedule:
    - cron: '0  4 * * *'  # 05:00 CET  / 06:00 CEST
    - cron: '0  6 * * *'  # 07:00 CET  / 08:00 CEST
    - cron: '0  9 * * *'  # 10:00 CET  / 11:00 CEST
    - cron: '0 12 * * *'  # 13:00 CET  / 14:00 CEST
    - cron: '0 17 * * *'  # 18:00 CET  / 19:00 CEST
    - cron: '0 21 * * *'  # 22:00 CET  / 23:00 CEST

  workflow_dispatch:

jobs:
  forecast:
    runs-on: ubuntu-latest
    environment: github-pages

    permissions:
      pages:     write
      id-token:  write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Restore model cache
        uses: actions/cache@v4
        with:
          path: model/weights.joblib
          key: wind-model-${{ github.run_id }}
          restore-keys: wind-model-

      - name: Download weather data
        env:
          SUPABASE_DB_URL:   ${{ secrets.SUPABASE_DB_URL }}
          ECOWITT_DEVICE_ID: ${{ secrets.ECOWITT_DEVICE_ID }}
          ECOWITT_AUTHORIZE: ${{ secrets.ECOWITT_AUTHORIZE }}
        run: |
          TODAY=$(date -u '+%Y-%m-%d')
          if [ -f model/weights.joblib ]; then
            START=$(date -u -d '3 days ago' '+%Y-%m-%d')
          else
            START=$(date -u -d '90 days ago' '+%Y-%m-%d')
          fi
          python input/scraper.py "$START" "$TODAY"

      - name: Stitch xlsx → database
        env:
          SUPABASE_DB_URL: ${{ secrets.SUPABASE_DB_URL }}
        run: python input/stitcher.py

      - name: Train model (only when weights are absent)
        env:
          SUPABASE_DB_URL: ${{ secrets.SUPABASE_DB_URL }}
        run: |
          if [ ! -f model/weights.joblib ]; then
            echo "No weights found — training from scratch."
            python model/train.py
          else
            echo "Weights present — skipping training."
          fi

      - name: Predict
        env:
          SUPABASE_DB_URL: ${{ secrets.SUPABASE_DB_URL }}
        run: python model/predict.py

      - name: Render HTML
        env:
          SUPABASE_DB_URL: ${{ secrets.SUPABASE_DB_URL }}
        run: python render_html.py

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: '.'     # workspace root; Pages serves index.html

      - name: Deploy to GitHub Pages
        uses: actions/deploy-pages@v4
```

- [ ] **Step 17.2: Add ECOWITT_DEVICE_ID and ECOWITT_AUTHORIZE to GitHub Secrets**

In GitHub repository → Settings → Secrets and variables → Actions, add:
- `ECOWITT_DEVICE_ID` — value from `config.toml [ecowitt].device_id`
- `ECOWITT_AUTHORIZE` — value from `config.toml [ecowitt].authorize`

- [ ] **Step 17.3: Switch GitHub Pages source to GitHub Actions**

In GitHub repository → Settings → Pages → Source, select **GitHub Actions** (not a branch).

- [ ] **Step 17.4: Commit**

```bash
git add .github/workflows/forecast.yml
git commit -m "ci: deploy Pages via artifact; remove git commit step; add ecowitt env vars"
```

---

### Task 18: Remove tracked binary files + final cleanup

**Files:**
- Modify: `.gitignore`
- Run: `git rm --cached`

- [ ] **Step 18.1: Remove binary/conflict-prone files from git tracking**

```bash
git rm --cached data.parquet predictions.json
# index.html may already be tracked from previous runs — remove it too
git rm --cached index.html 2>/dev/null || true
echo "data.parquet" >> .gitignore
echo "predictions.json" >> .gitignore
```

Note: `.env`, `local.db`, and `index.html` were already added to `.gitignore` in Task 9.

- [ ] **Step 18.2: Verify .gitignore contains all expected entries**

```bash
grep -E "data\.parquet|predictions\.json|index\.html|local\.db|\.env$" .gitignore
```
Expected: all four lines present.

- [ ] **Step 18.3: Commit the removal**

```bash
git add .gitignore
git commit -m "chore: remove data.parquet and predictions.json from git tracking"
```

---

### Task 19: Full end-to-end smoke test

- [ ] **Step 19.1: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 19.2: Run local pipeline (no download)**

```bash
python deploy.py --no-download --no-stitch --preview
```
Expected: `index.html` generated and opened in browser showing forecast data.

- [ ] **Step 19.3: Verify no binary files in git status**

```bash
git status
```
Expected: no `data.parquet`, `predictions.json`, or `index.html` in modified/untracked files.

- [ ] **Step 19.4: Verify features.py and open_meteo.py still use their own implementations**

`model/features.py` uses its own `_circular_std` (numpy-based) for the training feature computation — this is intentional (it operates on pd.Series). The `utils/circular.py` version is used by predict.py and render layer. Both are consistent. No further change needed to features.py.

- [ ] **Step 19.5: Update CLAUDE.md to reflect new architecture**

In `CLAUDE.md`, update the Architecture section to show:
- `data.parquet` → removed, replaced by Supabase `weather_readings`
- `predictions.json` → removed, replaced by Supabase `forecast_snapshots`
- `render/` → new sub-package
- `utils/` → new shared utilities layer

- [ ] **Step 19.6: Final commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to reflect v2 architecture (Supabase-backed)"
```

---

## Summary of files changed

| Action | Files |
|--------|-------|
| **New** | `utils/__init__.py`, `utils/circular.py`, `utils/config.py`, `utils/db.py` |
| **New** | `input/weather_store.py`, `render/__init__.py`, `render/charts.py`, `render/data.py` |
| **New** | `supabase/schema_additions.sql`, `supabase/migrate_parquet.py`, `.env.example` |
| **New** | `tests/__init__.py`, `tests/utils/`, `tests/input/`, `tests/model/` |
| **Modified** | `config.toml`, `input/scraper.py`, `input/stitcher.py` |
| **Modified** | `model/train.py`, `model/predict.py`, `model/history.py` |
| **Modified** | `render_html.py`, `deploy.py`, `.github/workflows/forecast.yml`, `.gitignore` |
| **Removed from tracking** | `data.parquet`, `predictions.json`, `index.html`, `local.db` |
