# windPredictor — Simplification & Future-Proofing Design

**Date:** 2026-03-15
**Status:** Approved

## Goal

Eliminate all binary/conflict-prone files from git commits. Move all mutable state to Supabase. Deploy GitHub Pages via artifact (no commits). Fix code-quality issues.

## Problem Statement

- `data.parquet` (~848 KB) is committed on every forecast run (6×/day) — binary churn
- `predictions.json` is committed on every run — causes git conflicts between local and CI runs
- `index.html` is committed on every run — same conflict problem
- `_circular_std` and `load_config` are duplicated across 3–4 modules
- Ecowitt credentials are hardcoded in `input/scraper.py`
- `config.toml [location]` section is broken (header commented out, keys are not)
- History-recording logic is duplicated between `deploy.py` and `model/predict.py`

## Design

### Principle

**Git contains only code and config. All mutable state lives in Supabase. All output is deployed as a Pages artifact — never committed.**

### Data Layer — Supabase Tables

| Table | Purpose | Replaces |
|-------|---------|---------|
| `weather_readings` | Weather station time series (40K rows, upsert by timestamp) | `data.parquet` |
| `forecast_snapshots` | Rolling 7-day window of per-snapshot predictions | `predictions.json` |
| `predictions` | ML prediction history (already exists) | — |
| `outcomes` | Actual observed outcomes (already exists) | — |

**`weather_readings` schema:**
```sql
CREATE TABLE IF NOT EXISTS weather_readings (
    timestamp           TEXT PRIMARY KEY,  -- ISO-8601 local naive (e.g. "2026-03-15T08:05:00"), no TZ suffix
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
```

**Timestamp convention:** All timestamps stored as ISO-8601 local naive strings (no UTC offset, no `Z`), matching the existing parquet index format. ISO-8601 local time sorts correctly as TEXT for date-range queries. `load_weather_readings()` reconstructs the DataFrame index with `pd.to_datetime(col)` (no `utc=True`), producing a `DatetimeTZNaive` index identical to the current parquet output — compatible with `between_time()`, `.loc[date_str]`, and all window arithmetic in `features.py`. DST-transition days (one missing or duplicated hour per year) are handled the same as today: the parquet already stores naive local time and the code is unaffected.

**`forecast_snapshots` schema:**
```sql
CREATE TABLE IF NOT EXISTS forecast_snapshots (
    snapshot        TEXT NOT NULL,      -- ISO-8601 datetime
    predicting_date TEXT NOT NULL,      -- YYYY-MM-DD
    payload         TEXT NOT NULL,      -- full JSON blob for this entry
    PRIMARY KEY (snapshot, predicting_date)
);
CREATE INDEX IF NOT EXISTS idx_fs_date ON forecast_snapshots(predicting_date);
```

**Local fallback:** when `SUPABASE_DB_URL` is unset, all four tables fall back to a single `local.db` SQLite file (same pattern as current `history.py`).

### Shared Utilities (new files)

| File | Responsibility |
|------|---------------|
| `utils/config.py` | Single `load_config(path)` that also calls `load_dotenv()`. Replaces the 4 duplicate copies. |
| `utils/db.py` | `get_connection(db_path="local.db") -> (connection, backend_str)`, `backend() -> str`, `placeholder(backend) -> str`. Shared by history.py, weather store, snapshot store. `db_path` is only used when backend is SQLite; Postgres ignores it. All callers updated to pass `db_path` if they previously used a custom SQLite path. The fallback file is `local.db` (replaces `predictions.db` — see migration notes). |
| `utils/circular.py` | Single `circular_std(angles: pd.Series | list) -> float` implementation. Replaces 3 copies. |

### Per-Module Changes

**`config.toml`**
- Fix `[location]` section: uncomment the header so `name/lat/lon` are actually under it
- Add `[ecowitt]` section with `device_id` and `authorize` keys (read at runtime; env vars override)

**`input/scraper.py`**
- Remove hardcoded `DEVICE_ID`, `AUTHORIZE`, `SORT_LIST`
- Read from `config.toml [ecowitt]` via `utils/config.py`; override with `ECOWITT_DEVICE_ID` / `ECOWITT_AUTHORIZE` env vars

**`input/stitcher.py`**
- Replace parquet write with upsert into `weather_readings`
- Keep `parse_xlsx()` unchanged (pure transformation, no I/O)
- `stitch()` becomes `stitch_to_db()` — parses xlsx files, upserts rows, returns count

**`model/train.py`**
- Replace `pd.read_parquet(...)` with `load_weather_readings()` (no date bounds — loads all history for training)
- `load_config()` → `utils/config.py`

**`model/predict.py`**
- Replace `pd.read_parquet(...)` with `load_weather_readings(start=date.today() - timedelta(days=30), end=date.today())` — sufficient for feature extraction (features use at most 28-day trailing window)
- `_circular_std` → `utils/circular.py`
- `load_config()` → `utils/config.py`
- `merge_predictions()` replaced by `save_forecast_snapshots(results, days_to_keep=7)` that:
  1. Upserts each entry as `(snapshot, predicting_date, payload=json.dumps(entry))` into `forecast_snapshots`
  2. Issues `DELETE FROM forecast_snapshots WHERE predicting_date < <cutoff_date>` to prune old entries
- `load_forecast_snapshots(days=7) -> list[dict]` added: queries `forecast_snapshots WHERE predicting_date >= cutoff`, calls `json.loads(row["payload"])` for each row, returns `list[dict]` — same type `build_html()` accepts
- History-recording (`record_predictions`, `backfill_outcomes`) lives only here, not duplicated in `deploy.py`

**`model/history.py`**
- Connection helpers delegate to `utils/db.py`
- API unchanged

**`deploy.py`**
- Remove `step_stitch()` parquet reference
- Remove history-recording from `step_predict()` (predict.py handles it)
- `step_render()`: call `load_forecast_snapshots()` instead of `json.load(pred_path)`
- `step_publish()` **removed entirely** — local runs generate `index.html` for local preview only; no git push from local. The file is opened in the default browser if `--preview` flag is passed.
- `index.html` added to `.gitignore`

**`render_html.py`** — split into:
- `render_html.py` — entry point, `build_html(predictions: list[dict], cfg: dict, db_path: str | None) -> str` signature preserved; `main()` updated to call `load_forecast_snapshots()` instead of `json.load(file)` — no `--predictions` argument needed in CI, but kept as an optional override for local debugging
- `render/charts.py` — `_prob_trend_svg()`, `_wind_svg()`
- `render/data.py` — `_window_stats()`, `_expected_wind_chips()`, `_stats_html()`, `_history_html()`. `_history_html()` uses `utils/db.backend()` (public) instead of importing `model.history._backend` (private)
- `_circular_std` → `utils/circular.py`

`deploy.py`'s `step_render()` is updated to call `load_forecast_snapshots()` and pass the result to `build_html()`, removing the `json.load(pred_path)` call.

**`input/weather_store.py`** (new)
- `load_weather_readings(start: date | None = None, end: date | None = None) -> pd.DataFrame` — loads from Supabase or SQLite fallback; returns the same DataFrame format that code currently gets from parquet. When `start` and `end` are both `None`, loads all rows (used by `train.py`). When only `end` is given, loads from the earliest available row up to `end`. Range queries use `WHERE timestamp >= <start_iso> AND timestamp <= <end_iso>` on the TEXT column (ISO-8601 local sorts correctly as text for same-timezone data).

### Credentials & Environment

**Local (`.env`, gitignored):**
```
SUPABASE_DB_URL=postgresql://...
ECOWITT_DEVICE_ID=...
ECOWITT_AUTHORIZE=...
```

**GitHub Secrets (CI):** `SUPABASE_DB_URL` already set. Add `ECOWITT_DEVICE_ID`, `ECOWITT_AUTHORIZE`.

### CI Changes (`forecast.yml`)

1. Remove the `git add / git commit / git push` step entirely — no files are committed
2. Replace `permissions: contents: write` with:
   ```yaml
   permissions:
     pages: write
     id-token: write
   ```
3. Add `environment: github-pages` to the job
4. Add deployment steps after the Render HTML step:
   ```yaml
   - name: Upload Pages artifact
     uses: actions/upload-pages-artifact@v3
     with:
       path: '.'          # workspace root; Pages serves index.html from here

   - name: Deploy to GitHub Pages
     uses: actions/deploy-pages@v4
   ```
5. Pass credentials as env vars to the Download weather data step:
   ```yaml
   - name: Download weather data
     env:
       SUPABASE_DB_URL:      ${{ secrets.SUPABASE_DB_URL }}
       ECOWITT_DEVICE_ID:    ${{ secrets.ECOWITT_DEVICE_ID }}
       ECOWITT_AUTHORIZE:    ${{ secrets.ECOWITT_AUTHORIZE }}
     run: ...
   ```
6. Note: GitHub Pages source must be set to **"GitHub Actions"** in the repository Settings → Pages. This is incompatible with the previous branch-based deployment; switch it once when deploying the new workflow.

### Migration

**`supabase/migrate_parquet.py`** — one-time script:
- Reads `data.parquet`
- Upserts all rows into `weather_readings` (ISO-8601 local naive timestamp strings as primary key)
- Run once locally before the new pipeline goes live

**Local SQLite migration:** `predictions.db` (existing local history) is superseded by `local.db` (unified fallback for all four tables). Developers who have local history in `predictions.db` can run `supabase/migrate_from_sqlite.py` to push it to Supabase, or accept that local-only history is abandoned. The file `predictions.db` is already gitignored. Add `local.db` to `.gitignore`.

**`input/scraper.py` SORT_LIST:** `SORT_LIST = "0|1|2|49|4|5|32"` is a station-specific sensor selection constant, not a credential. It is kept as a module-level constant (not moved to config). The spec note about removing it referred only to credentials.

### What Stays in Git

After this change, the only files the pipeline ever modifies are generated externally (Pages artifact). Git commits become: code changes only. `data.parquet`, `predictions.json`, `index.html` are all removed from git tracking.

## File Inventory

**New files:**
- `utils/__init__.py`
- `utils/config.py`
- `utils/db.py`
- `utils/circular.py`
- `input/weather_store.py`
- `render/__init__.py`
- `render/charts.py`
- `render/data.py`
- `supabase/migrate_parquet.py`
- `supabase/schema_additions.sql`
- `.env.example`

**Modified files:**
- `config.toml`
- `input/scraper.py`
- `input/stitcher.py`
- `model/train.py`
- `model/predict.py`
- `model/history.py`
- `deploy.py`
- `render_html.py`
- `.github/workflows/forecast.yml`
- `.gitignore`
- `supabase/schema.sql`

**Removed from git tracking:**
- `data.parquet`
- `predictions.json`
- `index.html` (added to `.gitignore`; local runs generate it for browser preview only)
- `local.db` (added to `.gitignore`)

## Non-Goals

- Replacing the Random Forest model
- Adding new UI features
- Changing the prediction algorithm
- Moving model weights out of the Actions cache
