# windPredictor

Predicts good sailing conditions from a personal Ecowitt weather station. A Random Forest model trained on weather-station history outputs a probability forecast published to GitHub Pages six times a day.

## How it works

1. **Scraper** downloads daily xlsx exports from ecowitt.net into a local folder
2. **Stitcher** upserts them into the `weather_readings` Supabase table
3. **Trainer** builds historical training pairs and saves a Random Forest model
4. **Predictor** loads the saved model, takes a weather snapshot, and saves a rolling 7-day forecast to the `forecast_snapshots` table
5. **Renderer** reads `forecast_snapshots` and builds a self-contained `index.html`
6. **CI** uploads `index.html` as a GitHub Pages artifact — no git commits needed

**Good sailing** is defined as: wind speed 2–12 knots with direction consistent within 30° (circular std) over any 2-hour rolling window, during the configurable sailing window (default 08:00–16:00). All thresholds live in `config.toml`.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # fill in SUPABASE_DB_URL and Ecowitt credentials
```

When `SUPABASE_DB_URL` is unset all four tables fall back to a local `local.db` SQLite file — useful for offline development.

## Workflow

### 1. Download data

```bash
python input/scraper.py                          # last 7 days (default)
python input/scraper.py 2026-03-01               # single date
python input/scraper.py 2026-02-01 2026-03-06    # date range
```

Files land in `input/downloaded_files/`. Already-downloaded dates are skipped.

### 2. Stitch into the weather database

```bash
python input/stitcher.py
```

Upserts all xlsx files into the `weather_readings` table (Supabase or `local.db`). Re-running is safe — rows are upserted by timestamp.

### 3. Train the model

```bash
python model/train.py
```

Saves weights to `model/weights.joblib`. See [Model methodology](#model-methodology) below.

### 4. Predict

```bash
python model/predict.py
```

Saves predictions to the `forecast_snapshots` table (rolling 7-day window per predicting date).

### 5. Render and preview

```bash
python render_html.py --out index.html
```

Or run the full pipeline with a local browser preview:

```bash
python deploy.py --preview
```

### 6. Explore (optional)

```bash
python explore.py             # all panels → exploration.png
python explore.py --data      # data overview (4 panels)
python explore.py --model     # feature importances + pressure anomaly
python explore.py --out my.png
```

Produces static PNGs — safe to run headlessly.

## Automated deployment (GitHub Actions)

`.github/workflows/forecast.yml` schedules the pipeline at each snapshot time using GitHub's CI runners.

**Required GitHub secrets:**

| Secret | Value |
|--------|-------|
| `SUPABASE_DB_URL` | PostgreSQL connection string |
| `ECOWITT_DEVICE_ID` | Station device ID |
| `ECOWITT_AUTHORIZE` | Station authorize token |

**Required repository setting:** go to Settings → Pages and set source to **"GitHub Actions"**.

The workflow publishes `index.html` via `actions/upload-pages-artifact` + `actions/deploy-pages` — no files are ever committed by the pipeline, so there are no git conflicts between local and CI runs.

**Schedule (UTC → local):**

| Cron (UTC) | CET  | CEST |
|-----------|------|------|
| `0 4 * * *`  | 05:00 | 06:00 |
| `0 6 * * *`  | 07:00 | 08:00 |
| `0 9 * * *`  | 10:00 | 11:00 |
| `0 12 * * *` | 13:00 | 14:00 |
| `0 17 * * *` | 18:00 | 19:00 |
| `0 21 * * *` | 22:00 | 23:00 |

The workflow is also triggerable manually from the Actions tab (`workflow_dispatch`).

## One-time migration

If you have an existing `data.parquet` from a previous version:

```bash
python supabase/migrate_parquet.py    # dry-run first (prints row count)
SUPABASE_DB_URL=... python supabase/migrate_parquet.py  # actual migration
```

If you have prediction history in `predictions.db`:

```bash
SUPABASE_DB_URL=... python supabase/migrate_from_sqlite.py
```

## Model methodology

### Target variable

For each calendar day, the **good-sailing fraction** is computed: the proportion of 5-minute intervals within the sailing window (08:00–16:00) where instantaneous conditions are good. A day is labelled **good (y=1)** if this fraction exceeds `min_good_fraction` (default 30%).

### Training pairs

The model predicts the sailing window from a weather snapshot:

- **Snapshots before the window closes** (before `window_end`, e.g. 10:00, 13:00) → predict **today's** window, incorporating live station readings up to the snapshot time
- **Snapshots after the window closes** (≥ `window_end`, e.g. 18:00, 22:00) → predict **the next day's** window

Each (snapshot time, date) pair becomes one training row.

### Feature engineering

Rather than feeding raw sensor values, the model uses **28-day trailing anomalies** for the key continuous readings:

| Feature | What it captures |
|---|---|
| `pressure_anomaly` | Pressure departure from 28-day mean — rising/falling systems |
| `temperature_anomaly` | Air mass character relative to the season |
| `wind_speed_anomaly` | Whether it's windier or calmer than usual |
| `wind_gust_anomaly`, `humidity_anomaly` | Similar |

Short-term relative features are kept as-is:

| Feature group | Examples |
|---|---|
| Trends | `pressure_trend_3h`, `temp_trend_3h`, `pressure_trend_6h/12h` |
| Rolling means | `wind_speed_mean_3h/6h/12h` |
| Consistency | `wind_dir_consistency_3h/6h` (circular std of direction) |
| Direction | `wind_dir_sin`, `wind_dir_cos` |

### Model

Random Forest classifier (`n_estimators=300`, `max_depth=6`, `class_weight=balanced`), evaluated with stratified 5-fold cross-validation (ROC-AUC). Trained on ~2 years of data: **CV ROC-AUC ≈ 0.87**.

## Configuration

`config.toml` — all tunable parameters:

```toml
[sailing]
window_start             = "08:00"
window_end               = "16:00"
wind_speed_min           = 2.0    # knots
wind_speed_max           = 12.0   # knots
wind_dir_consistency_max = 30.0   # max circular std over consistency window (degrees)
consistency_window_hours = 2

[prediction]
min_good_fraction = 0.30

[location]
name = "Traunsee"
lat  = 47.8071
lon  = 13.7790

[ecowitt]
device_id = "..."      # override with ECOWITT_DEVICE_ID env var
authorize = "..."      # override with ECOWITT_AUTHORIZE env var

[model]
n_estimators    = 300
max_depth       = 6
min_samples_leaf = 2
```

## Project structure

```
windPredictor/
├── config.toml              # all tunable parameters
├── .env.example             # environment variable template
├── deploy.py                # full pipeline: download → stitch → predict → render
├── render_html.py           # render forecast_snapshots → index.html
├── explore.py               # optional visualisation (data + model diagnostics)
├── requirements.txt
├── input/
│   ├── scraper.py           # download daily xlsx from ecowitt.net
│   ├── stitcher.py          # upsert xlsx files → weather_readings table
│   ├── weather_store.py     # load_weather_readings() / upsert_readings()
│   ├── open_meteo.py        # NWP enrichment from Open-Meteo
│   └── downloaded_files/    # raw daily xlsx (gitignored)
├── model/
│   ├── features.py          # feature extraction + target computation
│   ├── train.py             # build training pairs, train, save weights
│   ├── predict.py           # load weights, save forecast to DB
│   ├── history.py           # prediction history + outcomes
│   └── weights.joblib       # saved model artifact (gitignored)
├── utils/
│   ├── circular.py          # circular_std() shared utility
│   ├── config.py            # load_config() + dotenv loading
│   └── db.py                # get_connection(), backend(), placeholder()
├── render/
│   ├── charts.py            # SVG chart helpers
│   └── data.py              # HTML section builders
├── supabase/
│   ├── schema.sql           # predictions + outcomes tables
│   ├── schema_additions.sql # weather_readings + forecast_snapshots tables
│   ├── migrate_parquet.py   # one-time: parquet → weather_readings
│   └── migrate_from_sqlite.py # one-time: predictions.db → Supabase
├── tests/
│   ├── utils/               # tests for utils/
│   └── input/               # tests for input/
├── scripts/
│   └── install_cron.py      # install/remove cron jobs
└── .github/
    └── workflows/
        └── forecast.yml     # GitHub Actions scheduled deploy → Pages artifact
```

## Weather data columns

All weather readings stored in `weather_readings` (ISO-8601 local naive timestamp as primary key):

| Column | Unit |
|---|---|
| `temperature` | °C |
| `feels_like` | °C |
| `dew_point` | °C |
| `humidity` | % |
| `solar` | W/m² |
| `uvi` | index |
| `rain_rate` | mm/hr |
| `daily_rain` | mm |
| `pressure_relative` / `pressure_absolute` | hPa |
| `water_temperature` | °C |
| `wind_speed` / `wind_gust` | knots |
| `wind_direction` | degrees |
