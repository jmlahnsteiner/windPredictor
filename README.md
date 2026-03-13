# windPredictor

Predicts good sailing conditions from a personal Ecowitt weather station. Designed to run on a Raspberry Pi, publishing probability forecasts to a website.

## How it works

1. **Scraper** downloads daily xlsx exports from ecowitt.net into a local folder
2. **Stitcher** merges them into a single Parquet time series (`data.parquet`)
3. **Trainer** builds historical training pairs and saves a Random Forest model
4. **Predictor** loads the saved model, takes a weather snapshot, and emits a probability forecast as JSON

**Good sailing** is defined as: wind speed 2–12 knots with direction consistent within 30° (circular range) over any 2-hour rolling window, during the configurable sailing window (default 08:00–16:00). All thresholds live in `config.toml`.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Workflow

### 1. Download data

```bash
python input/scraper.py                          # last 7 days (default)
python input/scraper.py 2026-03-01               # single date
python input/scraper.py 2026-02-01 2026-03-06    # date range
```

Files land in `input/downloaded_files/`. Already-downloaded dates are skipped.

### 2. Stitch into a time series

```bash
python input/stitcher.py
```

Merges all xlsx files into `data.parquet`. Re-running is safe — existing data is carried over and deduplicated on timestamp.

### 3. Train the model

```bash
python model/train.py
```

Saves weights to `model/weights.joblib`. See [Model methodology](#model-methodology) below.

### 4. Predict

```bash
python model/predict.py                    # all configured snapshots for today
python model/predict.py 2026-03-07        # specific date
python model/predict.py 2026-03-07 18:00  # single snapshot
```

Outputs JSON to stdout and writes `predictions.json`:

```json
[
  {
    "snapshot": "2026-03-07T18:00:00",
    "predicting_date": "2026-03-08",
    "sailing_window": "08:00–16:00",
    "probability": 0.41,
    "good": true,
    "threshold": 0.3
  }
]
```

### 5. Deploy

One command runs the full pipeline and publishes `index.html` to GitHub Pages:

```bash
python deploy.py              # download last 2 days → stitch → predict → render → push
python deploy.py --dry-run    # same, but skip git push
python deploy.py --days 7     # download last 7 days instead
python deploy.py --no-download --no-stitch  # re-run from existing data
```

Only `index.html` is ever committed — data artifacts remain local/gitignored.

### 6. Explore (optional)

```bash
python explore.py             # all panels → exploration.png
python explore.py --data      # data overview (4 panels)
python explore.py --model     # feature importances + pressure anomaly
python explore.py --out my.png
```

Produces static PNGs — safe to run headlessly on the Pi.

## Automated deployment

### Local cron — Raspberry Pi or any always-on machine (recommended)

Install one cron job per snapshot time configured in `config.toml`:

```bash
python scripts/install_cron.py           # install
python scripts/install_cron.py --list    # preview without making changes
python scripts/install_cron.py --remove  # uninstall
```

Each job runs `deploy.py` at the configured snapshot time and appends to `logs/deploy.log`.
This is the recommended approach — persistent local storage means `data.parquet` and
`model/weights.joblib` accumulate naturally over time.

### GitHub Actions

`.github/workflows/forecast.yml` schedules the pipeline at each snapshot time using GitHub's
CI runners. `data.parquet` and `model/weights.joblib` are persisted between runs via the
Actions cache (10 GB limit, entries expire after 7 days of disuse — fine for daily runs).

**Timezone:** GitHub Actions cron is UTC. The offsets in the workflow file assume CET (UTC+1);
edit the cron expressions at the top of `forecast.yml` if you're in a different timezone or
during summer time (CEST = UTC+2).

The workflow is also triggerable manually from the Actions tab (`workflow_dispatch`).

### Other options

| Method | Persistent storage | Cost | Notes |
|---|---|---|---|
| Raspberry Pi + cron | ✅ local disk | Free | Recommended — `install_cron.py` handles setup |
| GitHub Actions | ⚠️ cache (7-day TTL) | Free (2 000 min/month) | `forecast.yml` included |
| Cloud VM (e.g. Hetzner CX11) | ✅ persistent disk | ~€4/month | Same `install_cron.py` setup |
| AWS Lambda + EventBridge | ✅ S3 bucket | ~$0 (pay-per-use) | More setup; store parquet in S3 |
| Render / Railway scheduled job | ❌ ephemeral | Free tier available | Data lost between runs — not ideal |

## Model methodology

### Target variable

For each calendar day, the **good-sailing fraction** is computed: the proportion of 5-minute intervals within the sailing window (08:00–16:00) where instantaneous conditions are good. A day is labelled **good (y=1)** if this fraction exceeds `min_good_fraction` (default 30%).

### Training pairs

The model predicts the sailing window from a weather snapshot:

- **Snapshots before the window closes** (before `window_end`, e.g. 10:00, 13:00) → predict **today's** window, incorporating live station readings up to the snapshot time
- **Snapshots after the window closes** (≥ `window_end`, e.g. 18:00, 22:00) → predict **the next day's** window

Each (snapshot time, date) pair becomes one training row, giving up to `n_days × n_snapshots` training examples.

### Feature engineering

Rather than feeding raw sensor values, the model uses **28-day trailing anomalies** for the key continuous readings:

| Feature | What it captures |
|---|---|
| `pressure_anomaly` | Pressure departure from 28-day mean — rising/falling systems |
| `temperature_anomaly` | Air mass character relative to the season |
| `wind_speed_anomaly` | Whether it's windier or calmer than usual |
| `wind_gust_anomaly`, `humidity_anomaly` | Similar |

Subtracting the 28-day rolling mean removes the seasonal baseline, so the model learns *synoptic weather signals* (fronts, pressure systems) rather than acting as a calendar lookup. A weak seasonal anchor is retained via `sin_doy` / `cos_doy` — the sailing season is genuinely seasonal and this should influence the prior.

Short-term relative features are kept as-is since they are already de-trended:

| Feature group | Examples |
|---|---|
| Trends | `pressure_trend_3h`, `temp_trend_3h`, `pressure_trend_6h/12h` |
| Rolling means | `wind_speed_mean_3h/6h/12h` |
| Consistency | `wind_dir_consistency_3h/6h` (circular std of direction) |
| Direction | `wind_dir_sin`, `wind_dir_cos` |

### Model

Random Forest classifier (`n_estimators=300`, `max_depth=6`, `class_weight=balanced`), evaluated with stratified 5-fold cross-validation (ROC-AUC). Trained on ~1 year of data: **CV ROC-AUC ≈ 0.87**.

Top features on that dataset (in order): seasonal position (`cos_doy`), wind direction consistency over 3h, pressure anomaly, wind speed means over 6–12h.

## Configuration

`config.toml` — all tunable parameters:

```toml
[sailing]
window_start             = "08:00"
window_end               = "16:00"
wind_speed_min           = 2.0    # knots
wind_speed_max           = 12.0   # knots
wind_dir_consistency_max = 30.0   # max circular range over consistency window (degrees)
consistency_window_hours = 2

[prediction]
# Snapshots before window_end → predict same day (incorporating live readings).
# Snapshots at or after window_end → predict next day.
min_good_fraction = 0.30

[paths]
data_parquet     = "data.parquet"
model_file       = "model/weights.joblib"
predictions_file = "predictions.json"

[model]
n_estimators    = 300
max_depth       = 6
min_samples_leaf = 2
```

## Project structure

```
windPredictor/
├── config.toml              # all tunable parameters
├── deploy.py                # full pipeline: download → stitch → predict → render → push
├── render_html.py           # render predictions.json → index.html
├── explore.py               # optional visualisation (data + model diagnostics)
├── index.html               # generated forecast page (GitHub Pages source)
├── requirements.txt
├── input/
│   ├── scraper.py           # download daily xlsx from ecowitt.net
│   ├── stitcher.py          # merge xlsx files → data.parquet
│   └── downloaded_files/    # raw daily xlsx (gitignored)
├── model/
│   ├── features.py          # feature extraction + target computation
│   ├── train.py             # build training pairs, train, save weights
│   ├── predict.py           # load weights, output forecast JSON
│   └── weights.joblib       # saved model artifact (gitignored)
├── scripts/
│   └── install_cron.py      # install/remove cron jobs from config snapshots
├── .github/
│   └── workflows/
│       └── forecast.yml     # GitHub Actions scheduled deploy
├── data.parquet             # master time series (gitignored)
├── predictions.json         # latest forecast output (gitignored)
└── logs/                    # cron output (gitignored)
```

## Data format

`data.parquet` — canonical dataset, indexed by timestamp (5-minute intervals, granularity may vary).

| Column | Unit |
|---|---|
| `temperature` | °C |
| `feels_like` | °C |
| `dew_point` | °C |
| `humidity` | % |
| `solar` | W/m² |
| `uvi` | index |
| `rain_rate` | mm/hr |
| `daily_rain` / `hourly_rain` / `weekly_rain` / `monthly_rain` / `yearly_rain` / `event_rain` | mm |
| `pressure_relative` / `pressure_absolute` | hPa |
| `water_temperature` | °C |
| `wind_speed` / `wind_gust` | knots |
| `wind_direction` | degrees |
