# windPredictor

Predicts good sailing conditions from a personal Ecowitt weather station. Runs on a Raspberry Pi, publishing forecasts to a website.

## How it works

1. **Scraper** downloads daily xlsx exports from ecowitt.net
2. **Stitcher** merges them into a single Parquet time series
3. **Trainer** learns from historical (snapshot → next-day conditions) pairs
4. **Predictor** takes scheduled snapshots and outputs a probability forecast as JSON

Good sailing is defined as: wind 2–12 knots with direction consistent within 30° over any 2-hour window, between 08:00–16:00. All thresholds are configurable in `config.toml`.

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

Merges all xlsx files into `data.parquet`. Re-running is safe — existing data is preserved and deduplicated.

### 3. Train the model

```bash
python model/train.py
```

Builds training pairs: features extracted at each configured snapshot time on day D are paired with the sailing quality of the target day (day D+1 for evening snapshots, same day for early-morning snapshots). Saves weights to `model/weights.joblib`.

Requires at least 2 days of data. More history = better model.

### 4. Predict

```bash
python model/predict.py                    # all snapshots for today
python model/predict.py 2026-03-07        # all snapshots for a specific date
python model/predict.py 2026-03-07 18:00  # single snapshot
```

Outputs JSON to stdout and writes `predictions.json`. Example output:

```json
[
  {
    "snapshot": "2026-03-07T06:00:00",
    "predicting_date": "2026-03-07",
    "sailing_window": "08:00–16:00",
    "probability": 0.327,
    "good": true,
    "threshold": 0.3
  }
]
```

## Configuration

All parameters live in `config.toml`:

```toml
[sailing]
window_start             = "08:00"
window_end               = "16:00"
wind_speed_min           = 2.0    # knots
wind_speed_max           = 12.0   # knots
wind_dir_consistency_max = 30.0   # degrees
consistency_window_hours = 2

[prediction]
# Snapshots before window_start predict the same day.
# Snapshots after window_start predict the next day.
snapshots         = ["06:00", "18:00", "00:00", "05:00"]
min_good_fraction = 0.30   # fraction of sailing window that must be good

[paths]
data_parquet     = "data.parquet"
model_file       = "model/weights.joblib"
predictions_file = "predictions.json"
```

## Project structure

```
windPredictor/
├── config.toml              # all tunable parameters
├── input/
│   ├── scraper.py           # download daily xlsx from ecowitt.net
│   ├── stitcher.py          # merge xlsx files → data.parquet
│   └── downloaded_files/    # raw daily xlsx (gitignored)
├── model/
│   ├── features.py          # feature extraction + target computation
│   ├── train.py             # train and save weights
│   ├── predict.py           # load weights, output forecast JSON
│   └── weights.joblib       # saved model (gitignored)
├── data.parquet             # master time series (gitignored)
├── predictions.json         # latest forecast output
├── main.py                  # exploratory analysis (legacy)
└── requirements.txt
```

## Data format

`data.parquet` is the canonical dataset, indexed by timestamp at 5-minute intervals (granularity may vary).

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
