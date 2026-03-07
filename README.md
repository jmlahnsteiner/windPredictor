# windPredictor

Predicts good sailing conditions from a personal Ecowitt weather station.

## How it works

1. **Scraper** downloads daily xlsx exports from ecowitt.net
2. **Stitcher** merges them into a single Parquet time series
3. **Model** trains a Random Forest classifier on the historical data

Good sailing is defined as: wind 2–12 knots with direction consistent within 30° over any 2-hour window, between 09:00–16:00.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Workflow

### 1. Download data

```bash
python input/scraper.py                          # last 7 days
python input/scraper.py 2026-03-01               # single date
python input/scraper.py 2026-02-01 2026-03-06    # date range
```

Files land in `input/downloaded_files/`.

### 2. Stitch into a time series

```bash
python input/stitcher.py
```

Merges all xlsx files (including any previously stitched data) into `data.parquet`. Re-running is safe — existing data is preserved and deduplicated.

### 3. Train and analyse

```bash
python main.py
```

Reads `input/*.xlsx` and `input/downloaded_files/*.xlsx`, trains a Random Forest model, and saves:
- `sailing_analysis.png` — pattern visualisations
- `processed_weather_data.csv` — feature-engineered dataset

## Project structure

```
windPredictor/
├── input/
│   ├── scraper.py          # download daily xlsx from ecowitt.net
│   ├── stitcher.py         # merge xlsx files → data.parquet
│   └── downloaded_files/   # raw daily xlsx (gitignored)
├── data.parquet            # master time series (gitignored)
├── main.py                 # feature engineering + ML model
└── requirements.txt
```

## Data format

`data.parquet` is the canonical dataset. Schema:

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

Index is a UTC timestamp at 5-minute intervals.
