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
