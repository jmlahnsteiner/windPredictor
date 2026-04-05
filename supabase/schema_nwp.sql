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
