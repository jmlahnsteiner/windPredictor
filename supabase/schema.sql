-- windPredictor — Supabase schema
-- Run this once in the Supabase SQL Editor (Dashboard → SQL Editor → New query)

CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
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
