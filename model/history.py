"""
model/history.py — Persist every prediction run to the database.

Supports two backends selected automatically via the SUPABASE_DB_URL env var:
  • PostgreSQL (Supabase) — when SUPABASE_DB_URL is set (preferred)
  • SQLite (local fallback) — when running without credentials

Schema
------
predictions
    id              SERIAL / AUTOINCREMENT PRIMARY KEY
    run_ts          TEXT    ISO-8601 UTC timestamp of when the prediction was made
    snapshot_dt     TEXT    ISO-8601 datetime the features were extracted from
    predicting_date TEXT    YYYY-MM-DD date being predicted
    probability     REAL    model output (0-1)
    good            INTEGER 1 = model says "good sailing day", 0 = not
    threshold       REAL    min_good_fraction used at prediction time

outcomes (filled in retrospectively once the day has passed)
    predicting_date TEXT    PRIMARY KEY
    actual_good     INTEGER 1 = conditions were actually good, 0 = not
    actual_frac     REAL    fraction of sailing window with good conditions

Usage
-----
    from model.history import record_predictions, load_history, record_outcome
    record_predictions(results)
    df = load_history(days=30)
"""

import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from utils.db import DEFAULT_SQLITE, backend as _backend_fn, get_connection, placeholder

# ── Backend detection ─────────────────────────────────────────────────────────

# ── SQLite schema ─────────────────────────────────────────────────────────────

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


# ── Connection helpers ────────────────────────────────────────────────────────

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


# ── Public API ────────────────────────────────────────────────────────────────

def record_predictions(results: list[dict], db_path: str = DEFAULT_SQLITE) -> int:
    """
    Insert prediction results into the database.
    Skips entries that have an "error" key or are missing required fields.
    Returns the number of rows inserted.
    """
    run_ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for r in results:
        if "error" in r or "predicting_date" not in r or "probability" not in r:
            continue
        rows.append((
            run_ts,
            r["snapshot"],
            r["predicting_date"],
            float(r["probability"]),
            int(bool(r["good"])),
            float(r.get("threshold", 0.30)),
        ))

    if not rows:
        return 0

    con, backend = _connect(db_path)
    ph = _ph(backend)
    sql = (
        f"INSERT INTO predictions "
        f"(run_ts, snapshot_dt, predicting_date, probability, good, threshold) "
        f"VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})"
    )
    try:
        cur = con.cursor()
        cur.executemany(sql, rows)
        con.commit()
    finally:
        con.close()
    return len(rows)


def record_outcome(
    predicting_date: str,
    actual_good: bool,
    actual_frac: float,
    db_path: str = DEFAULT_SQLITE,
) -> None:
    """
    Store the ground-truth outcome for a date so accuracy can be computed later.
    """
    con, backend = _connect(db_path)
    ph = _ph(backend)
    if backend == "postgres":
        sql = (
            f"INSERT INTO outcomes (predicting_date, actual_good, actual_frac) "
            f"VALUES ({ph}, {ph}, {ph}) "
            f"ON CONFLICT (predicting_date) DO UPDATE "
            f"SET actual_good = EXCLUDED.actual_good, actual_frac = EXCLUDED.actual_frac"
        )
    else:
        sql = (
            f"INSERT OR REPLACE INTO outcomes (predicting_date, actual_good, actual_frac) "
            f"VALUES ({ph}, {ph}, {ph})"
        )
    try:
        cur = con.cursor()
        cur.execute(sql, (predicting_date, int(actual_good), float(actual_frac)))
        con.commit()
    finally:
        con.close()


def backfill_outcomes(daily_quality: "pd.Series", db_path: str = DEFAULT_SQLITE) -> int:
    """
    Backfill the outcomes table from a pd.Series indexed by datetime.date.
    daily_quality values are fractions (0-1); threshold from the first prediction row is used.
    Returns the number of rows upserted.
    """
    if daily_quality.empty:
        return 0

    con, backend = _connect(db_path)
    ph = _ph(backend)

    try:
        cur = con.cursor()
        cur.execute("SELECT threshold FROM predictions LIMIT 1")
        row = cur.fetchone()
        threshold = row[0] if row else 0.30

        rows = [
            (str(d), int(float(v) >= threshold), float(v))
            for d, v in daily_quality.items()
        ]

        if backend == "postgres":
            sql = (
                f"INSERT INTO outcomes (predicting_date, actual_good, actual_frac) "
                f"VALUES ({ph}, {ph}, {ph}) "
                f"ON CONFLICT (predicting_date) DO UPDATE "
                f"SET actual_good = EXCLUDED.actual_good, actual_frac = EXCLUDED.actual_frac"
            )
        else:
            sql = (
                f"INSERT OR REPLACE INTO outcomes (predicting_date, actual_good, actual_frac) "
                f"VALUES ({ph}, {ph}, {ph})"
            )

        cur.executemany(sql, rows)
        con.commit()
    finally:
        con.close()
    return len(rows)


def load_history(
    db_path: str = DEFAULT_SQLITE,
    days: Optional[int] = 30,
    snapshot_hour: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load recent prediction history joined with outcomes (where available).

    Parameters
    ----------
    days          : look-back window in calendar days (None = all time)
    snapshot_hour : if given, filter to predictions made at this UTC hour

    Returns a DataFrame with columns:
        run_ts, snapshot_dt, predicting_date, probability, good,
        actual_good (NaN when not yet known), actual_frac (NaN when not yet known)
    """
    backend = _backend()

    # SQLite: don't bother if the file doesn't exist yet
    if backend == "sqlite" and not os.path.exists(db_path):
        return pd.DataFrame()

    ph = _ph(backend)
    where_clauses = []
    params: list = []

    if days is not None:
        if backend == "postgres":
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            where_clauses.append(f"p.predicting_date >= {ph}")
            params.append(cutoff)
        else:
            where_clauses.append(f"p.predicting_date >= date('now', {ph})")
            params.append(f"-{days} days")

    if snapshot_hour is not None:
        if backend == "postgres":
            where_clauses.append(
                f"EXTRACT(HOUR FROM p.snapshot_dt::timestamp)::integer = {ph}"
            )
        else:
            where_clauses.append(f"CAST(strftime('%H', p.snapshot_dt) AS INTEGER) = {ph}")
        params.append(snapshot_hour)

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT
            p.run_ts,
            p.snapshot_dt,
            p.predicting_date,
            p.probability,
            p.good,
            p.threshold,
            o.actual_good,
            o.actual_frac
        FROM predictions p
        LEFT JOIN outcomes o ON o.predicting_date = p.predicting_date
        {where}
        ORDER BY p.predicting_date, p.snapshot_dt
    """

    con, _ = _connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(sql, params) if params else cur.execute(sql)
        cols = [desc[0] for desc in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    except Exception:
        df = pd.DataFrame()
    finally:
        con.close()

    return df


def accuracy_summary(db_path: str = DEFAULT_SQLITE, days: int = 30) -> dict:
    """
    Compute prediction accuracy over the last `days` calendar days.
    Uses only the most-recent snapshot per target date (most informative).
    Returns a dict with keys: n_evaluated, accuracy, precision, recall.
    """
    df = load_history(db_path=db_path, days=days)
    if df.empty:
        return {}

    df = df.sort_values("snapshot_dt").groupby("predicting_date").last().reset_index()
    evaluated = df.dropna(subset=["actual_good"])
    if evaluated.empty:
        return {}

    n = len(evaluated)
    pred = evaluated["good"].astype(int)
    actual = evaluated["actual_good"].astype(int)

    correct = (pred == actual).sum()
    tp = ((pred == 1) & (actual == 1)).sum()
    fp = ((pred == 1) & (actual == 0)).sum()
    fn = ((pred == 0) & (actual == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall    = tp / (tp + fn) if (tp + fn) > 0 else None

    return {
        "n_evaluated": int(n),
        "accuracy":    round(correct / n, 3),
        "precision":   round(precision, 3) if precision is not None else None,
        "recall":      round(recall,    3) if recall    is not None else None,
    }
