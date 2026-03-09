"""
model/history.py — Persist every prediction run to a local SQLite database.

Schema
------
predictions
    id              INTEGER PRIMARY KEY AUTOINCREMENT
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
    record_predictions(results, db_path="predictions.db")
    df = load_history(db_path="predictions.db", days=30)
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

_SCHEMA = """
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


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(_SCHEMA)
    con.commit()
    return con


def record_predictions(results: list[dict], db_path: str = "predictions.db") -> int:
    """
    Insert prediction results (as returned by predict_all) into the database.

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

    with _connect(db_path) as con:
        con.executemany(
            "INSERT INTO predictions "
            "(run_ts, snapshot_dt, predicting_date, probability, good, threshold) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def record_outcome(
    predicting_date: str,
    actual_good: bool,
    actual_frac: float,
    db_path: str = "predictions.db",
) -> None:
    """
    Store the ground-truth outcome for a date so accuracy can be computed later.
    Called automatically by record_predictions_with_actuals when daily targets
    are available (i.e. the day has already passed).
    """
    with _connect(db_path) as con:
        con.execute(
            "INSERT OR REPLACE INTO outcomes (predicting_date, actual_good, actual_frac) "
            "VALUES (?, ?, ?)",
            (predicting_date, int(actual_good), float(actual_frac)),
        )


def backfill_outcomes(daily_quality: "pd.Series", db_path: str = "predictions.db") -> int:
    """
    Backfill the outcomes table from a pd.Series indexed by datetime.date.
    daily_quality values are fractions (0-1); threshold from the first prediction row is used.

    Returns the number of rows upserted.
    """
    if daily_quality.empty:
        return 0

    # Determine threshold from DB (use most common value)
    with _connect(db_path) as con:
        row = con.execute("SELECT threshold FROM predictions LIMIT 1").fetchone()
    threshold = row[0] if row else 0.30

    rows = [
        (str(d), int(float(v) >= threshold), float(v))
        for d, v in daily_quality.items()
    ]
    with _connect(db_path) as con:
        con.executemany(
            "INSERT OR REPLACE INTO outcomes (predicting_date, actual_good, actual_frac) "
            "VALUES (?, ?, ?)",
            rows,
        )
    return len(rows)


def load_history(
    db_path: str = "predictions.db",
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
    if not os.path.exists(db_path):
        return pd.DataFrame()

    where_clauses = []
    params: list = []

    if days is not None:
        where_clauses.append(
            "p.predicting_date >= date('now', ?)"
        )
        params.append(f"-{days} days")

    if snapshot_hour is not None:
        where_clauses.append("CAST(strftime('%H', p.snapshot_dt) AS INTEGER) = ?")
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
    with _connect(db_path) as con:
        df = pd.read_sql_query(sql, con, params=params)

    return df


def backfill_predictions_from_history(
    df: "pd.DataFrame",
    cfg: dict,
    bundle: dict,
    db_path: str = "predictions.db",
) -> int:
    """
    For every date in the outcomes table that has no row in predictions,
    run the model retrospectively and insert the results.

    Uses only data available up to each snapshot time (no lookahead leakage).
    Returns the number of rows inserted.
    """
    if not os.path.exists(db_path):
        return 0

    with _connect(db_path) as con:
        outcome_dates = {
            row[0] for row in con.execute("SELECT predicting_date FROM outcomes").fetchall()
        }
        pred_dates = {
            row[0] for row in con.execute(
                "SELECT DISTINCT predicting_date FROM predictions"
            ).fetchall()
        }

    missing = sorted(outcome_dates - pred_dates)
    if not missing:
        return 0

    # Inline imports to avoid circular dependency at module load time.
    from model.predict import predict_snapshot       # noqa: PLC0415
    from model.features import _target_date          # noqa: PLC0415
    import pandas as _pd                             # noqa: PLC0415

    run_ts = datetime.now(timezone.utc).isoformat()
    rows: list[tuple] = []

    for date_str in missing:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        for snap_str in cfg["prediction"]["snapshots"]:
            h, m = map(int, snap_str.split(":"))
            snap_dt = _pd.Timestamp(
                year=d.year, month=d.month, day=d.day, hour=h, minute=m,
            )
            # Only use snapshots whose predicted date matches the one we're backfilling.
            if str(_target_date(snap_dt, cfg["sailing"]["window_start"])) != date_str:
                continue
            result = predict_snapshot(df, snap_dt, bundle, cfg)
            if "error" in result or "predicting_date" not in result:
                continue
            rows.append((
                run_ts,
                result["snapshot"],
                result["predicting_date"],
                float(result["probability"]),
                int(bool(result["good"])),
                float(result.get("threshold", 0.30)),
            ))

    if not rows:
        return 0

    with _connect(db_path) as con:
        con.executemany(
            "INSERT INTO predictions "
            "(run_ts, snapshot_dt, predicting_date, probability, good, threshold) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def accuracy_summary(db_path: str = "predictions.db", days: int = 30) -> dict:
    """
    Compute prediction accuracy over the last `days` calendar days.

    Uses only the most-recent snapshot per target date (most informative).
    Returns a dict with keys: n_evaluated, accuracy, precision, recall.
    Returns {} if there are no evaluated days.
    """
    df = load_history(db_path=db_path, days=days)
    if df.empty:
        return {}

    # Keep most-recent prediction per predicting_date
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
