"""
supabase/migrate_from_sqlite.py — One-time migration of predictions.db → Supabase.

Run once after:
  1. Creating tables in Supabase (supabase/schema.sql)
  2. Setting SUPABASE_DB_URL in your environment

Usage:
    SUPABASE_DB_URL="postgresql://postgres.rvqtujbqshgwnjtovacw:PASSWORD@aws-1-eu-west-1.pooler.supabase.com:5432/postgres" \
        python supabase/migrate_from_sqlite.py

    # Or with a .env file:
    export SUPABASE_DB_URL="..."
    python supabase/migrate_from_sqlite.py
"""

import os
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQLITE_PATH = os.path.join(_ROOT, "predictions.db")


def main() -> None:
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        print("ERROR: SUPABASE_DB_URL environment variable is not set.")
        sys.exit(1)

    if not os.path.exists(SQLITE_PATH):
        print(f"ERROR: SQLite database not found at {SQLITE_PATH}")
        sys.exit(1)

    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2-binary is required. Run: pip install psycopg2-binary")
        sys.exit(1)

    print(f"Source : {SQLITE_PATH}")
    print(f"Target : {db_url[:40]}...")

    # ── Read from SQLite ───────────────────────────────────────────────────────
    src = sqlite3.connect(SQLITE_PATH)
    src.row_factory = sqlite3.Row

    predictions = src.execute(
        "SELECT run_ts, snapshot_dt, predicting_date, probability, good, threshold "
        "FROM predictions ORDER BY id"
    ).fetchall()

    outcomes = src.execute(
        "SELECT predicting_date, actual_good, actual_frac FROM outcomes"
    ).fetchall()

    src.close()

    print(f"\nFound {len(predictions)} prediction rows and {len(outcomes)} outcome rows.")

    # ── Write to Supabase ──────────────────────────────────────────────────────
    dst = psycopg2.connect(db_url)
    cur = dst.cursor()

    # predictions
    if predictions:
        cur.executemany(
            "INSERT INTO predictions "
            "(run_ts, snapshot_dt, predicting_date, probability, good, threshold) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [tuple(r) for r in predictions],
        )
        print(f"Inserted {len(predictions)} rows into predictions.")

    # outcomes
    if outcomes:
        cur.executemany(
            "INSERT INTO outcomes (predicting_date, actual_good, actual_frac) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (predicting_date) DO UPDATE "
            "SET actual_good = EXCLUDED.actual_good, actual_frac = EXCLUDED.actual_frac",
            [tuple(r) for r in outcomes],
        )
        print(f"Inserted/updated {len(outcomes)} rows into outcomes.")

    dst.commit()
    dst.close()

    print("\nMigration complete.")
    print("You can now remove predictions.db from git tracking:")
    print("  git rm --cached predictions.db")
    print("  echo 'predictions.db' >> .gitignore")


if __name__ == "__main__":
    main()
