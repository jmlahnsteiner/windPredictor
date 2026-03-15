"""
supabase/migrate_parquet.py — One-time migration of data.parquet → weather_readings.

Prerequisites:
  1. Run supabase/schema_additions.sql in Supabase
  2. Set SUPABASE_DB_URL in your environment (or .env file)

Usage:
    python supabase/migrate_parquet.py
    python supabase/migrate_parquet.py --dry-run   # count rows only
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

PARQUET_PATH = os.path.join(_ROOT, "data.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate data.parquet → weather_readings")
    parser.add_argument("--dry-run", action="store_true", help="Count rows only, do not write")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_ROOT, ".env"))
    except ImportError:
        pass

    if not os.path.exists(PARQUET_PATH):
        print(f"ERROR: {PARQUET_PATH} not found.")
        sys.exit(1)

    import pandas as pd
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df):,} rows  ({df.index.min().date()} → {df.index.max().date()})")

    if args.dry_run:
        print(f"[dry-run] Would upsert {len(df):,} rows into weather_readings. Exiting.")
        return

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        print("ERROR: SUPABASE_DB_URL is not set. Add it to .env or export it.")
        sys.exit(1)

    from input.weather_store import upsert_readings
    print("Upserting rows into weather_readings (this may take a minute) …")
    n = upsert_readings(df)
    print(f"Done: {n:,} rows upserted.")
    print("\nNext steps:")
    print("  git rm --cached data.parquet")
    print("  echo 'data.parquet' >> .gitignore")


if __name__ == "__main__":
    main()
