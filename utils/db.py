"""utils/db.py — Shared database connection helpers (Supabase / SQLite fallback)."""
import os
import sqlite3
from pathlib import Path

_ROOT = Path(__file__).parent.parent
DEFAULT_SQLITE = str(_ROOT / "local.db")

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass


def backend() -> str:
    """Return 'postgres' if SUPABASE_DB_URL is set, else 'sqlite'."""
    return "postgres" if os.environ.get("SUPABASE_DB_URL") else "sqlite"


def placeholder(bk: str) -> str:
    """SQL parameter placeholder for the given backend."""
    return "%s" if bk == "postgres" else "?"


def get_connection(db_path: str = DEFAULT_SQLITE):
    """
    Return (connection, backend_string).

    db_path is only used for the SQLite backend; Postgres reads from
    the SUPABASE_DB_URL environment variable.
    """
    bk = backend()
    if bk == "postgres":
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "psycopg2-binary is required for the Supabase backend. "
                "Run: pip install psycopg2-binary"
            ) from e
        return psycopg2.connect(os.environ["SUPABASE_DB_URL"]), "postgres"

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    return sqlite3.connect(db_path), "sqlite"
