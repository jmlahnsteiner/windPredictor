import os
import tempfile
import sqlite3
import pytest
from unittest.mock import patch


def test_backend_returns_sqlite_when_no_env():
    with patch.dict(os.environ, {}, clear=True):
        # Remove SUPABASE_DB_URL if present
        env = {k: v for k, v in os.environ.items() if k != "SUPABASE_DB_URL"}
        with patch.dict(os.environ, env, clear=True):
            from utils.db import backend
            assert backend() == "sqlite"


def test_backend_returns_postgres_when_env_set():
    with patch.dict(os.environ, {"SUPABASE_DB_URL": "postgresql://fake"}):
        from utils import db
        import importlib
        importlib.reload(db)  # re-evaluate module-level
        from utils.db import backend
        assert backend() == "postgres"


def test_placeholder_sqlite():
    from utils.db import placeholder
    assert placeholder("sqlite") == "?"


def test_placeholder_postgres():
    from utils.db import placeholder
    assert placeholder("postgres") == "%s"


def test_get_connection_sqlite_creates_file():
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test.db")
        with patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                      if k != "SUPABASE_DB_URL"}, clear=True):
            from utils.db import get_connection
            con, bk = get_connection(db_path)
            assert bk == "sqlite"
            assert isinstance(con, sqlite3.Connection)
            con.close()
        assert os.path.exists(db_path)
