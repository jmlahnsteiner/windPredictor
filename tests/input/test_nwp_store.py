import os
import tempfile
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest


def _make_nwp_df(n=8):
    """NWP DataFrame with tz-aware index (like fetch_forecast returns)."""
    idx = pd.date_range(
        "2026-04-05 08:00", periods=n, freq="1h",
        tz="Europe/Vienna",
    )
    return pd.DataFrame({
        "temperature":    [15.0] * n,
        "wind_speed":     [5.0]  * n,
        "wind_direction": [180.0] * n,
        "wind_gust":      [7.0]  * n,
        "cloud_cover":    [30.0] * n,
        "blh":            [800.0] * n,
        "direct_radiation": [400.0] * n,
    }, index=idx)


def _no_supabase():
    return {k: v for k, v in os.environ.items() if k != "SUPABASE_DB_URL"}


def test_upsert_and_load_roundtrip():
    df = _make_nwp_df()
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase(), clear=True):
            from input.nwp_store import upsert_nwp_readings, load_nwp_readings
            n = upsert_nwp_readings(df, db_path=db)
            assert n == 8
            result = load_nwp_readings(db_path=db)
            assert len(result) == 8
            assert isinstance(result.index, pd.DatetimeIndex)
            assert result["wind_speed"].iloc[0] == pytest.approx(5.0)


def test_upsert_idempotent():
    df = _make_nwp_df()
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase(), clear=True):
            from input.nwp_store import upsert_nwp_readings, load_nwp_readings
            upsert_nwp_readings(df, db_path=db)
            upsert_nwp_readings(df, db_path=db)
            assert len(load_nwp_readings(db_path=db)) == 8


def test_date_range_filter():
    df = _make_nwp_df()
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase(), clear=True):
            from input.nwp_store import upsert_nwp_readings, load_nwp_readings
            upsert_nwp_readings(df, db_path=db)
            result = load_nwp_readings(start=date(2026, 4, 6), db_path=db)
            assert result.empty


def test_load_empty_db():
    with patch.dict(os.environ, _no_supabase(), clear=True):
        from input.nwp_store import load_nwp_readings
        result = load_nwp_readings(db_path="/tmp/nonexistent_nwp_test.db")
        assert result.empty
        assert isinstance(result, pd.DataFrame)


def test_index_is_tz_aware():
    """Timestamps stored with tz offset round-trip back as tz-aware."""
    df = _make_nwp_df()
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase(), clear=True):
            from input.nwp_store import upsert_nwp_readings, load_nwp_readings
            upsert_nwp_readings(df, db_path=db)
            result = load_nwp_readings(db_path=db)
            assert result.index.tz is not None
