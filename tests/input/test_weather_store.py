import os
import tempfile
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest


def _make_df(n=5) -> pd.DataFrame:
    """Create a minimal weather DataFrame matching the parquet schema."""
    import numpy as np
    idx = pd.date_range("2026-03-10 08:00", periods=n, freq="5min")
    return pd.DataFrame({
        "wind_speed":       [3.0, 4.0, 5.0, 4.5, 3.5][:n],
        "wind_direction":   [180.0] * n,
        "wind_gust":        [5.0] * n,
        "temperature":      [15.0] * n,
        "humidity":         [60.0] * n,
        "pressure_relative":[1013.0] * n,
        "water_temperature":[12.0] * n,
    }, index=idx)


def _no_supabase_env():
    return {k: v for k, v in os.environ.items() if k != "SUPABASE_DB_URL"}


def test_upsert_and_load_roundtrip():
    """Upserted rows come back with correct types and index."""
    df = _make_df()
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase_env(), clear=True):
            from input.weather_store import upsert_readings, load_weather_readings
            n = upsert_readings(df, db_path=db)
            assert n == 5

            result = load_weather_readings(db_path=db)
            assert len(result) == 5
            assert isinstance(result.index, pd.DatetimeIndex)
            assert result["wind_speed"].iloc[0] == pytest.approx(3.0)


def test_upsert_is_idempotent():
    """Upserting the same rows twice doesn't create duplicates."""
    df = _make_df()
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase_env(), clear=True):
            from input.weather_store import upsert_readings, load_weather_readings
            upsert_readings(df, db_path=db)
            upsert_readings(df, db_path=db)
            result = load_weather_readings(db_path=db)
            assert len(result) == 5


def test_date_range_filtering():
    """load_weather_readings respects start/end date bounds."""
    df = _make_df(5)
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase_env(), clear=True):
            from input.weather_store import upsert_readings, load_weather_readings
            upsert_readings(df, db_path=db)
            result = load_weather_readings(
                start=date(2026, 3, 11),   # after all data
                db_path=db,
            )
            assert result.empty


def test_empty_df_returns_empty():
    """Loading from a nonexistent DB returns empty DataFrame."""
    with patch.dict(os.environ, _no_supabase_env(), clear=True):
        from input.weather_store import load_weather_readings
        result = load_weather_readings(db_path="/tmp/nonexistent_test.db")
        assert result.empty
        assert isinstance(result, pd.DataFrame)


def test_load_all_when_no_bounds():
    """None bounds load all available rows."""
    df = _make_df(5)
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        with patch.dict(os.environ, _no_supabase_env(), clear=True):
            from input.weather_store import upsert_readings, load_weather_readings
            upsert_readings(df, db_path=db)
            result = load_weather_readings(db_path=db)
            assert len(result) == 5
