import json
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _mock_response(start_date: str, n_hours: int = 4):
    """Build a fake Open-Meteo archive API response."""
    import pandas as _pd
    times = _pd.date_range(start_date + " 00:00", periods=n_hours, freq="1h")
    data = {
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "temperature_2m":        [15.0] * n_hours,
            "wind_speed_10m":        [5.0]  * n_hours,
            "wind_direction_10m":    [180.0] * n_hours,
            "wind_gusts_10m":        [7.0]  * n_hours,
            "cloud_cover":           [30.0] * n_hours,
            "boundary_layer_height": [800.0] * n_hours,
            "direct_radiation":      [400.0] * n_hours,
        },
        "timezone": "Europe/Vienna",
        "utc_offset_seconds": 3600,
    }
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = data
    return mock


def test_fetch_historical_chunk_returns_dataframe():
    with patch("requests.get", return_value=_mock_response("2026-04-05")) as mock_get:
        from input.open_meteo_historical import fetch_historical_chunk
        df = fetch_historical_chunk(47.8, 13.7, date(2026, 4, 5), date(2026, 4, 5))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "wind_speed" in df.columns
    assert "blh" in df.columns


def test_fetch_historical_chunk_uses_kn_and_timezone():
    """Must pass wind_speed_unit=kn and timezone=auto to the API."""
    with patch("requests.get", return_value=_mock_response("2026-04-05")) as mock_get:
        from input.open_meteo_historical import fetch_historical_chunk
        fetch_historical_chunk(47.8, 13.7, date(2026, 4, 5), date(2026, 4, 5))
    call_params = mock_get.call_args[1]["params"]
    assert call_params.get("wind_speed_unit") == "kn"
    assert call_params.get("timezone") == "auto"


def test_fetch_historical_chunk_returns_empty_on_error():
    mock = MagicMock()
    mock.raise_for_status.side_effect = Exception("network error")
    with patch("requests.get", return_value=mock):
        from input.open_meteo_historical import fetch_historical_chunk
        df = fetch_historical_chunk(47.8, 13.7, date(2026, 4, 5), date(2026, 4, 5))
    assert df.empty


def test_fetch_historical_range_chunks_90_days():
    """A range > 90 days should result in multiple API calls."""
    with patch("input.open_meteo_historical.fetch_historical_chunk",
               return_value=pd.DataFrame()) as mock_chunk:
        from input.open_meteo_historical import fetch_historical_range
        fetch_historical_range(47.8, 13.7, date(2024, 1, 1), date(2024, 6, 1))
    # 5 months > 90 days → at least 2 chunks
    assert mock_chunk.call_count >= 2
