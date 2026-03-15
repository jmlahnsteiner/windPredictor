import math
import numpy as np
import pandas as pd
import pytest
from utils.circular import circular_std


def test_perfectly_steady_direction():
    # All readings from the same direction → std ≈ 0
    angles = pd.Series([180.0] * 10)
    assert circular_std(angles) < 1.0


def test_opposite_directions_high_std():
    # N and S alternating → very high circular std
    angles = pd.Series([0.0, 180.0] * 5)
    assert circular_std(angles) > 80.0


def test_known_value():
    # 0°, 90°, 180°, 270° → mean resultant length R ≈ 0, std ≈ high
    angles = pd.Series([0.0, 90.0, 180.0, 270.0])
    result = circular_std(angles)
    assert result > 100.0


def test_accepts_list():
    result = circular_std([10.0, 20.0, 30.0])
    assert isinstance(result, float)
    assert result < 15.0


def test_single_value_returns_zero():
    result = circular_std([45.0])
    assert result == pytest.approx(0.0, abs=1e-6)


def test_empty_series_returns_nan():
    result = circular_std(pd.Series([], dtype=float))
    assert math.isnan(result)


def test_series_with_nans_ignored():
    angles = pd.Series([90.0, float('nan'), 90.0, 90.0])
    result = circular_std(angles)
    assert result < 1.0
