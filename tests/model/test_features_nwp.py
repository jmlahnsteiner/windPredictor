import numpy as np
import pandas as pd
import pytest


def _make_station_df():
    """Minimal station df — enough for extract_snapshot_features to return non-None."""
    idx = pd.date_range("2026-04-05 00:00", periods=48, freq="30min")
    n = len(idx)
    return pd.DataFrame({
        "wind_speed":        [5.0] * n,
        "wind_direction":    [180.0] * n,
        "wind_gust":         [7.0] * n,
        "temperature":       [15.0] * n,
        "humidity":          [60.0] * n,
        "pressure_relative": [1013.0] * n,
    }, index=idx)


def _make_nwp_df(target_date="2026-04-05"):
    """NWP df covering the sailing window of target_date (tz-aware)."""
    idx = pd.date_range(
        f"{target_date} 08:00", periods=8, freq="1h", tz="Europe/Vienna"
    )
    return pd.DataFrame({
        "temperature":    [18.0] * 8,
        "wind_speed":     [6.0]  * 8,
        "wind_direction": [200.0] * 8,
        "wind_gust":      [9.0]  * 8,
        "cloud_cover":    [20.0] * 8,
        "blh":            [1000.0] * 8,
        "direct_radiation": [500.0] * 8,
    }, index=idx)


def _cfg():
    return {
        "sailing": {
            "window_start": "08:00",
            "window_end":   "16:00",
        }
    }


def test_nwp_features_populated_when_provided():
    from model.features import extract_snapshot_features
    station_df = _make_station_df()
    nwp_df = _make_nwp_df("2026-04-05")
    cfg = _cfg()
    snap_dt = pd.Timestamp("2026-04-05 06:00")  # before 16:00 → target = same day

    feats = extract_snapshot_features(station_df, snap_dt, nwp_df=nwp_df, cfg=cfg)
    assert feats is not None

    assert feats["nwp_wind_speed_mean"] == pytest.approx(6.0)
    assert feats["nwp_wind_speed_max"]  == pytest.approx(6.0)
    assert feats["nwp_wind_gust_max"]   == pytest.approx(9.0)
    assert feats["nwp_cloud_cover_mean"] == pytest.approx(20.0)
    assert feats["nwp_blh_mean"]         == pytest.approx(1000.0)
    assert feats["nwp_direct_radiation_mean"] == pytest.approx(500.0)
    assert np.isfinite(feats["nwp_wind_dir_sin"])
    assert np.isfinite(feats["nwp_wind_dir_cos"])
    assert feats["nwp_dir_consistency"] == pytest.approx(0.0, abs=1e-3)


def test_nwp_features_nan_when_not_provided():
    from model.features import extract_snapshot_features
    station_df = _make_station_df()
    snap_dt = pd.Timestamp("2026-04-05 06:00")

    feats = extract_snapshot_features(station_df, snap_dt)
    assert feats is not None

    nwp_keys = [
        "nwp_wind_speed_mean", "nwp_wind_speed_max", "nwp_wind_gust_max",
        "nwp_wind_dir_sin", "nwp_wind_dir_cos", "nwp_dir_consistency",
        "nwp_cloud_cover_mean", "nwp_blh_mean", "nwp_direct_radiation_mean",
    ]
    for k in nwp_keys:
        assert k in feats, f"Missing key: {k}"
        assert np.isnan(feats[k]), f"Expected NaN for {k}, got {feats[k]}"


def test_nwp_target_date_after_window_end():
    """After window_end (16:00), target is next day — NWP for next day is used."""
    from model.features import extract_snapshot_features
    station_df = _make_station_df()
    nwp_df = _make_nwp_df("2026-04-06")  # NWP for next day
    cfg = _cfg()
    snap_dt = pd.Timestamp("2026-04-05 18:00")  # after 16:00 → target = 2026-04-06

    feats = extract_snapshot_features(station_df, snap_dt, nwp_df=nwp_df, cfg=cfg)
    assert feats is not None
    assert feats["nwp_wind_speed_mean"] == pytest.approx(6.0)


def test_existing_features_unchanged():
    """Adding NWP params must not break any of the original station features."""
    from model.features import extract_snapshot_features
    station_df = _make_station_df()
    snap_dt = pd.Timestamp("2026-04-05 06:00")

    feats_before = extract_snapshot_features(station_df, snap_dt)
    feats_after  = extract_snapshot_features(station_df, snap_dt, nwp_df=None, cfg=None)

    original_keys = [k for k in feats_before if not k.startswith("nwp_")]
    for k in original_keys:
        v_before = feats_before[k]
        v_after  = feats_after[k]
        if v_before is None or (isinstance(v_before, float) and np.isnan(v_before)):
            assert v_after is None or np.isnan(v_after)
        else:
            assert v_after == pytest.approx(v_before, rel=1e-6), f"Key {k} changed"
