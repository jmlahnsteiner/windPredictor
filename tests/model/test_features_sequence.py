import numpy as np
import pandas as pd
import pytest


def _make_station_df(hours=48, freq="5min"):
    n = hours * (60 // int(freq.rstrip("min")))
    idx = pd.date_range("2026-04-04 00:00", periods=n, freq=freq)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "wind_speed":        rng.uniform(3, 8, n),
        "wind_direction":    [180.0] * n,
        "temperature":       [15.0]  * n,
        "pressure_relative": [1013.0] * n,
        "humidity":          [60.0]  * n,
        "solar":             [200.0] * n,
    }, index=idx)


def _make_nwp_df(target_date="2026-04-05"):
    idx = pd.date_range(
        f"{target_date} 08:00", periods=8, freq="1h", tz="Europe/Vienna"
    )
    return pd.DataFrame({
        "temperature": [18.0] * 8, "wind_speed": [6.0] * 8,
        "wind_direction": [200.0] * 8, "wind_gust": [9.0] * 8,
        "cloud_cover": [20.0] * 8, "blh": [1000.0] * 8,
        "direct_radiation": [500.0] * 8,
    }, index=idx)


def _cfg():
    return {
        "sailing": {
            "window_start": "08:00",
            "window_end": "16:00",
            "wind_speed_min": 2.0,
            "wind_speed_max": 20.0,
            "wind_dir_consistency_max": 60.0,
            "min_temperature": 5.0,
        },
        "prediction": {"min_good_fraction": 0.25},
    }


def test_build_sequence_shape():
    from model.features_sequence import build_sequence, SEQ_FEATURES
    df = _make_station_df()
    snap_dt = pd.Timestamp("2026-04-05 06:00")
    seq = build_sequence(df, snap_dt)
    assert seq.ndim == 2
    assert seq.shape[1] == SEQ_FEATURES
    assert seq.shape[0] == 24  # always 24h


def test_build_sequence_no_nans():
    from model.features_sequence import build_sequence
    df = _make_station_df()
    snap_dt = pd.Timestamp("2026-04-05 06:00")
    seq = build_sequence(df, snap_dt)
    assert not np.any(np.isnan(seq))


def test_build_sequence_sparse_data():
    from model.features_sequence import build_sequence, SEQ_FEATURES
    df = _make_station_df(hours=6)
    snap_dt = pd.Timestamp("2026-04-04 06:00")
    seq = build_sequence(df, snap_dt)
    assert seq.shape[1] == SEQ_FEATURES


def test_build_nwp_context_shape():
    from model.features_sequence import build_nwp_context, CONTEXT_FEATURES
    nwp_df = _make_nwp_df("2026-04-05")
    snap_dt = pd.Timestamp("2026-04-05 06:00")
    ctx = build_nwp_context(nwp_df, snap_dt, _cfg())
    assert ctx.shape == (CONTEXT_FEATURES,)


def test_build_nwp_context_populated_when_data_present():
    from model.features_sequence import build_nwp_context
    nwp_df = _make_nwp_df("2026-04-05")
    snap_dt = pd.Timestamp("2026-04-05 06:00")
    ctx = build_nwp_context(nwp_df, snap_dt, _cfg())
    assert not np.any(np.isnan(ctx))
    assert ctx[0] != 0.0, "NWP wind speed mean should not be zero"


def test_build_sequence_training_pairs_shapes():
    from model.features_sequence import build_sequence_training_pairs, SEQ_FEATURES, CONTEXT_FEATURES
    df = _make_station_df(hours=5 * 24)
    sequences, contexts, labels = build_sequence_training_pairs(df, _cfg(), nwp_df=None)
    assert sequences.ndim == 3
    assert contexts.ndim == 2
    assert labels.ndim == 1
    assert sequences.shape[0] == contexts.shape[0] == labels.shape[0]
    assert sequences.shape[2] == SEQ_FEATURES
    assert contexts.shape[1] == CONTEXT_FEATURES
