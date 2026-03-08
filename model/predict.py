"""
model/predict.py — Load saved weights and predict sailing probability.

Produces a JSON result for each configured snapshot time. Can be run
standalone (reads live data from parquet) or imported as a module.

Usage:
    python model/predict.py                   # all snapshots for today
    python model/predict.py 2026-03-07        # all snapshots for a specific date
    python model/predict.py 2026-03-07 18:00  # single snapshot
"""

import json
import os
import sys
import tomllib
from datetime import date, datetime, timedelta
from typing import Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.features import extract_snapshot_features, _target_date

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "..", "config.toml")


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def predict_snapshot(
    df: pd.DataFrame,
    snap_dt: pd.Timestamp,
    bundle: dict,
    cfg: dict,
) -> dict:
    """
    Predict sailing probability for the sailing window following snap_dt.

    Returns a dict with probability, target date, and metadata.
    """
    feature_names: list[str] = bundle["feature_names"]
    feature_medians: dict = bundle["feature_medians"]
    clf = bundle["model"]

    features = extract_snapshot_features(df, snap_dt)

    if features is None:
        return {
            "snapshot": snap_dt.isoformat(),
            "error": "Insufficient data at this snapshot time",
        }

    X = pd.DataFrame([features])[feature_names]
    # Fill missing values with the medians seen during training
    for col in feature_names:
        if X[col].isna().any():
            X[col] = X[col].fillna(feature_medians.get(col, 0.0))

    proba = clf.predict_proba(X)[0]
    if len(proba) == 1:
        # Model was trained with only one class present in the data.
        prob = float(proba[0]) if int(clf.classes_[0]) == 1 else 0.0
    else:
        prob = float(proba[1])
    tgt = _target_date(snap_dt, cfg["sailing"]["window_start"])

    result = {
        "snapshot": snap_dt.isoformat(),
        "predicting_date": str(tgt),
        "sailing_window": f"{cfg['sailing']['window_start']}–{cfg['sailing']['window_end']}",
        "probability": round(prob, 3),
        "good": prob >= cfg["prediction"]["min_good_fraction"],
        "threshold": cfg["prediction"]["min_good_fraction"],
    }

    # Wind distribution during the target sailing window (available for
    # historical dates; silently absent for future predictions).
    window_data = _sailing_window_data(df, tgt, cfg)
    if window_data:
        result["window_wind"] = window_data

    return result


def _sailing_window_data(
    df: pd.DataFrame,
    tgt_date,
    cfg: dict,
) -> dict:
    """
    Extract wind speed and direction measurements from the sailing window
    of tgt_date.  Returns {} when the window data is not yet available.
    """
    sc = cfg["sailing"]
    try:
        day_df = df.loc[str(tgt_date)]
    except KeyError:
        return {}

    window = day_df.between_time(sc["window_start"], sc["window_end"])
    needed = window[["wind_speed", "wind_direction"]].dropna()
    if len(needed) < 3:
        return {}

    # Align gusts to the same index as speed/direction; gaps become None.
    gusts = window["wind_gust"].reindex(needed.index) if "wind_gust" in window.columns else pd.Series(index=needed.index, dtype=float)

    return {
        "times":          [t.strftime("%H:%M") for t in needed.index],
        "speeds_kn":      [round(float(v), 1) for v in needed["wind_speed"]],
        "directions_deg": [round(float(v))     for v in needed["wind_direction"]],
        "gusts_kn":       [round(float(v), 1) if pd.notna(v) else None for v in gusts],
    }


def _enrich_with_nwp(results: list[dict], cfg: dict) -> None:
    """
    Fetch Open-Meteo NWP for the configured location and attach an
    ``nwp_forecast`` dict to each result.  Mutates results in-place.
    No-op when [location] is absent from config or the fetch fails.
    """
    loc = cfg.get("location", {})
    lat = loc.get("lat")
    lon = loc.get("lon")
    if lat is None or lon is None:
        return

    try:
        from input.open_meteo import fetch_forecast, sailing_window_stats
    except ImportError:
        print("  [!] input/open_meteo.py not found — skipping NWP enrichment")
        return

    print("Fetching NWP forecast from Open-Meteo …", flush=True)
    nwp_df = fetch_forecast(lat, lon)
    if nwp_df.empty:
        return

    sc = cfg["sailing"]
    for result in results:
        if "error" in result or "predicting_date" not in result:
            continue
        tgt_date = date.fromisoformat(result["predicting_date"])
        stats = sailing_window_stats(nwp_df, tgt_date, sc["window_start"], sc["window_end"])
        if stats:
            result["nwp_forecast"] = stats


def predict_all(
    df: pd.DataFrame,
    ref_date: Optional[date] = None,
    config_path: str = DEFAULT_CONFIG,
) -> list[dict]:
    """
    Run all configured snapshot predictions relative to ref_date (default: today).
    Returns a list of result dicts, one per snapshot.
    """
    cfg = load_config(config_path)
    root = os.path.dirname(os.path.abspath(config_path))
    model_path = os.path.join(root, cfg["paths"]["model_file"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run model/train.py first."
        )

    bundle = joblib.load(model_path)

    if ref_date is None:
        ref_date = date.today()

    results = []
    for snap_str in cfg["prediction"]["snapshots"]:
        h, m = map(int, snap_str.split(":"))
        snap_dt = pd.Timestamp(
            year=ref_date.year, month=ref_date.month, day=ref_date.day,
            hour=h, minute=m,
        )
        result = predict_snapshot(df, snap_dt, bundle, cfg)
        results.append(result)

    _enrich_with_nwp(results, cfg)
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    config_path = DEFAULT_CONFIG
    cfg = load_config(config_path)
    root = os.path.dirname(os.path.abspath(config_path))

    # Load data
    df = pd.read_parquet(os.path.join(root, cfg["paths"]["data_parquet"]))

    # Parse args
    if len(args) == 0:
        ref_date = date.today()
        results = predict_all(df, ref_date, config_path)

    elif len(args) == 1:
        ref_date = datetime.strptime(args[0], "%Y-%m-%d").date()
        results = predict_all(df, ref_date, config_path)

    elif len(args) == 2:
        ref_date = datetime.strptime(args[0], "%Y-%m-%d").date()
        h, m = map(int, args[1].split(":"))
        snap_dt = pd.Timestamp(
            year=ref_date.year, month=ref_date.month, day=ref_date.day,
            hour=h, minute=m,
        )
        bundle = joblib.load(os.path.join(root, cfg["paths"]["model_file"]))
        results = [predict_snapshot(df, snap_dt, bundle, cfg)]

    else:
        print("Usage: python model/predict.py [date [HH:MM]]")
        sys.exit(1)

    output = json.dumps(results, indent=2)
    print(output)

    out_path = os.path.join(root, cfg["paths"]["predictions_file"])
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\nSaved: {out_path}", file=sys.stderr)
