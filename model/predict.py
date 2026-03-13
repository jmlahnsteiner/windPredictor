"""
model/predict.py — Load saved weights and predict sailing probability.

Each run predicts from the current moment forward through 4 days.
Extended days (day+2, day+3) use a probability decay to reflect
decreasing forecast skill at longer lead times.

Usage:
    python model/predict.py                   # predict from now
    python model/predict.py 2026-03-07 18:00  # predict from a specific time
"""

import json
import math
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


# Probability decay per calendar-day lead time from today.
# Days 0 and 1 use the ML model at full confidence.
# Days 2 and 3 are scaled down to signal increasing uncertainty.
_FORECAST_DECAY: dict[int, float] = {0: 1.0, 1: 1.0, 2: 0.82, 3: 0.64}

# Ordered from highest to lowest score threshold
_CONDITION_LEVELS: list[tuple[int, str, str]] = [
    (90, "Excellent",        "⛵"),
    (75, "Great",            "⛵"),
    (60, "Good",             "⛵"),
    (45, "Fair",             "⛵"),
    (30, "Marginal",         "〰"),
    (15, "Very light",       "💨"),
    ( 0, "No wind",          "🌫"),
]


def _condition_rating(result: dict, cfg: dict) -> tuple[int, str, str]:
    """
    Compute a continuous condition score (0–100), human-readable label, and icon.

    When observed window_wind data is present (historical/current day), the score
    is derived from actual wind speed and directional consistency.
    For future days without observed data, the model probability is mapped directly
    to a score.

    Returns (score, label, icon).
    """
    sc       = cfg.get("sailing", {})
    wind_min = sc.get("wind_speed_min", 2.0)
    wind_max = sc.get("wind_speed_max", 12.0)
    dir_max  = sc.get("wind_dir_consistency_max", 45.0)
    optimal  = (wind_min + wind_max) / 2.0

    ww     = result.get("window_wind", {})
    speeds = ww.get("speeds_kn", [])
    dirs   = ww.get("directions_deg", [])
    gusts  = [g for g in ww.get("gusts_kn", []) if g is not None]

    if speeds:
        mean_spd = sum(speeds) / len(speeds)
        max_gust = max(gusts) if gusts else max(speeds)

        # Hard overrides for dangerous/very-strong conditions
        if max_gust > 30 or mean_spd > 25:
            return 5,  "Storm",          "⚡"
        if max_gust > 22 or mean_spd > 18:
            return 20, "Strong / gusty", "⚠"

        # Speed score: bell curve peaking at the optimal mid-range speed
        half_range = max((wind_max - wind_min) / 2.0, 0.01)
        if mean_spd < wind_min:
            speed_score = (mean_spd / max(wind_min, 0.01)) * 45.0
        elif mean_spd <= optimal:
            speed_score = 45.0 + ((mean_spd - wind_min) / half_range) * 55.0
        elif mean_spd <= wind_max:
            speed_score = 100.0 - ((mean_spd - optimal) / half_range) * 55.0
        else:
            speed_score = max(0.0, 45.0 - (mean_spd - wind_max) * 5.0)

        # Direction consistency modifier: 0.6 (variable) → 1.0 (rock-steady)
        if len(dirs) >= 3:
            rad = [math.radians(d) for d in dirs]
            R   = math.hypot(
                sum(math.sin(r) for r in rad) / len(rad),
                sum(math.cos(r) for r in rad) / len(rad),
            )
            circ_std   = math.degrees(math.sqrt(-2 * math.log(max(R, 1e-9))))
            dir_factor = max(0.0, 1.0 - circ_std / dir_max)
        else:
            dir_factor = 0.5

        score = int(min(100, max(0, round(speed_score * (0.6 + 0.4 * dir_factor)))))
    else:
        # No observed data: map model probability directly to a 0-100 score
        score = int(round(result.get("probability", 0.0) * 100))

    label, icon = "No wind", "🌫"
    for threshold, lbl, icn in _CONDITION_LEVELS:
        if score >= threshold:
            label, icon = lbl, icn
            break

    return score, label, icon


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
    for col in feature_names:
        if X[col].isna().any():
            X[col] = X[col].fillna(feature_medians.get(col, 0.0))

    proba = clf.predict_proba(X)[0]
    if len(proba) == 1:
        prob = float(proba[0]) if int(clf.classes_[0]) == 1 else 0.0
    else:
        prob = float(proba[1])
    tgt = _target_date(snap_dt, cfg["sailing"]["window_end"])

    result = {
        "snapshot": snap_dt.isoformat(),
        "predicting_date": str(tgt),
        "sailing_window": f"{cfg['sailing']['window_start']}–{cfg['sailing']['window_end']}",
        "probability": round(prob, 3),
        "good": prob >= cfg["prediction"]["min_good_fraction"],
        "threshold": cfg["prediction"]["min_good_fraction"],
    }

    window_data = _sailing_window_data(df, tgt, cfg)
    if window_data:
        result["window_wind"] = window_data

    c_score, c_label, c_icon = _condition_rating(result, cfg)
    result["condition_score"]  = c_score
    result["condition_label"]  = c_label
    result["condition_icon"]   = c_icon
    result["condition_source"] = "observed" if window_data else "forecast"

    return result


def _sailing_window_data(df: pd.DataFrame, tgt_date, cfg: dict) -> dict:
    """Extract wind measurements from the sailing window of tgt_date."""
    sc = cfg["sailing"]
    try:
        day_df = df.loc[str(tgt_date)]
    except KeyError:
        return {}

    window = day_df.between_time(sc["window_start"], sc["window_end"])
    needed = window[["wind_speed", "wind_direction"]].dropna()
    if len(needed) < 3:
        return {}

    gusts = (window["wind_gust"].reindex(needed.index)
             if "wind_gust" in window.columns
             else pd.Series(index=needed.index, dtype=float))

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
    lat, lon = loc.get("lat"), loc.get("lon")
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


def predict_now(
    df: pd.DataFrame,
    config_path: str = DEFAULT_CONFIG,
    snap_dt: Optional[pd.Timestamp] = None,
) -> list[dict]:
    """
    Predict sailing conditions from the current moment (or a given snap_dt).

    Returns a list containing:
    - One direct ML prediction for today (before window_end) or tomorrow (after window_end)
    - Extended predictions for up to day+3 with probability decay

    During the sailing window the model incorporates live station readings up to snap_dt.
    After window_end, today's completed window data is surfaced as an observed-only entry.
    Direct predictions are saved to history; extended/observed ones are display-only.
    """
    cfg = load_config(config_path)
    root = os.path.dirname(os.path.abspath(config_path))
    model_path = os.path.join(root, cfg["paths"]["model_file"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run model/train.py first."
        )

    bundle = joblib.load(model_path)

    if snap_dt is None:
        snap_dt = pd.Timestamp.now().floor("h")

    today = snap_dt.date()

    # Direct ML prediction for this snapshot time
    result = predict_snapshot(df, snap_dt, bundle, cfg)
    if "error" in result:
        return [result]

    results: list[dict] = [result]
    direct_date = date.fromisoformat(result["predicting_date"])

    # When predicting tomorrow (after window_end), also surface today's observed
    # window data so today appears on the page with real wind readings.
    if direct_date > today:
        today_ww = _sailing_window_data(df, today, cfg)
        if today_ww:
            sc  = cfg.get("sailing", {})
            wmin, wmax = sc.get("wind_speed_min", 2.0), sc.get("wind_speed_max", 12.0)
            speeds = today_ww.get("speeds_kn", [])
            obs_frac = (sum(1 for s in speeds if wmin <= s <= wmax) / len(speeds)
                        if speeds else 0.0)
            today_entry: dict = {
                "snapshot":          result["snapshot"],
                "predicting_date":   str(today),
                "sailing_window":    result["sailing_window"],
                "probability":       round(obs_frac, 3),
                "good":              obs_frac >= cfg["prediction"]["min_good_fraction"],
                "threshold":         cfg["prediction"]["min_good_fraction"],
                "window_wind":       today_ww,
                "window_observed_only": True,   # not an ML prediction — exclude from DB
            }
            c_score, c_label, c_icon = _condition_rating(today_entry, cfg)
            today_entry["condition_score"]  = c_score
            today_entry["condition_label"]  = c_label
            today_entry["condition_icon"]   = c_icon
            today_entry["condition_source"] = "observed"
            results.insert(0, today_entry)

    # Extended predictions for remaining days up to today+3
    already_covered = {r["predicting_date"] for r in results if "predicting_date" in r}
    for lead in range(4):
        future_date = today + timedelta(days=lead)
        if str(future_date) in already_covered:
            continue  # already covered by direct prediction or observed-only entry
        decay = _FORECAST_DECAY.get(min(lead, 3), 0.50)
        decayed_prob = round(float(result["probability"]) * decay, 3)

        c_score = int(round(decayed_prob * 100))
        c_label, c_icon = "No wind", "🌫"
        for thr, lbl, icn in _CONDITION_LEVELS:
            if c_score >= thr:
                c_label, c_icon = lbl, icn
                break

        results.append({
            "snapshot":             result["snapshot"],
            "predicting_date":      str(future_date),
            "sailing_window":       result["sailing_window"],
            "probability":          decayed_prob,
            "good":                 decayed_prob >= cfg["prediction"]["min_good_fraction"],
            "threshold":            cfg["prediction"]["min_good_fraction"],
            "is_extended_forecast": True,
            "lead_days":            lead,
            "condition_score":      c_score,
            "condition_label":      c_label,
            "condition_icon":       c_icon,
            "condition_source":     "forecast",
        })

    _enrich_with_nwp(results, cfg)
    return results


def merge_predictions(
    new_results: list[dict],
    pred_path: str,
    days_to_keep: int = 7,
) -> list[dict]:
    """
    Merge new_results with existing predictions.json.

    Deduplicates by (snapshot, predicting_date), prunes entries older than
    days_to_keep days, saves, and returns the merged list.
    """
    existing: list[dict] = []
    if os.path.exists(pred_path):
        try:
            with open(pred_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    cutoff = (date.today() - timedelta(days=days_to_keep)).isoformat()
    by_key: dict[tuple, dict] = {}
    for p in existing:
        if p.get("predicting_date", "") >= cutoff:
            by_key[(p.get("snapshot", ""), p.get("predicting_date", ""))] = p
    for p in new_results:
        if "predicting_date" in p:
            by_key[(p.get("snapshot", ""), p["predicting_date"])] = p

    merged = sorted(
        by_key.values(),
        key=lambda x: (x.get("predicting_date", ""), x.get("snapshot", "")),
    )
    with open(pred_path, "w") as f:
        json.dump(merged, f, indent=2)
    return merged


if __name__ == "__main__":
    args = sys.argv[1:]
    config_path = DEFAULT_CONFIG
    cfg = load_config(config_path)
    root = os.path.dirname(os.path.abspath(config_path))

    df = pd.read_parquet(os.path.join(root, cfg["paths"]["data_parquet"]))

    # Optional: override snapshot time via CLI args (date HH:MM)
    snap_dt: Optional[pd.Timestamp] = None
    if len(args) == 2:
        d = datetime.strptime(args[0], "%Y-%m-%d").date()
        h, m = map(int, args[1].split(":"))
        snap_dt = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=h, minute=m)
    elif len(args) == 1:
        print("Usage: python model/predict.py [date HH:MM]", file=sys.stderr)
        sys.exit(1)

    results = predict_now(df, config_path, snap_dt=snap_dt)

    # Merge with existing predictions.json (rolling 7-day window)
    pred_path = os.path.join(root, cfg["paths"]["predictions_file"])
    merged = merge_predictions(results, pred_path)
    n_days = len({r["predicting_date"] for r in merged if "predicting_date" in r})
    print(f"Saved: {pred_path}  ({len(merged)} entries, {n_days} day(s))", flush=True)

    # ── Persist direct predictions to history DB ───────────────────────────────
    db_path = os.path.join(root, "predictions.db")
    try:
        from model.history import record_predictions, backfill_outcomes
        from model.features import compute_daily_target

        # Only record direct ML predictions (not extended forecasts or observed-only entries)
        direct = [r for r in results
                  if not r.get("is_extended_forecast") and not r.get("window_observed_only")]
        n_written = record_predictions(direct, db_path)
        print(f"History: wrote {n_written} row(s) to {db_path}", flush=True)

        daily_quality = compute_daily_target(df, cfg)
        n_outcomes = backfill_outcomes(daily_quality, db_path)
        print(f"History: upserted {n_outcomes} outcome(s)", flush=True)
    except Exception as exc:
        print(f"  [!] History recording skipped: {exc}", file=sys.stderr)
