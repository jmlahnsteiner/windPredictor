#!/usr/bin/env python3
"""
deploy.py — Local pipeline: download → stitch → predict → render.

Generates index.html for local preview. Does NOT push to git.
CI (GitHub Actions) handles deployment via artifact-based Pages.

Usage:
    python deploy.py                      # last 2 days, predict for today
    python deploy.py --date 2026-03-08   # predict for a specific date
    python deploy.py --days 7            # download last 7 days of data
    python deploy.py --no-download       # skip download, use existing xlsx files
    python deploy.py --preview           # open index.html in browser after render
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def _banner(msg: str) -> None:
    print(f"\n{'─' * 60}\n{msg}\n{'─' * 60}", flush=True)


def _load_config() -> dict:
    from utils.config import load_config
    return load_config(os.path.join(_ROOT, "config.toml"))


def step_download(days: int) -> None:
    _banner(f"[1/4] Downloading last {days} day(s) of weather data")
    from input.scraper import download_range

    today = date.today()
    end   = today
    start = end - timedelta(days=days - 1)
    print(f"Range: {start} → {end}  (today re-fetched for latest readings)", flush=True)

    results = download_range(start, end, force_dates={today})
    n_ok = sum(results.values())
    print(f"\nDownloaded: {n_ok}/{len(results)} day(s)", flush=True)
    if n_ok == 0:
        raise RuntimeError("No files downloaded — check network / credentials.")


def step_stitch() -> None:
    _banner("[2/4] Stitching xlsx → database")
    from input.stitcher import stitch_to_db
    stitch_to_db(input_dir=os.path.join(_ROOT, "input", "downloaded_files"))


def step_predict(ref_date: date | None) -> None:
    label = str(ref_date or date.today())
    _banner(f"[3/4] Running predictions  (ref date: {label})")

    import pandas as pd
    from input.weather_store import load_weather_readings
    from model.predict import predict_now, save_forecast_snapshots

    df = load_weather_readings(
        start=date.today() - timedelta(days=35),
        end=date.today(),
    )
    if df.empty:
        raise RuntimeError("No weather data found. Run step_stitch first.")

    config_path = os.path.join(_ROOT, "config.toml")

    snap_dt = None
    if ref_date is not None:
        now = datetime.now()
        snap_dt = pd.Timestamp(
            year=ref_date.year, month=ref_date.month, day=ref_date.day,
            hour=now.hour, minute=0,
        )

    results = predict_now(df, config_path, snap_dt=snap_dt)

    # Save forecast snapshots to DB (replaces predictions.json)
    save_forecast_snapshots(results)
    print(f"Forecast snapshots saved to DB  ({len(results)} entries)", flush=True)

    # History recording
    from model.history import record_predictions, backfill_outcomes
    from model.features import compute_daily_target
    from utils.config import load_config

    cfg = load_config(config_path)
    direct = [r for r in results
              if not r.get("is_extended_forecast") and not r.get("window_observed_only")]
    n_written = record_predictions(direct)
    print(f"History: wrote {n_written} row(s)", flush=True)

    daily_quality = compute_daily_target(df, cfg)
    n_outcomes = backfill_outcomes(daily_quality)
    print(f"History: upserted {n_outcomes} outcome(s)", flush=True)


def step_render() -> None:
    _banner("[4/4] Rendering index.html")
    from render_html import build_html
    from model.predict import load_forecast_snapshots
    from utils.config import load_config

    predictions = load_forecast_snapshots()
    cfg = load_config(os.path.join(_ROOT, "config.toml"))

    html = build_html(predictions, cfg)

    out_path = os.path.join(_ROOT, "index.html")
    with open(out_path, "w") as f:
        f.write(html)

    n_days = len({p["predicting_date"] for p in predictions if "predicting_date" in p})
    print(f"Written: {out_path}  ({len(predictions)} predictions, {n_days} day(s))", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wind-predictor local pipeline (no git push)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--date",        metavar="YYYY-MM-DD")
    parser.add_argument("--days",        type=int, default=2)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-stitch",   action="store_true")
    parser.add_argument("--preview",     action="store_true", help="Open index.html in browser")
    args = parser.parse_args()

    ref_date: date | None = None
    if args.date:
        ref_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    try:
        if not args.no_download:
            step_download(args.days)
        if not args.no_stitch:
            step_stitch()
        step_predict(ref_date)
        step_render()
        print("\nAll done.", flush=True)

        if args.preview:
            import webbrowser
            webbrowser.open(os.path.join(_ROOT, "index.html"))

    except Exception as exc:
        print(f"\nFailed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
