#!/usr/bin/env python3
"""
deploy.py — Full pipeline: download → stitch → predict → render → publish.

Downloads the latest weather data, runs all configured snapshot predictions,
renders index.html, and pushes it to the remote (GitHub Pages).

Usage:
    python deploy.py                      # last 2 days, predict for today
    python deploy.py --date 2026-03-08   # predict for a specific date
    python deploy.py --days 7            # download last 7 days of data
    python deploy.py --dry-run           # skip git push
    python deploy.py --no-download       # skip download, use existing xlsx files
"""

import argparse
import json
import os
import subprocess
import sys
import tomllib
from datetime import date, datetime, timedelta

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print(f"\n{'─' * 60}\n{msg}\n{'─' * 60}", flush=True)


def _load_config() -> dict:
    with open(os.path.join(_ROOT, "config.toml"), "rb") as f:
        return tomllib.load(f)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_download(days: int) -> None:
    _banner(f"[1/4] Downloading last {days} day(s) of weather data")
    from input.scraper import download_range

    end = date.today() - timedelta(days=1)      # yesterday (complete)
    start = end - timedelta(days=days - 1)
    print(f"Range: {start} → {end}", flush=True)

    results = download_range(start, end)
    n_ok = sum(results.values())
    print(f"\nDownloaded: {n_ok}/{len(results)} day(s)", flush=True)
    if n_ok == 0:
        raise RuntimeError("No files downloaded — check network / credentials.")


def step_stitch() -> None:
    _banner("[2/4] Stitching xlsx → parquet")
    from input.stitcher import stitch

    cfg = _load_config()
    stitch(
        input_dir=os.path.join(_ROOT, "input", "downloaded_files"),
        output_path=os.path.join(_ROOT, cfg["paths"]["data_parquet"]),
    )


def step_predict(ref_date: date | None) -> None:
    import pandas as pd
    from model.predict import predict_all
    from model.history import record_predictions, backfill_outcomes
    from model.features import compute_daily_target

    label = str(ref_date or date.today())
    _banner(f"[3/4] Running predictions  (ref date: {label})")

    cfg = _load_config()
    parquet = os.path.join(_ROOT, cfg["paths"]["data_parquet"])
    if not os.path.exists(parquet):
        raise FileNotFoundError(f"No parquet data at {parquet} — run step_stitch first.")

    df = pd.read_parquet(parquet)
    results = predict_all(df, ref_date, os.path.join(_ROOT, "config.toml"))

    out_path = os.path.join(_ROOT, cfg["paths"]["predictions_file"])
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_days = len({r["predicting_date"] for r in results if "predicting_date" in r})
    print(f"Saved: {out_path}  ({len(results)} snapshot(s), {n_days} day(s))", flush=True)

    # ── Persist to history DB ─────────────────────────────────────────────────
    db_path = os.path.join(_ROOT, "predictions.db")
    n_written = record_predictions(results, db_path)
    print(f"History: wrote {n_written} row(s) to {db_path}", flush=True)

    # Backfill ground-truth outcomes for any past days now in the parquet
    daily_quality = compute_daily_target(df, cfg)
    n_outcomes = backfill_outcomes(daily_quality, db_path)
    print(f"History: upserted {n_outcomes} outcome(s)", flush=True)


def step_render() -> None:
    _banner("[4/4] Rendering index.html")
    from render_html import build_html, load_config

    cfg = _load_config()
    pred_path = os.path.join(_ROOT, cfg["paths"]["predictions_file"])
    with open(pred_path) as f:
        predictions = json.load(f)

    html = build_html(
        predictions,
        load_config(os.path.join(_ROOT, "config.toml")),
        db_path=os.path.join(_ROOT, "predictions.db"),
    )

    out_path = os.path.join(_ROOT, "index.html")
    with open(out_path, "w") as f:
        f.write(html)

    n_days = len({p["predicting_date"] for p in predictions if "predicting_date" in p})
    print(f"Written: {out_path}  ({len(predictions)} predictions, {n_days} day(s))", flush=True)


def step_publish(dry_run: bool) -> None:
    _banner("Publishing — git add / commit / push")

    msg = f"forecast: update {datetime.now().strftime('%-d %b %Y %H:%M')}"

    # Stage index.html and the predictions history DB (if it exists)
    to_stage = ["index.html"]
    db_path = os.path.join(_ROOT, "predictions.db")
    if os.path.exists(db_path):
        to_stage.append("predictions.db")
    subprocess.run(["git", "add"] + to_stage, cwd=_ROOT, check=True)

    # Check whether there is actually something new to commit
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=_ROOT,
    )
    if diff.returncode == 0:
        print("index.html unchanged — nothing to commit.", flush=True)
        return

    subprocess.run(["git", "commit", "-m", msg], cwd=_ROOT, check=True)
    print(f"Committed: {msg}", flush=True)

    if dry_run:
        print("[dry-run] Skipping git push.", flush=True)
    else:
        subprocess.run(["git", "push"], cwd=_ROOT, check=True)
        print("Pushed to remote (GitHub Pages).", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wind-predictor full deploy pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date", metavar="YYYY-MM-DD",
        help="Reference date for predictions (default: today)",
    )
    parser.add_argument(
        "--days", type=int, default=2, metavar="N",
        help="Days of history to download (default: 2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip git push",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip scraping step (use existing xlsx files)",
    )
    parser.add_argument(
        "--no-stitch", action="store_true",
        help="Skip stitching step (use existing parquet)",
    )
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
        step_publish(args.dry_run)
        print("\nAll done.", flush=True)
    except Exception as exc:
        print(f"\nFailed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
