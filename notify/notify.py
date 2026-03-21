"""notify/notify.py — Good-day email notification and pipeline error alerts."""
from __future__ import annotations

import argparse
import os
import sys
import tomllib
from datetime import date, datetime
from pathlib import Path

# Ensure project root is on sys.path so `model` package is importable
# when this script is run directly (python notify/notify.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import resend  # top-level import — makes monkeypatching reliable in tests
from model.predict import load_forecast_snapshots

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_today_entry(predictions: list[dict], today: str) -> dict | None:
    """Return the latest-snapshot prediction entry for *today*, or None."""
    todays = [e for e in predictions if e.get("predicting_date") == today]
    if not todays:
        return None
    return max(todays, key=lambda e: e.get("snapshot", ""))


def build_body(entry: dict, window_start: str, window_end: str) -> str:
    """Build the plain-text email body for a good-day notification."""
    pct = round(float(entry.get("probability", 0)) * 100)
    label = entry.get("condition_label") or "Good"

    nwp = entry.get("nwp_forecast") or {}
    mean_wind = nwp.get("mean_wind_kn")
    max_gust = nwp.get("max_gust_kn")
    wind_line = (
        f"Expected wind: avg {mean_wind} kn · gust {max_gust} kn\n"
        if mean_wind is not None and max_gust is not None
        else ""
    )

    return (
        f"Good sailing conditions forecast for today's window "
        f"({window_start}–{window_end}).\n"
        f"\n"
        f"Probability:  {pct}%  ({label})\n"
        f"{wind_line}"
        f"Sailing window: {window_start}–{window_end}\n"
        f"\n"
        f"Full forecast: https://jmlahnsteiner.github.io/windPredictor/\n"
    )


def _format_subject_date(today: str) -> str:
    """Format today's date as 'Monday, 15 March 2026' (no leading zero, cross-platform)."""
    dt = datetime.strptime(today, "%Y-%m-%d")
    return f"{dt.strftime('%A')}, {dt.day} {dt.strftime('%B %Y')}"


def _from_address() -> str:
    """Return the from address, preferring NOTIFY_FROM_EMAIL over the Resend sandbox default."""
    return os.environ.get("NOTIFY_FROM_EMAIL", "WindPredictor <onboarding@resend.dev>")


def send_error_email(message: str) -> None:
    """Send a pipeline error alert email."""
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        print("ERROR: RESEND_API_KEY environment variable is not set.")
        sys.exit(1)

    notify_email = os.environ.get("NOTIFY_EMAIL")
    if not notify_email:
        print("ERROR: NOTIFY_EMAIL environment variable is not set.")
        sys.exit(1)

    today = date.today().isoformat()
    subject = f"⚠️ WindPredictor pipeline error — {_format_subject_date(today)}"
    body = f"The WindPredictor forecast pipeline encountered an error:\n\n{message}\n"

    resend.api_key = api_key
    try:
        resend.Emails.send({
            "from": _from_address(),
            "to": [notify_email],
            "subject": subject,
            "text": body,
        })
    except Exception as exc:
        print(f"ERROR: Resend API call failed: {exc}")
        sys.exit(1)

    print(f"Error notification sent to {notify_email}: {subject}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--error", metavar="MESSAGE",
                        help="Send a pipeline error alert instead of a good-day notification")
    args = parser.parse_args()

    if args.error:
        send_error_email(args.error)
        return

    # 1-2. Fail fast: check required env vars before any file I/O
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        print("ERROR: RESEND_API_KEY environment variable is not set.")
        sys.exit(1)

    notify_email = os.environ.get("NOTIFY_EMAIL")
    if not notify_email:
        print("ERROR: NOTIFY_EMAIL environment variable is not set.")
        sys.exit(1)

    from_email = os.environ.get("NOTIFY_FROM_EMAIL", "WindPredictor <onboarding@resend.dev>")

    # 3. Load config
    _root = Path(__file__).resolve().parent.parent
    with open(_root / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    sailing = cfg.get("sailing", {})
    window_start = sailing.get("window_start", "08:00")
    window_end = sailing.get("window_end", "16:00")

    # 4. Load forecast snapshots from DB
    predictions = load_forecast_snapshots()

    # 5-6. Find today's latest entry
    today = date.today().isoformat()
    entry = load_today_entry(predictions, today)
    if entry is None:
        sys.exit(0)

    # 7. Check if good
    if not entry.get("good"):
        sys.exit(0)

    # 8-9. Build email
    subject = f"⛵ Good sailing today — {_format_subject_date(today)}"
    body = build_body(entry, window_start, window_end)

    # 10. Send via Resend
    resend.api_key = api_key
    try:
        resend.Emails.send({
            "from": from_email,
            "to": [notify_email],
            "subject": subject,
            "text": body,
        })
    except Exception as exc:
        print(f"ERROR: Resend API call failed: {exc}")
        sys.exit(1)

    # 11. Confirm
    print(f"Notification sent to {notify_email}: {subject}")
    sys.exit(0)


if __name__ == "__main__":
    main()
