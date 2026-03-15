"""notify/notify.py — Good-day email notification."""
from __future__ import annotations

import os

import resend  # top-level import — makes monkeypatching reliable in tests

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
