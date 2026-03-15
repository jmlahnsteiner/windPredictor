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
