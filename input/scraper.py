"""
scraper.py — Download daily Ecowitt weather xlsx files.

Usage:
    python scraper.py                          # last 7 days
    python scraper.py 2026-03-01              # single date
    python scraper.py 2026-02-01 2026-03-06   # date range

Credentials are read from (in priority order):
  1. Environment variables: ECOWITT_DEVICE_ID, ECOWITT_AUTHORIZE
  2. config.toml [ecowitt] section
  3. .env file (loaded automatically via utils/config.py)
"""

import os
import sys
import time
from datetime import date, datetime, timedelta

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from utils.config import load_config

_SORT_LIST = "0|1|2|49|4|5|32"   # station-specific sensor selection, not a credential

DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "downloaded_files")


def _credentials() -> tuple[str, str]:
    """Return (device_id, authorize) from env vars or config.toml."""
    cfg = load_config()
    ecowitt = cfg.get("ecowitt", {})
    device_id = os.environ.get("ECOWITT_DEVICE_ID") or ecowitt.get("device_id", "")
    authorize = os.environ.get("ECOWITT_AUTHORIZE") or ecowitt.get("authorize", "")
    if not device_id or not authorize:
        raise RuntimeError(
            "Ecowitt credentials not found. Set ECOWITT_DEVICE_ID and "
            "ECOWITT_AUTHORIZE in .env or config.toml [ecowitt]."
        )
    return device_id, authorize


def _make_headers(device_id: str, authorize: str) -> dict:
    referer = (
        f"https://www.ecowitt.net/home/share?authorize={authorize}"
        f"&device_id={device_id}&units=1,3,7,12,16,24"
    )
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
        ),
        "Referer": referer,
        "Content-Type": "application/x-www-form-urlencoded",
    }


def download_date(
    date_str: str,
    session: requests.Session,
    output_dir: str,
    edate: str | None = None,
) -> bool:
    """Download weather data for a single date (YYYY-MM-DD). Returns True on success."""
    edate = edate or "23:59"
    device_id, authorize = _credentials()
    headers = _make_headers(device_id, authorize)

    ts = int(time.time() * 1000)
    resp = session.post(
        f"https://www.ecowitt.net/index/export_excel?time={ts}",
        data={
            "device_id": device_id,
            "authorize": authorize,
            "mode": "0",
            "sdate": f"{date_str} 00:00",
            "edate": f"{date_str} {edate}",
            "sortList": _SORT_LIST,
            "hideList": "",
        },
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()

    try:
        data = resp.json()
        xlsx_url = data["url"]
    except Exception:
        print(f"  [!] {date_str}: export_excel response not valid JSON "
              f"(status={resp.status_code}, body={resp.text[:200]!r})")
        date_compact = date_str.replace("-", "")
        edate_compact = edate.replace(":", "")
        xlsx_url = (
            f"https://www.ecowitt.net/uploads/{device_id}/"
            f"Wetterstation%28{date_compact}0000-{date_compact}{edate_compact}%29.xlsx"
        )

    resp = session.get(xlsx_url, headers=headers, timeout=15)
    resp.raise_for_status()

    if resp.content[:2] != b"PK":
        print(f"  [!] {date_str}: not a valid xlsx ({len(resp.content)} bytes)")
        return False

    out_path = os.path.join(output_dir, f"Wetterstation_{date_str}.xlsx")
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"  [+] {date_str}: {len(resp.content):,} bytes → {out_path}")
    return True


def download_range(
    start: date,
    end: date,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    skip_existing: bool = True,
    force_dates: set | None = None,
    delay: float = 1.0,
) -> dict[str, bool]:
    """
    Download one xlsx per day for [start, end] inclusive.
    Skips dates that already have a file unless they appear in force_dates.
    Returns {date_str: success} for every date in the range.
    """
    os.makedirs(output_dir, exist_ok=True)
    force_dates = force_dates or set()
    results: dict[str, bool] = {}
    session = requests.Session()

    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        out_path = os.path.join(output_dir, f"Wetterstation_{date_str}.xlsx")
        forced = current in force_dates

        if skip_existing and not forced and os.path.exists(out_path):
            print(f"  [=] {date_str}: already exists, skipping")
            results[date_str] = True
            current += timedelta(days=1)
            continue

        try:
            edate = datetime.now().strftime("%H:%M") if forced else None
            results[date_str] = download_date(date_str, session, output_dir, edate=edate)
        except requests.exceptions.RequestException as e:
            print(f"  [!] {date_str}: {e}")
            results[date_str] = False

        time.sleep(delay)
        current += timedelta(days=1)

    return results


def last_week() -> tuple[date, date]:
    """Return (start, end) for the 7 complete days ending yesterday."""
    yesterday = date.today() - timedelta(days=1)
    return yesterday - timedelta(days=6), yesterday


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        start, end = last_week()
    elif len(args) == 1:
        start = end = datetime.strptime(args[0], "%Y-%m-%d").date()
    elif len(args) == 2:
        start = datetime.strptime(args[0], "%Y-%m-%d").date()
        end   = datetime.strptime(args[1], "%Y-%m-%d").date()
    else:
        print("Usage: python scraper.py [start_date [end_date]]")
        sys.exit(1)

    print(f"Downloading {start} → {end}  ({(end - start).days + 1} day(s))")
    results = download_range(start, end)
    n_ok = sum(results.values())
    print(f"\nDone: {n_ok}/{len(results)} downloaded successfully")
    if n_ok == 0 and len(results) > 0:
        sys.exit(1)
