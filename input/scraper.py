"""
scraper.py — Download daily Ecowitt weather xlsx files.

Usage:
    python scraper.py                          # last 7 days
    python scraper.py 2026-03-01              # single date
    python scraper.py 2026-02-01 2026-03-06   # date range
"""

import os
import sys
import time
from datetime import date, datetime, timedelta

import requests

DEVICE_ID = "cHF5MGpPMzZFeDEvbFAvL2J6QjBWdz09"
AUTHORIZE = "E6K45F"
SORT_LIST = "0|1|2|49|4|5|32"
REFERER = (
    f"https://www.ecowitt.net/home/share?authorize={AUTHORIZE}"
    f"&device_id={DEVICE_ID}&units=1,3,7,12,16,24"
)
BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    ),
    "Referer": REFERER,
    "Content-Type": "application/x-www-form-urlencoded",
}

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "downloaded_files")


def download_date(date_str: str, session: requests.Session, output_dir: str) -> bool:
    """Download weather data for a single date (YYYY-MM-DD). Returns True on success."""
    date_compact = date_str.replace("-", "")
    xlsx_url = (
        f"https://www.ecowitt.net/uploads/156707/"
        f"Wetterstation%28{date_compact}0000-{date_compact}2359%29.xlsx"
    )

    # Step 1: POST to trigger server-side file generation
    ts = int(time.time() * 1000)
    resp = session.post(
        f"https://www.ecowitt.net/index/export_excel?time={ts}",
        data={
            "device_id": DEVICE_ID,
            "authorize": AUTHORIZE,
            "mode": "0",
            "sdate": f"{date_str} 00:00",
            "edate": f"{date_str} 23:59",
            "sortList": SORT_LIST,
            "hideList": "",
        },
        headers=BASE_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()

    # Step 2: GET the generated xlsx
    resp = session.get(xlsx_url, headers=BASE_HEADERS, timeout=15)
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
    delay: float = 1.0,
) -> dict[str, bool]:
    """
    Download one xlsx per day for [start, end] inclusive.
    Skips dates that already have a file on disk.
    Returns {date_str: success} for every date in the range.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: dict[str, bool] = {}
    session = requests.Session()

    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        out_path = os.path.join(output_dir, f"Wetterstation_{date_str}.xlsx")

        if skip_existing and os.path.exists(out_path):
            print(f"  [=] {date_str}: already exists, skipping")
            results[date_str] = True
            current += timedelta(days=1)
            continue

        try:
            results[date_str] = download_date(date_str, session, output_dir)
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
        end = datetime.strptime(args[1], "%Y-%m-%d").date()
    else:
        print("Usage: python scraper.py [start_date [end_date]]")
        sys.exit(1)

    print(f"Downloading {start} → {end}  ({(end - start).days + 1} day(s))")
    results = download_range(start, end)
    n_ok = sum(results.values())
    print(f"\nDone: {n_ok}/{len(results)} downloaded successfully")
