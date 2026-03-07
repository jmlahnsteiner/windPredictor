import os
import time
from datetime import datetime

import requests

DEVICE_ID = "cHF5MGpPMzZFeDEvbFAvL2J6QjBWdz09"
AUTHORIZE = "E6K45F"
SORT_LIST = "0|1|2|49|4|5|32"
REFERER = f"https://www.ecowitt.net/home/share?authorize={AUTHORIZE}&device_id={DEVICE_ID}&units=1,3,7,12,16,24"

# Set output directory
output_dir = "downloaded_files"
os.makedirs(output_dir, exist_ok=True)


def download_file(date_str):
    """Download weather data xlsx for a given date (YYYY-MM-DD)."""
    sdate = f"{date_str} 00:00"
    edate = f"{date_str} 23:59"
    date_compact = date_str.replace("-", "")
    xlsx_url = f"https://www.ecowitt.net/uploads/156707/Wetterstation%28{date_compact}0000-{date_compact}2359%29.xlsx"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        "Referer": REFERER,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    try:
        session = requests.Session()

        # Step 1: POST to trigger server-side file generation
        ts = int(time.time() * 1000)
        resp = session.post(
            f"https://www.ecowitt.net/index/export_excel?time={ts}",
            data={
                "device_id": DEVICE_ID,
                "authorize": AUTHORIZE,
                "mode": "0",
                "sdate": sdate,
                "edate": edate,
                "sortList": SORT_LIST,
                "hideList": "",
            },
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()

        # Step 2: GET the generated xlsx file
        resp = session.get(xlsx_url, headers=headers, timeout=15)
        resp.raise_for_status()

        if resp.content[:2] != b'PK':
            print(f"Error: Response is not a valid xlsx file ({len(resp.content)} bytes)")
            print(f"Content preview: {resp.content[:200]}")
            return

        file_path = os.path.join(output_dir, f"Wetterstation_{date_str}.xlsx")
        with open(file_path, "wb") as f:
            f.write(resp.content)
        print(f"Saved: {file_path}")

    except requests.exceptions.RequestException as e:
        print("Error downloading file:", e)
    except Exception as e:
        print("Unexpected error:", e)


# Example usage
date = "2026-03-04"
download_file(date)
