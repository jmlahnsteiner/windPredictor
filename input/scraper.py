import os
from datetime import datetime, timedelta

import requests

# Set output directory
output_dir = "downloaded_files"
os.makedirs(output_dir, exist_ok=True)


def generate_file_url(date_str):
    start_date = datetime.strptime(date_str, "%Y-%m-%d")
    end_date = start_date + timedelta(days=1) - timedelta(minutes=1)
    start_time = start_date.strftime("%Y%m%d%H%M%S")
    end_time = end_date.strftime("%Y%m%d%H%M%S")
    return f"https://www.ecowitt.net/uploads/156707/Wetterstation%28{start_time}-{end_time}%29.xlsx"


def download_file(url, date_str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        "Referer": "https://www.ecowitt.net/home/share?authorize=E6K45F&device_id=cHF5MGpPMzZFeDEvbFAvL2J6QjBWdz09&units=1,3,7,12,16,24",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise error for 4xx/5xx
        # Validate xlsx content (xlsx files start with PK magic bytes)
        if not response.content[:2] == b'PK':
            print(f"Error: Response is not a valid xlsx file (got {len(response.content)} bytes)")
            print(f"Content preview: {response.content[:200]}")
            return
        file_path = os.path.join(output_dir, f"Wetterstation_{date_str}.xlsx")
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Saved: {file_path}")
    except requests.exceptions.RequestException as e:
        print("Error downloading file:", e)
    except Exception as e:
        print("Unexpected error:", e)


# Example usage
date = "2026-03-04"
url = generate_file_url(date)
download_file(url, date)
