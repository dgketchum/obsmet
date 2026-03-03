"""MADIS raw netCDF acquisition from NOAA archive.

Downloads gzip-compressed hourly netCDF files from the MADIS Research
LDAD mesonet archive.  Files are named YYYYMMDD_HHMM.gz (24 per day).

Source URL: https://madis-data.ncep.noaa.gov/madisResearch/data
Requires MADIS Research account credentials.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


BASE_URL = "https://madis-data.ncep.noaa.gov/madisResearch/data"

DEFAULT_RAW_DIR = "/nas/climate/obsmet/raw/madis"


def load_credentials(path: str | Path | None = None) -> tuple[str, str]:
    """Load MADIS username/password from a JSON file.

    Default location: ~/.config/obsmet/madis_credentials.json
    Expected format: {"usr": "...", "pswd": "..."}
    """
    if path is None:
        path = Path.home() / ".config" / "obsmet" / "madis_credentials.json"
    path = Path(path)
    with open(path) as f:
        creds = json.load(f)
    return creds["usr"], creds["pswd"]


def expected_files_for_day(day: datetime) -> list[str]:
    """Return the 24 expected filenames for a UTC day."""
    prefix = day.strftime("%Y%m%d")
    return [f"{prefix}_{h:02d}00.gz" for h in range(24)]


def download_day(
    day: datetime,
    dest_dir: Path | str,
    username: str,
    password: str,
    *,
    timeout: int = 600,
) -> tuple[str, bool, str]:
    """Download all 24 hourly files for a single day.

    Returns (day_str, success, message).
    Skips if all 24 files already exist.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    day_str = day.strftime("%Y%m%d")
    expected = expected_files_for_day(day)
    targets = [dest_dir / f for f in expected]

    if all(t.exists() for t in targets):
        return day_str, True, "exists"

    date_path = day.strftime("%Y/%m/%d")
    remote_dir = f"/archive/{date_path}/LDAD/mesonet/netCDF"
    url = f"{BASE_URL}{remote_dir}/"

    cmd = [
        "wget",
        "--user",
        username,
        "--password",
        password,
        "--no-check-certificate",
        "--no-directories",
        "--recursive",
        "--level=1",
        "--accept",
        "*.gz",
        "-q",
        f"--timeout={timeout}",
        url,
    ]

    try:
        subprocess.run(cmd, check=True, cwd=str(dest_dir), capture_output=True)
        return day_str, True, "downloaded"
    except subprocess.CalledProcessError as e:
        return day_str, False, f"wget failed: {e}"


def download_range(
    start: datetime,
    end: datetime,
    dest_dir: Path | str,
    username: str,
    password: str,
    *,
    rate_limit_interval: float = 0.5,
    rate_limit_every: int = 10,
) -> list[tuple[str, bool, str]]:
    """Download all days in [start, end] inclusive.  Sequential with rate limiting."""
    results = []
    current = start
    count = 0
    while current <= end:
        result = download_day(current, dest_dir, username, password)
        results.append(result)
        day_str, success, msg = result
        if msg != "exists":
            count += 1
            if count % rate_limit_every == 0:
                time.sleep(rate_limit_interval)
        current += timedelta(days=1)
    return results
