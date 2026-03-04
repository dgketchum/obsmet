"""GDAS PrepBUFR archive download from GDEX (NCAR d337000).

Downloads daily PrepBUFR tar archives with manifest-based resume.
No authentication needed. Tries NR format first (2009+), falls back
to WO40 format (pre-2009).

Ported from GDASApp/scripts/download_gdas_prepbufr.py.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError

DEFAULT_RAW_DIR = "/nas/climate/gdas/prepbufr"

_BASE_URL = "https://osdf-director.osg-htc.org/ncar/gdex/d337000/tarfiles"
_SUFFIXES = [".nr.tar.gz", ".wo40.tar.gz"]

_ARCHIVE_START = date(1997, 4, 30)

# Files smaller than 50 MB are flagged as suspect
_SUSPECT_SIZE = 50 * 1024 * 1024


def download_day(
    day: date,
    dest_dir: Path | str,
    *,
    retries: int = 3,
) -> tuple[str, bool, str, str | None]:
    """Download a single day's PrepBUFR archive.

    Tries NR suffix first, falls back to WO40.

    Returns (date_str, success, message, local_path_or_None).
    """
    dest_dir = Path(dest_dir)
    date_str = day.strftime("%Y%m%d")
    year_str = day.strftime("%Y")
    year_dir = dest_dir / year_str
    year_dir.mkdir(parents=True, exist_ok=True)

    for suffix in _SUFFIXES:
        filename = f"prepbufr.{date_str}{suffix}"
        url = f"{_BASE_URL}/{year_str}/{filename}"
        local_path = year_dir / filename

        if local_path.exists() and local_path.stat().st_size > 0:
            return date_str, True, "exists", str(local_path)

        for attempt in range(retries):
            try:
                with urlopen(url, timeout=120) as resp:
                    data = resp.read()
                local_path.write_bytes(data)
                return date_str, True, "ok", str(local_path)
            except HTTPError as exc:
                if exc.code == 404:
                    break  # Try next suffix
                if attempt < retries - 1:
                    time.sleep(2**attempt)
            except Exception:
                if attempt < retries - 1:
                    time.sleep(2**attempt)

    return date_str, False, "not_found", None


def download_range(
    start: date,
    end: date,
    dest_dir: Path | str,
    *,
    workers: int = 10,
    done_dates: set[str] | None = None,
) -> list[tuple[str, bool, str, str | None]]:
    """Download PrepBUFR archives for a date range.

    Parameters
    ----------
    start, end : Date range (inclusive).
    dest_dir : Root destination directory.
    workers : Number of download threads.
    done_dates : Set of YYYYMMDD strings already downloaded (for resume).
    """
    dest_dir = Path(dest_dir)
    from datetime import timedelta

    days = []
    current = max(start, _ARCHIVE_START)
    while current <= end:
        ds = current.strftime("%Y%m%d")
        if done_dates is None or ds not in done_dates:
            days.append(current)
        current += timedelta(days=1)

    if not days:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_day, d, dest_dir): d for d in days}
        for fut in as_completed(futures):
            results.append(fut.result())

    return results
