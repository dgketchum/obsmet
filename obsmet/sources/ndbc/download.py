"""NDBC station metadata and historical stdmet data download.

Downloads station metadata and historical standard meteorological
observations from NOAA's National Data Buoy Center public HTTPS.

Ported from dads-mvp/extract/met_data/obs/ndbc_download.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd

DEFAULT_RAW_DIR = "/nas/climate/obsmet/raw/ndbc"

_STATION_TABLE_URL = "https://www.ndbc.noaa.gov/data/stations/station_table.txt"
_HISTORICAL_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet"
_LATEST_URL = "https://www.ndbc.noaa.gov/data/l_stdmet"

# Rate limiting delay between requests (seconds)
_REQUEST_DELAY = 1.0

# Standard column order for NDBC stdmet files
STDMET_COLUMNS = [
    "YY",
    "MM",
    "DD",
    "hh",
    "mm",
    "WDIR",
    "WSPD",
    "GST",
    "WVHT",
    "DPD",
    "APD",
    "MWD",
    "PRES",
    "ATMP",
    "WTMP",
    "DEWP",
    "VIS",
    "TIDE",
]


# --------------------------------------------------------------------------- #
# Station metadata
# --------------------------------------------------------------------------- #


def _parse_location(loc_str: str) -> tuple[float | None, float | None]:
    """Parse NDBC location string to (latitude, longitude)."""
    # Try decimal degrees with N/S/E/W
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(N|S)\s*([+-]?\d+(?:\.\d+)?)\s*(W|E)", loc_str)
    if m:
        lat = float(m.group(1))
        if m.group(2) == "S":
            lat = -lat
        lon = float(m.group(3))
        if m.group(4) == "W":
            lon = -lon
        return lat, lon

    # DMS fallback
    m = re.search(
        r"(\d+)[^\d]+(\d+)[^\d]+(\d+).*?([NS]).*?(\d+)[^\d]+(\d+)[^\d]+(\d+).*?([WE])", loc_str
    )
    if m:
        lat = int(m.group(1)) + int(m.group(2)) / 60 + int(m.group(3)) / 3600
        if m.group(4) == "S":
            lat = -lat
        lon = int(m.group(5)) + int(m.group(6)) / 60 + int(m.group(7)) / 3600
        if m.group(8) == "W":
            lon = -lon
        return lat, lon

    return None, None


def get_ndbc_stations() -> pd.DataFrame:
    """Download and parse NDBC station metadata table.

    Returns DataFrame with columns: station_id, latitude, longitude,
    plus other metadata from the station table.
    """
    with urlopen(_STATION_TABLE_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    # Find header line (starts with # and contains |)
    lines = text.strip().split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("#") and "|" in line:
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame()

    header = lines[header_idx].lstrip("#").strip()
    cols = [c.strip() for c in header.split("|")]

    rows = []
    for line in lines[header_idx + 1 :]:
        if line.startswith("#") or not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= len(cols):
            rows.append(parts[: len(cols)])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)

    # Rename for consistency
    rename = {}
    for c in df.columns:
        if "STATION" in c.upper():
            rename[c] = "station_id"
        elif "LOCATION" in c.upper():
            rename[c] = "location"
    df = df.rename(columns=rename)

    # Force station IDs to uppercase
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].str.upper()

    # Parse coordinates from location field
    if "location" in df.columns:
        coords = df["location"].apply(_parse_location)
        df["latitude"] = coords.apply(lambda x: x[0])
        df["longitude"] = coords.apply(lambda x: x[1])
        df = df.dropna(subset=["latitude", "longitude"])

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Historical data download
# --------------------------------------------------------------------------- #


def download_station_year(
    station_id: str,
    year: int,
    dest_dir: Path | str,
) -> tuple[str, int, bool, str]:
    """Download a single station-year stdmet file.

    Returns (station_id, year, success, message).
    """
    dest_dir = Path(dest_dir)
    sid = station_id.lower()
    filename = f"{sid}h{year}.txt.gz"
    url = f"{_HISTORICAL_URL}/{filename}"
    local_path = dest_dir / filename

    if local_path.exists() and local_path.stat().st_size > 0:
        return station_id, year, True, "exists"

    try:
        with urlopen(url, timeout=30) as resp:
            data = resp.read()
        local_path.write_bytes(data)
        return station_id, year, True, "ok"
    except HTTPError as exc:
        if exc.code == 404:
            return station_id, year, False, "not_found"
        return station_id, year, False, str(exc)
    except Exception as exc:
        return station_id, year, False, str(exc)


def download_station_latest(
    station_id: str,
    dest_dir: Path | str,
) -> tuple[str, bool, str]:
    """Download latest (current year) stdmet file for a station.

    Returns (station_id, success, message).
    """
    dest_dir = Path(dest_dir)
    sid = station_id.lower()
    filename = f"{sid}_latest.txt"
    url = f"{_LATEST_URL}/{sid}.txt"
    local_path = dest_dir / filename

    try:
        with urlopen(url, timeout=30) as resp:
            data = resp.read()
        local_path.write_bytes(data)
        return station_id, True, "ok"
    except HTTPError as exc:
        if exc.code == 404:
            return station_id, False, "not_found"
        return station_id, False, str(exc)
    except Exception as exc:
        return station_id, False, str(exc)
