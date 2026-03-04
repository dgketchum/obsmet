"""RAWS WRCC station inventory scraper and daily data downloader.

Scrapes station inventory from WRCC region pages, downloads daily
meteorological data via HTTP POST to WRCC CGI endpoint.

Ported from dads-mvp/extract/met_data/obs/raws_wrcc.py.
"""

from __future__ import annotations

import re
import time
from datetime import date
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_RAW_DIR = "/nas/climate/obsmet/raw/raws_wrcc"

_BASE_URL = "https://raws.dri.edu"
_DATA_URL = "https://wrcc.dri.edu/cgi-bin/wea_dysimts2.pl"

# Max calendar years per request (WRCC silently returns empty for > ~5)
_MAX_CHUNK_YEARS = 4

# Rate limiting delay between requests (seconds)
_REQUEST_DELAY = 2.0

# 25 WRCC region pages
REGION_PAGES = [
    "wa",
    "or",
    "sid",
    "nidwmt",
    "emt",
    "wy",
    "nca",
    "cca",
    "sca",
    "nvut",
    "co",
    "az",
    "nm",
    "nd",
    "sd",
    "ne",
    "ks",
    "ok",
    "tx",
    "mn",
    "ia",
    "mo",
    "ar",
    "la",
    "ak",
]

# Daily CSV column names (15 columns)
DAILY_COLUMNS = [
    "date",
    "year",
    "doy",
    "day_of_run",
    "srad_total_kwh_m2",
    "wspd_ave_ms",
    "wdir_vec_deg",
    "wspd_gust_ms",
    "tair_ave_c",
    "tair_max_c",
    "tair_min_c",
    "rh_ave_pct",
    "rh_max_pct",
    "rh_min_pct",
    "prcp_total_mm",
]


# --------------------------------------------------------------------------- #
# Station inventory scraping
# --------------------------------------------------------------------------- #


def _parse_dms(dms_str: str) -> float | None:
    """Parse degrees-minutes-seconds string to decimal degrees."""
    m = re.search(r"(\d+)\s*\u00b0\s*(\d+)\s*[']\s*(\d+)\s*[\"]*", dms_str)
    if not m:
        # Try simpler pattern with deg/min/sec separated by spaces
        m = re.search(r"(\d+)\s+(\d+)\s+(\d+)", dms_str)
    if not m:
        try:
            return float(dms_str)
        except ValueError:
            return None

    deg, mn, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return deg + mn / 60.0 + sec / 3600.0


def scrape_region_stations(region: str) -> list[dict]:
    """Scrape station list from a WRCC region page.

    Returns list of dicts with keys: wrcc_id, name, region_page.
    """
    url = f"{_BASE_URL}/wraws/{region}lst.html"
    try:
        with urlopen(url, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    stations = []
    # Extract station IDs from rawMAIN links
    ids = re.findall(r"rawMAIN\.pl\?(\w+)", html)
    # Extract station names from JavaScript update() calls
    names = re.findall(r"update\('([^']+?)\s*\(RAWS\)'\)", html)

    for i, wrcc_id in enumerate(ids):
        name = names[i] if i < len(names) else wrcc_id
        stations.append({"wrcc_id": wrcc_id, "name": name, "region_page": region})

    return stations


def scrape_station_metadata(wrcc_id: str) -> dict:
    """Scrape metadata for a single station from WRCC info page.

    Returns dict with keys: latitude, longitude, elevation_m, ness_id,
    nws_id, agency, location.
    """
    url = f"{_BASE_URL}/cgi-bin/wea_info.pl?{wrcc_id}"
    try:
        with urlopen(url, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return {}

    meta = {}

    loc_m = re.search(r"<b>Location\s*</b>\s*</td>\s*<td>\s*(.+?)\s*\n", html, re.I)
    if loc_m:
        meta["location"] = loc_m.group(1).strip()

    lat_m = re.search(r"<b>Latitude\s*</b>\s*</td>\s*<td>\s*(.+?)\s*\n", html, re.I)
    if lat_m:
        meta["latitude"] = _parse_dms(lat_m.group(1).strip())

    lon_m = re.search(r"<b>Longitude\s*</b>\s*</td>\s*<td>\s*(.+?)\s*\n", html, re.I)
    if lon_m:
        val = _parse_dms(lon_m.group(1).strip())
        if val is not None:
            meta["longitude"] = -abs(val)  # Western hemisphere

    elev_m = re.search(r"<b>Elevation\s*</b>\s*</td>\s*<td>\s*(\d+)\s*ft", html, re.I)
    if elev_m:
        meta["elevation_m"] = int(elev_m.group(1)) * 0.3048

    ness_m = re.search(r"<b>NESS ID\s*</b>\s*</td>\s*<td>\s*(\w+)", html, re.I)
    if ness_m:
        meta["ness_id"] = ness_m.group(1)

    nws_m = re.search(r"<b>NWS ID\s*</b>\s*</td>\s*<td>\s*(\w+)", html, re.I)
    if nws_m:
        meta["nws_id"] = nws_m.group(1)

    agency_m = re.search(r"<b>Agency\s*</b>\s*</td>\s*<td>\s*(.+?)\s*\n", html, re.I)
    if agency_m:
        meta["agency"] = agency_m.group(1).strip()

    return meta


# --------------------------------------------------------------------------- #
# Data download
# --------------------------------------------------------------------------- #


def download_station_data(
    wrcc_id: str,
    start: date,
    end: date,
    *,
    delay: float = _REQUEST_DELAY,
) -> str:
    """Download daily data for a station from WRCC.

    Chunks into _MAX_CHUNK_YEARS windows to avoid silent truncation.
    Returns concatenated response text.
    """
    # WRCC form uses station code without 2-char region prefix
    stn_code = wrcc_id[2:]

    all_text = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end_year = chunk_start.year + _MAX_CHUNK_YEARS
        chunk_end = min(date(chunk_end_year, 12, 31), end)

        payload = {
            "stn": stn_code,
            "smon": f"{chunk_start.month:02d}",
            "sday": f"{chunk_start.day:02d}",
            "syea": f"{chunk_start.year % 100:02d}",
            "emon": f"{chunk_end.month:02d}",
            "eday": f"{chunk_end.day:02d}",
            "eyea": f"{chunk_end.year % 100:02d}",
            "qBasic": "ON",
            "unit": "M",
            "Ofor": "A",
            "Datareq": "A",
            "qc": "Y",
            "miss": "08",
            "obs": "N",
            "WsMon": "01",
            "WsDay": "01",
            "WeMon": "12",
            "WeDay": "31",
            "Submit Info": "Submit Info",
        }

        data = urlencode(payload).encode("utf-8")
        req = Request(_DATA_URL, data=data, method="POST")

        try:
            with urlopen(req, timeout=60) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception:
            html = ""

        all_text.append(html)

        if delay > 0:
            time.sleep(delay)

        chunk_start = date(chunk_end_year + 1, 1, 1)

    return "\n".join(all_text)
