"""SNOTEL hourly data downloader via NRCS AWDB REST API.

Downloads hourly observations for SNTL network stations. Each station is saved
as a single parquet file with all requested elements in wide form.

The AWDB REST API provides structured JSON with element metadata, unit codes,
and hourly values. No authentication required.

Station triplet format: {stationId}:{stateCode}:SNTL (e.g., 562:MT:SNTL)

Timezone convention
-------------------
ALL SNOTEL timestamps are in **Pacific Standard Time (PST = UTC-8)** year-round,
regardless of the station's physical location. Alaska stations use AKST (UTC-9).
There is no DST adjustment — PST/AKST is used even in summer months.

The AWDB ``dataTimeZone`` field confirms this: -8.0 for all lower-48 states,
-9.0 for Alaska. This is a longstanding NRCS convention (NWCC HQ is in Portland, OR).

Consequence: a Colorado station reporting ``2024-01-15 00:00`` means midnight PST,
which is 08:00 UTC and 01:00 MST. The ``_convert_to_utc`` function uses the
``dataTimeZone`` offset directly, which is correct because it represents the
**data** timezone (PST/AKST), not the station's physical timezone.

Note: AWDB station IDs differ from the IDs used in the legacy daily CSV files
(snotel_records/). The station inventory maps AWDB IDs to names/coordinates.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

AWDB_BASE = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"

# Core elements for hourly download.
# WTEQ = snow water equivalent (in), SNWD = snow depth (in),
# PREC = accumulated precipitation (in), TOBS = observed air temp (degF)
DEFAULT_ELEMENTS = ["WTEQ", "SNWD", "PREC", "TOBS"]

# Unit conversions: AWDB returns imperial by default
_UNIT_CONVERT = {
    "in_to_mm": 25.4,
    "degF_to_degC": lambda f: (f - 32.0) * 5.0 / 9.0,
}

# Physical timezone by state (IANA names) for downstream local-day aggregation.
# SNOTEL data timestamps use PST/AKST, but the station's physical timezone
# matters for understanding when "midnight" actually occurs at the site.
_STATE_TIMEZONE = {
    "AK": "US/Alaska",
    "AZ": "US/Arizona",  # no DST
    "CA": "US/Pacific",
    "CO": "US/Mountain",
    "ID": "US/Mountain",
    "MT": "US/Mountain",
    "NM": "US/Mountain",
    "NV": "US/Pacific",
    "OR": "US/Pacific",
    "SD": "US/Mountain",
    "UT": "US/Mountain",
    "WA": "US/Pacific",
    "WY": "US/Mountain",
}


def fetch_station_inventory(
    *,
    network: str = "SNTL",
    active_only: bool = True,
) -> pd.DataFrame:
    """Fetch SNOTEL station metadata from AWDB REST API.

    Returns DataFrame with columns:
        station_triplet, station_id, state, name, lat, lon, elev_ft,
        tz_offset, physical_tz, begin_date, end_date

    tz_offset is the AWDB dataTimeZone (PST=-8 or AKST=-9, the DATA timezone).
    physical_tz is the IANA timezone for the station's physical location.
    """
    url = f"{AWDB_BASE}/stations"
    params = {
        "activeOnly": str(active_only).lower(),
        "networkCds": network,
        "returnFields": (
            "stationTriplet,name,latitude,longitude,elevation,stationDataTimeZone,beginDate,endDate"
        ),
        "logicalAnd": "true",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    raw = resp.json()

    # Filter to exact network (API sometimes returns other networks)
    records = []
    for s in raw:
        if s.get("networkCode") != network:
            continue
        state = s.get("stateCode", "")
        records.append(
            {
                "station_triplet": s.get("stationTriplet"),
                "station_id": s.get("stationId"),
                "state": state,
                "name": s.get("name"),
                "lat": s.get("latitude"),
                "lon": s.get("longitude"),
                "elev_ft": s.get("elevation"),
                "tz_offset": s.get("dataTimeZone"),
                "physical_tz": _STATE_TIMEZONE.get(state, ""),
                "begin_date": s.get("beginDate"),
                "end_date": s.get("endDate"),
            }
        )

    df = pd.DataFrame(records)
    logger.info("Fetched %d %s stations from AWDB", len(df), network)
    return df


def _fetch_hourly_chunk(
    station_triplet: str,
    begin_date: str,
    end_date: str,
    elements: list[str],
) -> pd.DataFrame:
    """Fetch a single time chunk of hourly data from the AWDB REST API."""
    url = f"{AWDB_BASE}/data"
    params = {
        "stationTriplets": station_triplet,
        "elements": ",".join(elements),
        "duration": "HOURLY",
        "beginDate": begin_date,
        "endDate": end_date,
        "periodRef": "END",
    }

    resp = requests.get(url, params=params, timeout=300)
    resp.raise_for_status()
    payload = resp.json()

    if not payload or not payload[0].get("data"):
        return pd.DataFrame()

    station_data = payload[0]["data"]

    # Build one series per element, keyed by timestamp
    element_series = {}
    element_units = {}
    for elem_block in station_data:
        code = elem_block["stationElement"]["elementCode"]
        unit = elem_block["stationElement"]["storedUnitCode"]
        element_units[code] = unit
        vals = {v["date"]: v.get("value") for v in elem_block["values"]}
        element_series[code] = vals

    if not element_series:
        return pd.DataFrame()

    # Merge all elements on timestamp
    all_timestamps = sorted(set().union(*(s.keys() for s in element_series.values())))
    rows = []
    for ts in all_timestamps:
        row = {"datetime_local": ts}
        for code, vals in element_series.items():
            row[code] = vals.get(ts)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["datetime_local"] = pd.to_datetime(df["datetime_local"])

    # Convert units to metric
    for code, unit in element_units.items():
        if code not in df.columns:
            continue
        if unit == "in":
            df[code] = pd.to_numeric(df[code], errors="coerce") * _UNIT_CONVERT["in_to_mm"]
        elif unit == "degF":
            df[code] = pd.to_numeric(df[code], errors="coerce").apply(_UNIT_CONVERT["degF_to_degC"])

    return df


# Maximum years per API request to avoid timeouts (~46s per 5 years)
_CHUNK_YEARS = 5


def fetch_hourly_data(
    station_triplet: str,
    begin_date: str,
    end_date: str,
    elements: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch hourly data for a single station from AWDB REST API.

    Automatically chunks long date ranges into multi-year windows to avoid
    API timeouts. A 36-year request becomes ~8 sequential API calls.

    Parameters
    ----------
    station_triplet : e.g., "562:MT:SNTL"
    begin_date, end_date : "YYYY-MM-DD"
    elements : list of element codes (default: WTEQ, SNWD, PREC, TOBS)

    Returns
    -------
    DataFrame with datetime_local and one column per element, values
    converted to metric (mm, degC).
    """
    if elements is None:
        elements = DEFAULT_ELEMENTS

    start = pd.Timestamp(begin_date)
    end = pd.Timestamp(end_date)

    # Build year-chunked date ranges
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.DateOffset(years=_CHUNK_YEARS) - pd.Timedelta(days=1), end)
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + pd.Timedelta(days=1)

    frames = []
    for c_start, c_end in chunks:
        df = _fetch_hourly_chunk(station_triplet, c_start, c_end, elements)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _convert_to_utc(
    df: pd.DataFrame,
    tz_offset: float,
) -> pd.DataFrame:
    """Convert datetime_local to datetime_utc using station timezone offset.

    Keeps datetime_local and raw_tz_offset for audit.
    """
    if df.empty or "datetime_local" not in df.columns:
        return df

    df = df.copy()
    offset = pd.Timedelta(hours=tz_offset)
    df["datetime_utc"] = df["datetime_local"] - offset
    df["datetime_utc"] = df["datetime_utc"].dt.tz_localize("UTC")
    df["raw_tz_offset"] = tz_offset
    return df


def download_snotel_hourly(
    out_dir: str | Path,
    begin_date: str,
    end_date: str,
    *,
    elements: list[str] | None = None,
    states: list[str] | None = None,
    station_triplets: list[str] | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict[str, int]:
    """Download hourly SNOTEL data for all (or selected) stations.

    Writes one parquet per station to out_dir with columns:
        datetime_utc, datetime_local, raw_tz_offset,
        WTEQ (mm), SNWD (mm), PREC (mm, accumulated), TOBS (degC)

    Parameters
    ----------
    out_dir : Output directory for per-station parquets
    begin_date, end_date : "YYYY-MM-DD"
    elements : Element codes to download (default: WTEQ, SNWD, PREC, TOBS)
    states : Filter to specific states (e.g., ["MT", "ID"])
    station_triplets : Explicit list of triplets (overrides states filter)
    max_retries : Retry count for transient API failures
    retry_delay : Seconds between retries

    Returns
    -------
    Dict mapping station_triplet to row count downloaded.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if elements is None:
        elements = DEFAULT_ELEMENTS

    # Build station list — include inactive stations for full POR coverage
    if station_triplets is not None:
        # Use provided triplets; still need metadata for timezone
        inventory = fetch_station_inventory(active_only=False)
        inv_lookup = {row["station_triplet"]: row for _, row in inventory.iterrows()}
        triplets = station_triplets
    else:
        inventory = fetch_station_inventory(active_only=False)
        if states:
            inventory = inventory[inventory["state"].isin(states)]
        inv_lookup = {row["station_triplet"]: row for _, row in inventory.iterrows()}
        triplets = inventory["station_triplet"].tolist()

    # Save inventory alongside data
    inventory.to_parquet(out_dir / "station_inventory.parquet", index=False)
    logger.info(
        "Downloading %d stations, %s to %s, elements=%s",
        len(triplets),
        begin_date,
        end_date,
        elements,
    )

    stats = {}
    for i, triplet in enumerate(triplets):
        meta = inv_lookup.get(triplet)
        if meta is None:
            logger.warning("No metadata for %s, skipping", triplet)
            continue

        tz_offset = meta.get("tz_offset") or meta.get("raw_tz_offset")
        if tz_offset is None:
            logger.warning("No timezone for %s, skipping", triplet)
            continue

        safe_name = triplet.replace(":", "_")
        out_path = out_dir / f"{safe_name}.parquet"

        # Skip if already downloaded (resume semantics)
        if out_path.exists():
            stats[triplet] = -1  # already exists
            continue

        for attempt in range(max_retries):
            try:
                df = fetch_hourly_data(triplet, begin_date, end_date, elements)
                break
            except requests.RequestException as exc:
                if attempt < max_retries - 1:
                    logger.warning(
                        "%s attempt %d failed: %s, retrying in %.0fs",
                        triplet,
                        attempt + 1,
                        exc,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error("%s failed after %d attempts: %s", triplet, max_retries, exc)
                    df = pd.DataFrame()

        if df.empty:
            stats[triplet] = 0
            continue

        df = _convert_to_utc(df, float(tz_offset))

        # Add station metadata columns
        df["station_triplet"] = triplet
        df["station_name"] = meta.get("name")
        df["lat"] = meta.get("lat")
        df["lon"] = meta.get("lon")
        df["elev_ft"] = meta.get("elev_ft")
        df["physical_tz"] = meta.get("physical_tz", "")

        df.to_parquet(out_path, index=False, compression="snappy")
        stats[triplet] = len(df)

        if (i + 1) % 50 == 0:
            logger.info("  progress: %d/%d stations", i + 1, len(triplets))

    done = sum(1 for v in stats.values() if v > 0)
    skipped = sum(1 for v in stats.values() if v == -1)
    empty = sum(1 for v in stats.values() if v == 0)
    logger.info(
        "Download complete: %d downloaded, %d skipped (exist), %d empty",
        done,
        skipped,
        empty,
    )
    return stats
