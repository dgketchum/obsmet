"""Download small test datasets for integration tests.

Fetches minimal raw data for MADIS (2 days), NDBC (3 stations × 1 year),
and RAWS (3 stations) into /tmp directories.

Run: uv run python tests/integration/ingest_test_data.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


def ingest_madis():
    """Download 2 days of MADIS raw netCDF to /tmp."""
    from obsmet.sources.madis.download import download_day, load_credentials

    dest = Path("/tmp/obsmet_inttest_madis_raw")
    dest.mkdir(parents=True, exist_ok=True)

    try:
        username, password = load_credentials()
    except FileNotFoundError:
        print("MADIS: credentials not found, skipping")
        return False

    days = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    ok = 0
    for day in days:
        day_str, success, msg = download_day(day, dest, username, password)
        print(f"  MADIS {day_str}: {msg}")
        if success:
            ok += 1
    print(f"MADIS: {ok}/{len(days)} days ok → {dest}")
    return ok > 0


def ingest_ndbc():
    """Download 3 well-known buoys × 1 year to /tmp."""
    from obsmet.sources.ndbc.download import download_station_year

    dest = Path("/tmp/obsmet_inttest_ndbc_raw")
    dest.mkdir(parents=True, exist_ok=True)

    stations = ["41001", "46025", "BURL1"]
    year = 2024
    ok = 0
    for sid in stations:
        _, _, success, msg = download_station_year(sid, year, dest)
        print(f"  NDBC {sid}/{year}: {msg}")
        if success:
            ok += 1
    print(f"NDBC: {ok}/{len(stations)} station-years ok → {dest}")
    return ok > 0


def ingest_raws():
    """Download 3 RAWS stations to /tmp."""
    from obsmet.sources.raws_wrcc.download import (
        download_station_data,
        scrape_region_stations,
    )
    from obsmet.sources.raws_wrcc.extract import parse_response

    dest = Path("/tmp/obsmet_inttest_raws_raw")
    dest.mkdir(parents=True, exist_ok=True)

    # Scrape a few real station IDs from eastern MT region
    print("  Scraping station inventory...")
    all_stns = scrape_region_stations("emt")
    stations = [s["wrcc_id"] for s in all_stns[:3]]
    if not stations:
        print("  RAWS: failed to scrape station inventory")
        return False
    print(f"  Using stations: {stations}")
    start = datetime(2024, 6, 1).date()
    end = datetime(2024, 6, 30).date()
    ok = 0
    for wrcc_id in stations:
        try:
            html = download_station_data(wrcc_id, start, end)
            df = parse_response(html)
            if df.empty:
                print(f"  RAWS {wrcc_id}: no data")
                continue
            out = dest / f"{wrcc_id}.parquet"
            df.to_parquet(out, index=False, compression="snappy")
            print(f"  RAWS {wrcc_id}: {len(df)} rows → {out.name}")
            ok += 1
        except Exception as exc:
            print(f"  RAWS {wrcc_id}: FAILED {exc}")
    print(f"RAWS: {ok}/{len(stations)} stations ok → {dest}")
    return ok > 0


if __name__ == "__main__":
    sources = sys.argv[1:] if len(sys.argv) > 1 else ["madis", "ndbc", "raws"]

    results = {}
    if "madis" in sources:
        print("=== MADIS ===")
        results["madis"] = ingest_madis()
    if "ndbc" in sources:
        print("=== NDBC ===")
        results["ndbc"] = ingest_ndbc()
    if "raws" in sources:
        print("=== RAWS ===")
        results["raws"] = ingest_raws()

    print("\n=== Summary ===")
    for src, ok in results.items():
        print(f"  {src}: {'OK' if ok else 'FAILED'}")
