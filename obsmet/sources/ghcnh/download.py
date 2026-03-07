"""GHCNh bulk downloader — fetch PSV station files from NCEI.

Downloads one PSV file per station from the v1beta access endpoint.
Each file contains the full period-of-record for that station.
~24,800 files, ~200 GB total.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BASE_URL = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-hourly/v1beta/access"
)
DEFAULT_RAW_DIR = "/nas/climate/ghcnh"
INDEX_URL = BASE_URL + "/"

# Timeout for individual file downloads (large files can be 200+ MB)
DOWNLOAD_TIMEOUT = 600


def list_remote_files() -> list[str]:
    """Scrape the NCEI directory listing for .psv filenames."""
    import re

    resp = requests.get(INDEX_URL, timeout=60)
    resp.raise_for_status()
    return sorted(re.findall(r'href="([A-Z0-9][^"]+\.psv)"', resp.text))


def download_file(filename: str, out_dir: Path, overwrite: bool = False) -> tuple[str, bool, str]:
    """Download a single PSV file. Returns (filename, success, message)."""
    out_path = out_dir / filename
    if out_path.exists() and not overwrite:
        return filename, True, "exists"

    url = f"{BASE_URL}/{filename}"
    try:
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()

        tmp_path = out_path.with_suffix(".psv.tmp")
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        tmp_path.rename(out_path)
        return filename, True, "ok"
    except Exception as e:
        return filename, False, str(e)


def download_all(
    out_dir: Path | str,
    workers: int = 8,
    overwrite: bool = False,
    done_keys: set[str] | None = None,
) -> list[tuple[str, bool, str]]:
    """Download all GHCNh PSV files in parallel.

    Parameters
    ----------
    out_dir : Target directory for PSV files.
    workers : Number of parallel download threads.
    overwrite : Re-download existing files.
    done_keys : Set of filenames already done (from manifest).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Listing remote GHCNh files...")
    remote_files = list_remote_files()
    logger.info("Found %d remote files", len(remote_files))

    if done_keys:
        remote_files = [f for f in remote_files if f not in done_keys]
        logger.info("After resume filter: %d files to download", len(remote_files))

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_file, f, out_dir, overwrite): f for f in remote_files}
        done_count = 0
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            done_count += 1
            if done_count % 100 == 0:
                ok = sum(1 for _, s, _ in results if s)
                fail = sum(1 for _, s, _ in results if not s)
                logger.info(
                    "Progress: %d/%d (ok=%d, fail=%d)", done_count, len(remote_files), ok, fail
                )

    return results
