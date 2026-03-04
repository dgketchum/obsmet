"""ISD archive download from S3.

Downloads ISD station-year .gz files from the public NOAA S3 bucket
(noaa-isd-pds) with manifest-based resume semantics.

Ported from pyisd/scripts/download_isd_archive.py.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_RAW_DIR = "/nas/climate/isd/raw"

_S3_BUCKET = "noaa-isd-pds"


def _make_s3_client():
    """Create an unsigned S3 client for public bucket access."""
    import boto3
    from botocore.config import Config
    from botocore import UNSIGNED

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_s3_keys(year: int) -> list[str]:
    """List all ISD .gz keys for a given year on S3."""
    client = _make_s3_client()
    prefix = f"data/{year}/"
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".gz"):
                keys.append(key)
    return keys


def download_file(
    s3_key: str,
    dest_dir: Path,
    *,
    retries: int = 3,
) -> tuple[str, bool, str]:
    """Download a single ISD file from S3.

    Returns (s3_key, success, message).
    """
    # s3_key like "data/2024/720538-00164-2024.gz"
    filename = s3_key.split("/")[-1]
    year = s3_key.split("/")[1]
    local_dir = dest_dir / year
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / filename

    if local_path.exists() and local_path.stat().st_size > 0:
        return s3_key, True, "exists"

    client = _make_s3_client()
    for attempt in range(retries):
        try:
            client.download_file(_S3_BUCKET, s3_key, str(local_path))
            return s3_key, True, "ok"
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                return s3_key, False, str(exc)
    return s3_key, False, "exhausted retries"


def download_year(
    year: int,
    dest_dir: Path | str,
    *,
    workers: int = 16,
    done_keys: set[str] | None = None,
) -> list[tuple[str, bool, str]]:
    """Download all ISD files for a year using threaded S3 access.

    Parameters
    ----------
    year : Year to download.
    dest_dir : Root destination directory.
    workers : Number of download threads.
    done_keys : Set of S3 keys already downloaded (for resume).

    Returns list of (s3_key, success, message) tuples.
    """
    dest_dir = Path(dest_dir)
    keys = list_s3_keys(year)
    if done_keys:
        keys = [k for k in keys if k not in done_keys]

    if not keys:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_file, k, dest_dir): k for k in keys}
        for fut in as_completed(futures):
            results.append(fut.result())

    return results
