"""Release builder — versioned, auditable dataset snapshots.

Layout:
    /nas/climate/obsmet/releases/v0.1.0/
        release_metadata.json
        manifest.parquet
        station_por/<source>/<station_key>.parquet  (hardlinks)

    /nas/climate/obsmet/channels/
        candidate -> ../releases/v0.1.0
        prod -> ../releases/v0.1.0
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from obsmet.core.provenance import RunProvenance, file_hash

logger = logging.getLogger(__name__)

RELEASES_ROOT = Path("/nas/climate/obsmet/releases")
CHANNELS_ROOT = Path("/nas/climate/obsmet/channels")


def build_release(
    version: str,
    channel: str,
    sources: list[str],
    provenance: RunProvenance,
    *,
    qc_profile: str = "",
    tier2_rules: str = "",
    station_por_root: Path | str = "/nas/climate/obsmet/products/station_por",
    releases_root: Path | str | None = None,
    channels_root: Path | str | None = None,
) -> Path:
    """Build a versioned release by hardlinking station POR parquets.

    Returns the release directory path.
    """
    r_root = Path(releases_root) if releases_root else RELEASES_ROOT
    c_root = Path(channels_root) if channels_root else CHANNELS_ROOT
    station_por_root = Path(station_por_root)

    release_dir = r_root / version
    release_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []

    for source in sources:
        src_dir = station_por_root / source
        if not src_dir.exists():
            logger.warning("Station POR dir not found: %s", src_dir)
            continue

        dest_dir = release_dir / "station_por" / source
        dest_dir.mkdir(parents=True, exist_ok=True)

        for pf in sorted(src_dir.glob("*.parquet")):
            if pf.name == "manifest.parquet":
                continue

            dest = dest_dir / pf.name
            # Hardlink (fall back to copy on cross-filesystem)
            try:
                if dest.exists():
                    dest.unlink()
                os.link(pf, dest)
            except OSError:
                shutil.copy2(pf, dest)

            # Compute hash and collect metadata
            sha = file_hash(dest)
            station_key = pf.stem

            try:
                df = pd.read_parquet(pf)
                row_count = len(df)
                if "date" in df.columns:
                    dates = pd.to_datetime(df["date"])
                    date_min = str(dates.min().date())
                    date_max = str(dates.max().date())
                else:
                    date_min = ""
                    date_max = ""
                qc_state = (
                    df["qc_state"].value_counts().to_dict() if "qc_state" in df.columns else {}
                )
            except Exception:
                row_count = 0
                date_min = ""
                date_max = ""
                qc_state = {}

            manifest_records.append(
                {
                    "station_key": station_key,
                    "source": source,
                    "qc_summary": json.dumps(qc_state),
                    "row_count": row_count,
                    "date_min": date_min,
                    "date_max": date_max,
                    "sha256": sha,
                }
            )

    # Write manifest parquet
    if manifest_records:
        mdf = pd.DataFrame(manifest_records)
        manifest_schema = pa.schema(
            [
                pa.field("station_key", pa.string()),
                pa.field("source", pa.string()),
                pa.field("qc_summary", pa.string()),
                pa.field("row_count", pa.int64()),
                pa.field("date_min", pa.string()),
                pa.field("date_max", pa.string()),
                pa.field("sha256", pa.string()),
            ]
        )
        table = pa.Table.from_pandas(mdf, schema=manifest_schema)
        pq.write_table(table, release_dir / "manifest.parquet")

    # Write release metadata
    metadata = {
        "version": version,
        "schema_version": provenance.schema_version,
        "qaqc_rules_version": provenance.qaqc_rules_version,
        "transform_version": provenance.transform_version,
        "release_timestamp": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "station_count": len(manifest_records),
        "qc_profile": qc_profile,
        "tier2_rules": tier2_rules,
        "provenance": provenance.to_dict(),
    }
    with open(release_dir / "release_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Create/update channel symlink
    c_root.mkdir(parents=True, exist_ok=True)
    link = c_root / channel
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(release_dir)

    return release_dir


def validate_release(
    version: str,
    *,
    releases_root: Path | str | None = None,
) -> tuple[bool, list[str]]:
    """Validate release by checking file checksums against manifest.

    Returns (ok, list_of_errors).
    """
    r_root = Path(releases_root) if releases_root else RELEASES_ROOT
    release_dir = r_root / version

    manifest_path = release_dir / "manifest.parquet"
    if not manifest_path.exists():
        return False, ["manifest.parquet not found"]

    mdf = pd.read_parquet(manifest_path)
    errors = []

    for _, row in mdf.iterrows():
        source = row["source"]
        station_key = row["station_key"]
        expected_sha = row["sha256"]
        fpath = release_dir / "station_por" / source / f"{station_key}.parquet"

        if not fpath.exists():
            errors.append(f"missing: {fpath}")
            continue

        actual_sha = file_hash(fpath)
        if actual_sha != expected_sha:
            errors.append(
                f"checksum mismatch: {fpath} (expected {expected_sha[:12]}..., got {actual_sha[:12]}...)"
            )

    return len(errors) == 0, errors


def promote_release(
    version: str,
    channel: str,
    *,
    releases_root: Path | str | None = None,
    channels_root: Path | str | None = None,
) -> None:
    """Promote a release to a channel after validation.

    Validates checksums first, then updates the channel symlink.
    Raises ValueError on validation failure.
    """
    r_root = Path(releases_root) if releases_root else RELEASES_ROOT
    c_root = Path(channels_root) if channels_root else CHANNELS_ROOT

    ok, errors = validate_release(version, releases_root=r_root)
    if not ok:
        raise ValueError(f"Release {version} failed validation: {errors}")

    release_dir = r_root / version
    c_root.mkdir(parents=True, exist_ok=True)
    link = c_root / channel
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(release_dir)
