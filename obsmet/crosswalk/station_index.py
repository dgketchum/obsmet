"""Station metadata index — scan normalized parquet to catalog every station.

Produces station_index.parquet with one row per (source, station):
  canonical_id, source, source_station_id, lat, lon, elev_m,
  por_start, por_end, obs_count, temporal_res
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sources whose normalized output is one parquet per station
_PER_STATION_SOURCES = {"ghcnd", "ghcnh", "snotel", "ndbc", "raws_wrcc"}

# Sources whose normalized output is one parquet per day (many stations per file)
_PER_DAY_SOURCES = {"madis", "gdas"}

_TEMPORAL_RES = {
    "ghcnd": "daily",
    "ghcnh": "hourly",
    "snotel": "daily",
    "ndbc": "hourly",
    "raws_wrcc": "daily",
    "madis": "hourly",
    "gdas": "hourly",
}

# How to derive station_key from filename when the column is NaN
_FILENAME_KEY_PREFIX = {
    "ghcnd": "ghcnd",
    "snotel": "snotel",
    "ghcnh": "ghcnh",
    "ndbc": "ndbc",
    "raws_wrcc": "raws",
}


def _station_id_from_filename(filename: str, source: str) -> str:
    """Derive source_station_id from a parquet filename."""
    stem = Path(filename).stem
    if source == "snotel":
        m = re.match(r"^(\d+)_", stem)
        return m.group(1) if m else stem
    return stem


def _index_per_station_source(source: str, norm_dir: Path, time_col: str) -> list[dict]:
    """Index a source with one parquet file per station."""
    records = []
    parquet_files = sorted(norm_dir.glob("*.parquet"))
    parquet_files = [p for p in parquet_files if p.name != "manifest.parquet"]

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception:
            continue

        if df.empty:
            continue

        # Derive station identity
        sk = df.get("station_key")
        if sk is not None and sk.notna().any():
            station_key = sk.dropna().iloc[0]
            src = df["source"].dropna().iloc[0] if "source" in df.columns else source
            sid = (
                df["source_station_id"].dropna().iloc[0]
                if "source_station_id" in df.columns
                else station_key.split(":", 1)[-1]
            )
        else:
            # Fallback: derive from filename
            sid = _station_id_from_filename(pf.name, source)
            prefix = _FILENAME_KEY_PREFIX.get(source, source)
            station_key = f"{prefix}:{sid}"
            src = source

        lat = df["lat"].median() if "lat" in df.columns else np.nan
        lon = df["lon"].median() if "lon" in df.columns else np.nan
        elev = df["elev_m"].median() if "elev_m" in df.columns else np.nan

        if time_col in df.columns:
            times = pd.to_datetime(df[time_col], errors="coerce").dropna()
            por_start = times.min() if len(times) else pd.NaT
            por_end = times.max() if len(times) else pd.NaT
        else:
            por_start = pd.NaT
            por_end = pd.NaT

        records.append(
            {
                "canonical_id": station_key,
                "source": src,
                "source_station_id": sid,
                "lat": float(lat),
                "lon": float(lon),
                "elev_m": float(elev),
                "por_start": por_start,
                "por_end": por_end,
                "obs_count": len(df),
                "temporal_res": _TEMPORAL_RES.get(source, "unknown"),
            }
        )

    return records


def _index_per_day_source(source: str, norm_dir: Path, sample_days: int = 30) -> list[dict]:
    """Index a source with one parquet file per day (many stations per file)."""
    parquet_files = sorted(norm_dir.glob("*.parquet"))
    parquet_files = [p for p in parquet_files if p.name != "manifest.parquet"]

    if not parquet_files:
        return []

    # Sample evenly across available files
    if len(parquet_files) > sample_days:
        indices = np.linspace(0, len(parquet_files) - 1, sample_days, dtype=int)
        sampled = [parquet_files[i] for i in indices]
    else:
        sampled = parquet_files

    # Accumulate per-station stats across sampled files
    station_data: dict[str, dict] = {}

    for pf in sampled:
        try:
            df = pd.read_parquet(pf)
        except Exception:
            continue

        if df.empty or "station_key" not in df.columns:
            continue

        for sk, grp in df.groupby("station_key"):
            if pd.isna(sk):
                continue
            if sk not in station_data:
                station_data[sk] = {
                    "lats": [],
                    "lons": [],
                    "elevs": [],
                    "obs_count": 0,
                    "source": grp["source"].iloc[0] if "source" in grp.columns else source,
                    "source_station_id": (
                        grp["source_station_id"].iloc[0]
                        if "source_station_id" in grp.columns
                        else str(sk).split(":", 1)[-1]
                    ),
                }
            sd = station_data[sk]
            if "lat" in grp.columns:
                sd["lats"].extend(grp["lat"].dropna().tolist())
            if "lon" in grp.columns:
                sd["lons"].extend(grp["lon"].dropna().tolist())
            if "elev_m" in grp.columns:
                sd["elevs"].extend(grp["elev_m"].dropna().tolist())
            sd["obs_count"] += len(grp)

    # Get POR from first and last files
    por_start = pd.NaT
    por_end = pd.NaT
    try:
        first_df = pd.read_parquet(parquet_files[0], columns=["datetime_utc"])
        por_start = pd.to_datetime(first_df["datetime_utc"], errors="coerce").min()
    except Exception:
        pass
    try:
        last_df = pd.read_parquet(parquet_files[-1], columns=["datetime_utc"])
        por_end = pd.to_datetime(last_df["datetime_utc"], errors="coerce").max()
    except Exception:
        pass

    records = []
    for sk, sd in station_data.items():
        records.append(
            {
                "canonical_id": sk,
                "source": sd["source"],
                "source_station_id": sd["source_station_id"],
                "lat": float(np.median(sd["lats"])) if sd["lats"] else np.nan,
                "lon": float(np.median(sd["lons"])) if sd["lons"] else np.nan,
                "elev_m": float(np.median(sd["elevs"])) if sd["elevs"] else np.nan,
                "por_start": por_start,
                "por_end": por_end,
                "obs_count": sd["obs_count"],
                "temporal_res": _TEMPORAL_RES.get(source, "hourly"),
            }
        )

    return records


def build_station_index(
    norm_base: Path,
    sources: list[str] | None = None,
    out_path: Path | None = None,
    sample_days: int = 30,
) -> pd.DataFrame:
    """Build a station metadata index from all normalized source directories.

    Parameters
    ----------
    norm_base : Root of normalized output (e.g. /mnt/mco_nas1/shared/obsmet/normalized)
    sources : List of source names to index. Default: all present directories.
    out_path : Where to write station_index.parquet. If None, returns DataFrame only.
    sample_days : Number of day files to sample for per-day sources.
    """
    all_sources = sources or [
        d.name for d in sorted(norm_base.iterdir()) if d.is_dir() and d.name != "isd"
    ]

    all_records = []
    for source in all_sources:
        norm_dir = norm_base / source
        if not norm_dir.exists():
            logger.info("Skipping %s: directory not found", source)
            continue

        logger.info("Indexing %s ...", source)

        if source in _PER_DAY_SOURCES:
            records = _index_per_day_source(source, norm_dir, sample_days)
        else:
            time_col = "date" if _TEMPORAL_RES.get(source) == "daily" else "datetime_utc"
            records = _index_per_station_source(source, norm_dir, time_col)

        logger.info("  %s: %d stations", source, len(records))
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    if not df.empty:
        df = df.sort_values(["source", "canonical_id"]).reset_index(drop=True)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("Wrote station index: %s (%d stations)", out_path, len(df))

    return df
