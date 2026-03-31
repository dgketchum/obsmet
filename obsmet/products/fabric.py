"""Observation fabric — unified multi-source product with precedence-based dedup.

Given geographic bounds, merges data from all matched sources per canonical station
using a configurable precedence matrix. Each variable at each timestep comes from
the highest-priority source that has a non-null, QC-passing value.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from obsmet.crosswalk.precedence import PrecedenceConfig

logger = logging.getLogger(__name__)

# Where to find per-station normalized (hourly or daily) data
_NORM_DIRS = {
    "ghcnd": "/mnt/mco_nas1/shared/obsmet/normalized/ghcnd",
    "ghcnh": "/mnt/mco_nas1/shared/obsmet/normalized/ghcnh",
    "isd": "/mnt/mco_nas1/shared/obsmet/normalized/isd",
    "snotel": "/mnt/mco_nas1/shared/obsmet/normalized/snotel",
    "ndbc": "/mnt/mco_nas1/shared/obsmet/normalized/ndbc",
    "raws_wrcc": "/mnt/mco_nas1/shared/obsmet/normalized/raws_wrcc",
    "eccc": "/mnt/mco_nas1/shared/obsmet/normalized/eccc",
}

# Sources with station_por products (daily aggregation + Tier 2 QC).
# Keyed by the crosswalk/precedence source name (not the CLI alias).
# For resolution="daily", these are preferred over _NORM_DIRS.
# For resolution="hourly", hourly-native sources (ndbc) skip this and use _NORM_DIRS.
_STATION_POR_DIRS = {
    "madis": "/mnt/mco_nas1/shared/obsmet/products/station_por/madis",
    "gdas": "/mnt/mco_nas1/shared/obsmet/products/station_por/gdas",
    "raws_wrcc": "/mnt/mco_nas1/shared/obsmet/products/station_por/raws",
    "ndbc": "/mnt/mco_nas1/shared/obsmet/products/station_por/ndbc",
    "snotel": "/mnt/mco_nas1/shared/obsmet/products/station_por/snotel",
}

# Hourly-native sources that should NOT use station_por for hourly resolution
_HOURLY_NATIVE_SOURCES = {"ndbc", "ghcnh"}


def _load_station_data(source: str, source_station_id: str, resolution: str) -> pd.DataFrame | None:
    """Load per-station data from normalized or station_por directory.

    For daily resolution, prefer station_por (QC'd daily aggregates) when available.
    For hourly resolution, hourly-native sources (ndbc, ghcnh) use normalized data
    directly — station_por files are daily and would be wrong for hourly fabric.
    """
    # Skip station_por for hourly-native sources when building hourly fabric
    use_por = source in _STATION_POR_DIRS and not (
        resolution == "hourly" and source in _HOURLY_NATIVE_SOURCES
    )
    if use_por:
        por_dir = Path(_STATION_POR_DIRS[source])
        candidates = [
            por_dir / f"{source}_{source_station_id}.parquet",
            por_dir / f"{source_station_id}.parquet",
            por_dir / f"{source}:{source_station_id}.parquet",
        ]
        for p in candidates:
            if p.exists():
                try:
                    return pd.read_parquet(p)
                except Exception:
                    pass
        return None

    # Per-station normalized files
    if source in _NORM_DIRS:
        norm_dir = Path(_NORM_DIRS[source])
        if source == "isd":
            frames = []
            for f in sorted(norm_dir.glob(f"*_{source_station_id}-*.parquet")):
                try:
                    frames.append(pd.read_parquet(f))
                except Exception:
                    pass
            if frames:
                return pd.concat(frames, ignore_index=True)
        # Try exact filename match
        p = norm_dir / f"{source_station_id}.parquet"
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                pass
        # SNOTEL filenames include name and state
        if source == "snotel":
            for f in norm_dir.glob(f"{source_station_id}_*.parquet"):
                try:
                    return pd.read_parquet(f)
                except Exception:
                    pass
    return None


def _apply_precedence_daily(
    source_dfs: dict[str, pd.DataFrame],
    precedence: dict[str, list[str]],
) -> pd.DataFrame:
    """Merge daily DataFrames from multiple sources using precedence."""
    # Collect all dates across sources
    all_dates = set()
    for df in source_dfs.values():
        if "date" in df.columns:
            all_dates.update(pd.to_datetime(df["date"]).dt.date)

    if not all_dates:
        return pd.DataFrame()

    date_index = sorted(all_dates)
    result = pd.DataFrame({"date": date_index})

    # Collect all variable columns across sources
    all_vars = set()
    for var_name, src_list in precedence.items():
        for src in src_list:
            if src in source_dfs:
                if var_name in source_dfs[src].columns:
                    all_vars.add(var_name)

    # For each variable, pick from highest-priority source
    for var_name in sorted(all_vars):
        src_priority = precedence.get(var_name, [])
        result[var_name] = np.nan
        result[f"{var_name}_source"] = ""

        for src in src_priority:
            if src not in source_dfs:
                continue
            sdf = source_dfs[src]
            if var_name not in sdf.columns:
                continue

            sdf_aligned = sdf.copy()
            sdf_aligned["date"] = pd.to_datetime(sdf_aligned["date"]).dt.date

            # Only take rows that passed QC
            if "qc_state" in sdf_aligned.columns:
                sdf_aligned = sdf_aligned[sdf_aligned["qc_state"] != "fail"]

            merged = result[["date"]].merge(
                sdf_aligned[["date", var_name]],
                on="date",
                how="left",
                suffixes=("", "_new"),
            )

            new_col = var_name + "_new" if var_name + "_new" in merged.columns else var_name
            fill_mask = result[var_name].isna() & merged[new_col].notna()
            result.loc[fill_mask, var_name] = merged.loc[fill_mask, new_col].values
            result.loc[fill_mask, f"{var_name}_source"] = src

    return result


def _apply_precedence_hourly(
    source_dfs: dict[str, pd.DataFrame],
    precedence: dict[str, list[str]],
) -> pd.DataFrame:
    """Merge hourly DataFrames from multiple sources using precedence."""
    all_times = set()
    for df in source_dfs.values():
        if "datetime_utc" in df.columns:
            all_times.update(pd.to_datetime(df["datetime_utc"], utc=True))

    if not all_times:
        return pd.DataFrame()

    time_index = sorted(all_times)
    result = pd.DataFrame({"datetime_utc": time_index})

    all_vars = set()
    for var_name, src_list in precedence.items():
        for src in src_list:
            if src in source_dfs and var_name in source_dfs[src].columns:
                all_vars.add(var_name)

    for var_name in sorted(all_vars):
        src_priority = precedence.get(var_name, [])
        result[var_name] = np.nan
        result[f"{var_name}_source"] = ""

        for src in src_priority:
            if src not in source_dfs:
                continue
            sdf = source_dfs[src]
            if var_name not in sdf.columns:
                continue

            sdf_aligned = sdf.copy()
            sdf_aligned["datetime_utc"] = pd.to_datetime(sdf_aligned["datetime_utc"], utc=True)

            if "qc_state" in sdf_aligned.columns:
                sdf_aligned = sdf_aligned[sdf_aligned["qc_state"] != "fail"]

            merged = result[["datetime_utc"]].merge(
                sdf_aligned[["datetime_utc", var_name]],
                on="datetime_utc",
                how="left",
                suffixes=("", "_new"),
            )

            new_col = var_name + "_new" if var_name + "_new" in merged.columns else var_name
            fill_mask = result[var_name].isna() & merged[new_col].notna()
            result.loc[fill_mask, var_name] = merged.loc[fill_mask, new_col].values
            result.loc[fill_mask, f"{var_name}_source"] = src

    return result


def build_fabric(
    crosswalk_path: Path,
    precedence: PrecedenceConfig,
    out_dir: Path,
    bounds: tuple[float, float, float, float] | None = None,
    resolution: str = "daily",
    start: str | None = None,
    end: str | None = None,
) -> dict[str, int]:
    """Build the observation fabric product.

    Parameters
    ----------
    crosswalk_path : Path to crosswalk.parquet
    precedence : PrecedenceConfig with per-variable source priority
    out_dir : Output directory for fabric parquet files
    bounds : Optional (west, south, east, north) bounding box filter
    resolution : "daily" or "hourly"
    start, end : Optional date range filter (YYYY-MM-DD)

    Returns
    -------
    Dict mapping canonical_station_id to row count written.
    """
    xwalk = pd.read_parquet(crosswalk_path)
    logger.info("Loaded crosswalk: %d entries", len(xwalk))

    # Filter to bounding box
    if bounds is not None:
        west, south, east, north = bounds
        xwalk = xwalk[
            (xwalk["lat"] >= south)
            & (xwalk["lat"] <= north)
            & (xwalk["lon"] >= west)
            & (xwalk["lon"] <= east)
        ]
        logger.info("After bounds filter: %d entries", len(xwalk))

    if xwalk.empty:
        logger.warning("No stations in bounds")
        return {}

    prec_map = precedence.daily if resolution == "daily" else precedence.hourly
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    canonical_groups = xwalk.groupby("canonical_station_id")
    n_groups = len(canonical_groups)

    for i, (canon_id, group) in enumerate(canonical_groups):
        # Load data from each source for this canonical station
        source_dfs = {}
        station_lat = group["lat"].median()
        station_lon = group["lon"].median()
        station_elev = group["elev_m"].median()

        for _, row in group.iterrows():
            src = row["source"]
            sid = row["source_station_id"]
            df = _load_station_data(src, sid, resolution)
            if df is not None and not df.empty:
                source_dfs[src] = df

        if not source_dfs:
            continue

        # Apply precedence
        if resolution == "daily":
            merged = _apply_precedence_daily(source_dfs, prec_map)
        else:
            merged = _apply_precedence_hourly(source_dfs, prec_map)

        if merged.empty:
            continue

        # Date range filter
        time_col = "date" if resolution == "daily" else "datetime_utc"
        if start is not None:
            start_dt = pd.Timestamp(start)
            merged = merged[pd.to_datetime(merged[time_col]) >= start_dt]
        if end is not None:
            end_dt = pd.Timestamp(end)
            merged = merged[pd.to_datetime(merged[time_col]) <= end_dt]

        if merged.empty:
            continue

        # Add station metadata
        merged.insert(0, "canonical_station_id", canon_id)
        merged.insert(1, "lat", station_lat)
        merged.insert(2, "lon", station_lon)
        merged.insert(3, "elev_m", station_elev)

        # Count contributing sources per row
        source_cols = [c for c in merged.columns if c.endswith("_source")]
        merged["n_sources"] = (merged[source_cols] != "").sum(axis=1)

        # Write
        safe_name = str(canon_id).replace(":", "_").replace("/", "_")
        out_path = out_dir / f"{safe_name}.parquet"
        merged.to_parquet(out_path, index=False, compression="snappy")
        stats[str(canon_id)] = len(merged)

        if (i + 1) % 500 == 0:
            logger.info("  progress: %d/%d canonical stations", i + 1, n_groups)

    logger.info(
        "Fabric complete: %d stations written, %d total rows",
        len(stats),
        sum(stats.values()),
    )
    return stats
