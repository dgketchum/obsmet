"""Crosswalk builder — match stations across sources.

Reads station_index.parquet, matches stations by:
  1. Exact GHCN ID match (GHCNh <-> GHCN-Daily share the same namespace)
  2. Geospatial proximity (<1 km) for cross-source matching

Outputs crosswalk.parquet with canonical_station_id assigned per group.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from obsmet.crosswalk.matchers import MatchConfidence, haversine_distance

logger = logging.getLogger(__name__)

# Maximum distance (meters) for a geospatial match
_MAX_DISTANCE_M = 1000.0

# Sources that use GHCN station IDs (the suffix after the colon is the GHCN ID)
_GHCN_SOURCES = {"ghcnh", "ghcnd"}

# When assigning canonical IDs to non-GHCN-only clusters, prefer this order
_CANONICAL_PRIORITY = ["madis", "gdas", "ndbc", "raws_wrcc", "snotel"]


def _extract_ghcn_id(canonical_id: str, source: str) -> str | None:
    """Extract the GHCN station ID suffix if the source uses GHCN IDs."""
    if source in _GHCN_SOURCES:
        return canonical_id.split(":", 1)[-1] if ":" in canonical_id else None
    return None


def build_crosswalk(
    index_path: Path,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Build station crosswalk from a station index.

    Parameters
    ----------
    index_path : Path to station_index.parquet
    out_path : Where to write crosswalk.parquet. If None, returns DataFrame only.
    """
    idx = pd.read_parquet(index_path)
    logger.info("Loaded station index: %d entries", len(idx))

    # ------------------------------------------------------------------ #
    # Phase 1: Exact GHCN ID matching
    # ------------------------------------------------------------------ #
    idx["ghcn_id"] = [
        _extract_ghcn_id(cid, src) for cid, src in zip(idx["canonical_id"], idx["source"])
    ]

    # Group GHCN sources by their shared ID
    ghcn_mask = idx["ghcn_id"].notna()
    ghcn_df = idx[ghcn_mask].copy()
    non_ghcn_df = idx[~ghcn_mask].copy()

    # Assign canonical_station_id for GHCN-matched stations
    # Use ghcn:<ID> as the canonical form
    ghcn_df["canonical_station_id"] = "ghcn:" + ghcn_df["ghcn_id"]
    ghcn_df["confidence"] = MatchConfidence.EXACT.value
    ghcn_df["match_method"] = "id_exact_ghcn"
    ghcn_df["distance_m"] = np.nan

    logger.info(
        "Phase 1: %d GHCN entries, %d unique canonical stations",
        len(ghcn_df),
        ghcn_df["canonical_station_id"].nunique(),
    )

    # ------------------------------------------------------------------ #
    # Phase 2: Geospatial proximity matching (non-GHCN vs GHCN + each other)
    # ------------------------------------------------------------------ #
    # For non-GHCN stations, try to match to the nearest GHCN station
    # Use spatial binning to keep runtime manageable

    # Build a reference set of GHCN station locations (one per canonical)
    ghcn_ref = (
        ghcn_df.groupby("canonical_station_id")
        .agg({"lat": "median", "lon": "median", "elev_m": "median"})
        .reset_index()
    )

    match_results = []
    unmatched = []

    if not non_ghcn_df.empty and not ghcn_ref.empty:
        ref_lats = ghcn_ref["lat"].values
        ref_lons = ghcn_ref["lon"].values
        ref_ids = ghcn_ref["canonical_station_id"].values

        for _, row in non_ghcn_df.iterrows():
            if np.isnan(row["lat"]) or np.isnan(row["lon"]):
                unmatched.append(row)
                continue

            # Coarse filter: only check stations within ~0.02 degrees (~2 km)
            lat_close = np.abs(ref_lats - row["lat"]) < 0.02
            lon_close = np.abs(ref_lons - row["lon"]) < 0.02
            candidates = lat_close & lon_close

            if not candidates.any():
                unmatched.append(row)
                continue

            cand_idx = np.where(candidates)[0]
            distances = np.array(
                [
                    haversine_distance(row["lat"], row["lon"], ref_lats[i], ref_lons[i])
                    for i in cand_idx
                ]
            )

            min_idx = distances.argmin()
            min_dist = distances[min_idx]

            if min_dist <= _MAX_DISTANCE_M:
                match_results.append(
                    {
                        "canonical_id": row["canonical_id"],
                        "source": row["source"],
                        "source_station_id": row["source_station_id"],
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "elev_m": row["elev_m"],
                        "canonical_station_id": ref_ids[cand_idx[min_idx]],
                        "confidence": MatchConfidence.PROBABLE.value,
                        "match_method": "geospatial_proximity",
                        "distance_m": float(min_dist),
                    }
                )
            else:
                unmatched.append(row)

    logger.info(
        "Phase 2: %d geospatial matches, %d unmatched",
        len(match_results),
        len(unmatched),
    )

    # ------------------------------------------------------------------ #
    # Phase 3: Unmatched stations get their own canonical ID
    # ------------------------------------------------------------------ #
    unmatched_records = []
    for row in unmatched:
        row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        unmatched_records.append(
            {
                "canonical_id": row_dict["canonical_id"],
                "source": row_dict["source"],
                "source_station_id": row_dict["source_station_id"],
                "lat": row_dict["lat"],
                "lon": row_dict["lon"],
                "elev_m": row_dict["elev_m"],
                "canonical_station_id": row_dict["canonical_id"],
                "confidence": MatchConfidence.NONE.value,
                "match_method": "none",
                "distance_m": np.nan,
            }
        )

    # ------------------------------------------------------------------ #
    # Combine all results
    # ------------------------------------------------------------------ #
    ghcn_records = ghcn_df[
        [
            "canonical_id",
            "source",
            "source_station_id",
            "lat",
            "lon",
            "elev_m",
            "canonical_station_id",
            "confidence",
            "match_method",
            "distance_m",
        ]
    ].to_dict("records")

    all_records = ghcn_records + match_results + unmatched_records
    result = pd.DataFrame(all_records)

    if not result.empty:
        result = result.rename(columns={"canonical_id": "source_station_key"})
        result = result.sort_values(["canonical_station_id", "source"]).reset_index(drop=True)

    logger.info(
        "Crosswalk complete: %d entries, %d canonical stations",
        len(result),
        result["canonical_station_id"].nunique() if not result.empty else 0,
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path, index=False)
        logger.info("Wrote crosswalk: %s", out_path)

    return result
