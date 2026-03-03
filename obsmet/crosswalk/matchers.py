"""Station identity matching (plan section 13).

Build station_crosswalk.parquet with confidence classes:
exact_match, probable_match, manual_required, no_match.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class MatchConfidence(str, Enum):
    EXACT = "exact_match"
    PROBABLE = "probable_match"
    MANUAL = "manual_required"
    NONE = "no_match"


@dataclass(frozen=True)
class CrosswalkMatch:
    """A single station identity match between two sources."""

    station_key: str
    source_a: str
    source_a_id: str
    source_b: str
    source_b_id: str
    confidence: MatchConfidence
    match_method: str  # e.g. "id_exact", "geospatial_proximity", "name_similarity"
    distance_m: float | None = None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two lat/lon points."""
    R = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(R * 2 * np.arcsin(np.sqrt(a)))
