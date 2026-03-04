"""NDBC source adapter — maps NDBC stdmet data to obsmet canonical schema.

Implements the SourceAdapter interface for NOAA National Data Buoy Center
observations. Maps NDBC variable names to canonical names and preserves
buoy-specific extension variables (wave_height, water_temp, tide).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.sources.base import SourceAdapter

# --------------------------------------------------------------------------- #
# NDBC → canonical variable mapping
# --------------------------------------------------------------------------- #

# Core met variables (mapped to canonical names)
VARIABLE_MAP = {
    "air_temp": "tair",
    "dewpoint": "td",
    "wind_speed": "wind",
    "wind_dir": "wind_dir",
    "pressure": "slp",
    "visibility": "vis",
}

# Extension variables (buoy-specific, kept as-is)
EXTENSION_VARS = [
    "wave_height",
    "dominant_wave_period",
    "average_wave_period",
    "mean_wave_dir",
    "water_temp",
    "tide",
    "wind_gust",
]

UNIT_MAP = {
    "tair": "degC",
    "td": "degC",
    "wind": "m s-1",
    "wind_dir": "deg",
    "slp": "Pa",
    "vis": "nmi",
    "wave_height": "m",
    "water_temp": "degC",
    "wind_gust": "m s-1",
}


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def normalize_to_canonical_wide(
    df: pd.DataFrame,
    station_id: str,
    provenance: RunProvenance,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    raw_source_uri: str = "",
) -> pd.DataFrame:
    """Convert NDBC parsed DataFrame to canonical wide-form.

    NDBC data is already in standard units (°C, m/s, hPa, etc.),
    so no conversion is needed — just name mapping.
    """
    n = len(df)
    out = pd.DataFrame()

    out["datetime_utc"] = df["datetime_utc"].values
    out["station_key"] = [f"ndbc:{station_id}"] * n
    out["source"] = ["ndbc"] * n
    out["source_station_id"] = [station_id] * n
    out["lat"] = [latitude] * n
    out["lon"] = [longitude] * n

    # Map core met variable names; pressure needs hPa → Pa conversion
    for ndbc_var, canon_var in VARIABLE_MAP.items():
        if ndbc_var in df.columns:
            vals = pd.to_numeric(df[ndbc_var], errors="coerce").values
            if ndbc_var == "pressure":
                vals = vals * 100.0  # hPa → Pa
            out[canon_var] = vals

    # Keep extension variables
    for ext_var in EXTENSION_VARS:
        if ext_var in df.columns:
            out[ext_var] = pd.to_numeric(df[ext_var], errors="coerce").values

    out["ingest_run_id"] = provenance.run_id
    out["transform_version"] = provenance.transform_version

    return out


# --------------------------------------------------------------------------- #
# SourceAdapter implementation
# --------------------------------------------------------------------------- #


class NdbcAdapter(SourceAdapter):
    """NDBC source adapter."""

    source_name = "ndbc"

    def __init__(
        self,
        raw_dir: str | Path = "/nas/climate/obsmet/raw/ndbc",
    ):
        self.raw_dir = Path(raw_dir)

    def discover_keys(self, start, end) -> list[str]:
        """List station IDs that have data files in raw_dir."""
        import re

        keys = set()
        for f in self.raw_dir.glob("*.txt.gz"):
            m = re.match(r"(\w+)h\d{4}\.txt\.gz", f.name)
            if m:
                keys.add(m.group(1).upper())
        return sorted(keys)

    def fetch_raw(self, key: str, dest_dir: Path) -> Path:
        """Return path to raw directory for a station key."""
        return self.raw_dir

    def normalize_key(self, key: str, provenance: RunProvenance, **kwargs) -> pd.DataFrame | None:
        """Normalize a single NDBC station key."""
        from obsmet.sources.ndbc.extract import read_station_files

        df = read_station_files(self.raw_dir, key)
        if df.empty:
            return None
        return normalize_to_canonical_wide(df, key, provenance)

    def output_filename(self, key: str) -> str:
        """Derive output filename from station ID."""
        return f"{key}.parquet"

    def normalize(self, raw_path: Path, provenance: RunProvenance) -> pd.DataFrame:
        """Read and normalize a station's stdmet files."""
        from obsmet.sources.ndbc.extract import read_station_files

        # Determine station ID from filename pattern
        station_id = raw_path.stem.split("h")[0].upper() if "h" in raw_path.stem else raw_path.stem
        df = read_station_files(raw_path.parent, station_id)
        if df.empty:
            return pd.DataFrame()

        uri = f"ndbc://{raw_path}"
        return normalize_to_canonical_wide(df, station_id, provenance, raw_source_uri=uri)
