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
        raw_dir: str | Path = "/nas/climate/ndbc/ndbc_records",
    ):
        self.raw_dir = Path(raw_dir)

    def discover_keys(self, start, end) -> list[str]:
        """List station IDs from parquet files in raw_dir."""
        keys = []
        for f in sorted(self.raw_dir.glob("*.parquet")):
            keys.append(f.stem.upper())
        return keys

    def fetch_raw(self, key: str, dest_dir: Path) -> Path:
        """Return path to raw parquet for a station key."""
        return self.raw_dir / f"{key.lower()}.parquet"

    def normalize_key(self, key: str, provenance: RunProvenance, **kwargs) -> pd.DataFrame | None:
        """Normalize a single NDBC station from pre-existing parquet."""
        raw_path = self.raw_dir / f"{key.lower()}.parquet"
        if not raw_path.exists():
            return None

        df = pd.read_parquet(raw_path)
        if df.empty:
            return None

        # The on-disk parquet has a tz-naive DatetimeIndex — reset and localize
        if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()
            # The index becomes a column; rename to datetime_utc
            idx_col = [
                c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()
            ]
            if idx_col:
                df = df.rename(columns={idx_col[0]: "datetime_utc"})

        if "datetime_utc" not in df.columns:
            # Try first column as datetime
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "datetime_utc"})

        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

        return normalize_to_canonical_wide(df, key, provenance)

    def output_filename(self, key: str) -> str:
        """Derive output filename from station ID."""
        return f"{key}.parquet"

    def normalize(self, raw_path: Path, provenance: RunProvenance) -> pd.DataFrame:
        """Read and normalize a station's parquet file."""
        station_id = raw_path.stem.upper()
        df = pd.read_parquet(raw_path)
        if df.empty:
            return pd.DataFrame()

        if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()
            idx_col = [
                c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()
            ]
            if idx_col:
                df = df.rename(columns={idx_col[0]: "datetime_utc"})

        if "datetime_utc" not in df.columns:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "datetime_utc"})

        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

        uri = f"ndbc://{raw_path}"
        return normalize_to_canonical_wide(df, station_id, provenance, raw_source_uri=uri)
