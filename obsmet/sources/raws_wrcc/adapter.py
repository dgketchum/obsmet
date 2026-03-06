"""RAWS WRCC source adapter — maps WRCC daily data to obsmet canonical schema.

Implements the SourceAdapter interface for RAWS data from the Western
Regional Climate Center. Data arrives as daily summaries (already metric),
so the adapter performs name mapping with no unit conversion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.sources.base import SourceAdapter

# --------------------------------------------------------------------------- #
# RAWS → canonical variable mapping
# --------------------------------------------------------------------------- #

# WRCC daily column names → obsmet canonical variable names
VARIABLE_MAP = {
    "tair_ave_c": "tmean",
    "tair_max_c": "tmax",
    "tair_min_c": "tmin",
    "wspd_ave_ms": "wind",
    "wdir_vec_deg": "wind_dir",
    "wspd_gust_ms": "wind_gust",
    "rh_ave_pct": "rh",
    "rh_max_pct": "rh_max",
    "rh_min_pct": "rh_min",
    "prcp_total_mm": "prcp",
    "srad_total_kwh_m2": "rsds",
}

# All RAWS data arrives in metric, no conversion needed
UNIT_MAP = {
    "tmean": "degC",
    "tmax": "degC",
    "tmin": "degC",
    "wind": "m s-1",
    "wind_dir": "deg",
    "wind_gust": "m s-1",
    "rh": "percent",
    "rh_max": "percent",
    "rh_min": "percent",
    "prcp": "mm",
    "rsds": "kWh m-2",
}


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def normalize_to_canonical_wide(
    df: pd.DataFrame,
    wrcc_id: str,
    provenance: RunProvenance,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    elevation_m: float | None = None,
    raw_source_uri: str = "",
) -> pd.DataFrame:
    """Convert RAWS parsed DataFrame to canonical wide-form.

    RAWS provides daily data (not hourly), so the output is already
    at daily resolution. The 'date' column is used instead of datetime_utc.
    """
    n = len(df)
    out = pd.DataFrame()

    out["date"] = df["date"].values
    out["station_key"] = [f"raws:{wrcc_id}"] * n
    out["source"] = ["raws_wrcc"] * n
    out["source_station_id"] = [wrcc_id] * n
    out["lat"] = [latitude] * n
    out["lon"] = [longitude] * n
    out["elev_m"] = [elevation_m] * n

    # Map variable names (no unit conversion — already metric)
    for wrcc_col, canon_var in VARIABLE_MAP.items():
        if wrcc_col in df.columns:
            out[canon_var] = pd.to_numeric(df[wrcc_col], errors="coerce").values

    out["ingest_run_id"] = provenance.run_id
    out["transform_version"] = provenance.transform_version

    return out


# --------------------------------------------------------------------------- #
# SourceAdapter implementation
# --------------------------------------------------------------------------- #


class RawsAdapter(SourceAdapter):
    """RAWS WRCC source adapter."""

    source_name = "raws_wrcc"

    def __init__(
        self,
        raw_dir: str | Path = "/nas/climate/raws/wrcc/station_data",
    ):
        self.raw_dir = Path(raw_dir)

    def discover_keys(self, start, end) -> list[str]:
        """List station CSV files available in raw_dir.

        Keys are station WRCC IDs derived from filenames.
        """
        keys = []
        for f in sorted(self.raw_dir.glob("*.csv")):
            keys.append(f.stem)
        return keys

    def fetch_raw(self, key: str, dest_dir: Path) -> Path:
        """Return path to raw CSV for a station key."""
        return self.raw_dir / f"{key}.csv"

    def normalize(self, raw_path: Path, provenance: RunProvenance) -> pd.DataFrame:
        """Read and normalize a single station's raw CSV."""
        wrcc_id = raw_path.stem
        df = pd.read_csv(raw_path)
        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        return normalize_to_canonical_wide(df, wrcc_id, provenance)
