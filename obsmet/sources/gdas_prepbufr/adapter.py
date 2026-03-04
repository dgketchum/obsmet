"""GDAS PrepBUFR source adapter — maps GDAS surface obs to obsmet canonical schema.

Implements the SourceAdapter interface for GDAS PrepBUFR ADPSFC/SFCSHP
observations. Handles variable mapping (TOB → tair, POB → psfc, UOB/VOB
→ u/v wind components), QM column preservation, and normalization.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.sources.base import SourceAdapter

# --------------------------------------------------------------------------- #
# GDAS → canonical variable mapping
# --------------------------------------------------------------------------- #

VARIABLE_MAP = {
    "temperature": "tair",
    "pressure": "psfc",
    "specific_humidity": "q",
    "u_wind": "u",
    "v_wind": "v",
    "sst": "sst",
}

UNIT_MAP = {
    "temperature": "degC",
    "pressure": "Pa",
    "specific_humidity": "kg kg-1",
    "u_wind": "m s-1",
    "v_wind": "m s-1",
    "sst": "K",
}

# QM columns to preserve alongside each variable
QM_MAP = {
    "temperature": "temperature_qm",
    "pressure": "pressure_qm",
    "specific_humidity": "humidity_qm",
    "u_wind": "wind_qm",
    "v_wind": "wind_qm",
    "sst": "sst_qm",
}


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def normalize_to_canonical_wide(
    df: pd.DataFrame,
    provenance: RunProvenance,
    *,
    raw_source_uri: str = "",
) -> pd.DataFrame:
    """Convert GDAS extraction DataFrame to canonical wide-form.

    The extraction layer already handles unit conversions (hPa→Pa, mg/kg→kg/kg),
    so no additional conversion is needed here.
    """
    out = pd.DataFrame()

    out["station_key"] = "gdas:" + df["station_id"].astype(str)
    out["source"] = df["msg_type"].str.lower().apply(lambda x: f"gdas_{x}")
    out["source_station_id"] = df["station_id"].astype(str)
    out["datetime_utc"] = df["datetime_utc"]
    out["lat"] = df["latitude"].values
    out["lon"] = df["longitude"].values
    out["elev_m"] = df["elevation"].values
    out["cycle"] = df["cycle"].values
    out["obs_type"] = df["obs_type"].values
    out["msg_type"] = df["msg_type"].values

    # Map variable names (units already converted in extract layer)
    for gdas_var, canon_var in VARIABLE_MAP.items():
        if gdas_var in df.columns:
            out[canon_var] = pd.to_numeric(df[gdas_var], errors="coerce").values

    # Preserve QM columns
    for gdas_var, qm_col in QM_MAP.items():
        if qm_col in df.columns:
            canon_qm = f"{VARIABLE_MAP[gdas_var]}_qm"
            if canon_qm not in out.columns:
                out[canon_qm] = df[qm_col].values

    out["ingest_run_id"] = provenance.run_id
    out["transform_version"] = provenance.transform_version

    return out


# --------------------------------------------------------------------------- #
# SourceAdapter implementation
# --------------------------------------------------------------------------- #


class GdasAdapter(SourceAdapter):
    """GDAS PrepBUFR source adapter."""

    source_name = "gdas"

    def __init__(
        self,
        raw_dir: str | Path = "/nas/climate/gdas/prepbufr",
    ):
        self.raw_dir = Path(raw_dir)

    def discover_keys(self, start, end) -> list[str]:
        """List YYYYMMDD day strings in [start, end]."""
        keys = []
        current = start
        while current <= end:
            keys.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return keys

    def fetch_raw(self, key: str, dest_dir: Path) -> Path:
        """Return path to raw tar archive for a given day key."""
        year = key[:4]
        for suffix in [".nr.tar.gz", ".wo40.tar.gz"]:
            path = self.raw_dir / year / f"prepbufr.{key}{suffix}"
            if path.exists():
                return path
        return self.raw_dir / year / f"prepbufr.{key}.nr.tar.gz"

    def normalize(self, raw_path: Path, provenance: RunProvenance) -> pd.DataFrame:
        """Extract and normalize a single day's PrepBUFR tar archive."""
        from obsmet.sources.gdas_prepbufr.extract import extract_day

        date_str = raw_path.name.split(".")[1][:8]
        df = extract_day(raw_path, date_str)
        if df.empty:
            return pd.DataFrame()

        uri = f"gdas://{raw_path}"
        return normalize_to_canonical_wide(df, provenance, raw_source_uri=uri)
