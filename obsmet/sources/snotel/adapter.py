"""SNOTEL source adapter — NRCS high-elevation snow and met stations.

Supports two raw data formats:
  1. Legacy daily CSV: /nas/climate/snotel/snotel_records/{id}_{name}_{state}.csv
     Daily resolution, local day, already metric.
  2. Hourly AWDB parquet: /nas/climate/snotel/hourly/{id}_{state}_SNTL.parquet
     Hourly resolution with datetime_utc, from download.py. Already metric.

The adapter auto-detects the format based on file extension in raw_dir.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.sources.base import SourceAdapter

DEFAULT_RAW_DIR = "/nas/climate/snotel/snotel_records"
HOURLY_RAW_DIR = "/nas/climate/snotel/hourly"

# CSV column → (canonical name, unit)
COLUMN_MAP = {
    "swe": ("swe", "mm"),
    "tmin": ("tmin", "degC"),
    "tmax": ("tmax", "degC"),
    "tavg": ("tmean", "degC"),
    "prec": ("prcp", "mm"),  # cumulative precip → will need differencing
    "rh": ("rh", "%"),
    "ws": ("wind", "m s-1"),
}


def _parse_station_id(filename: str) -> str:
    """Extract site ID from filename like '100_Bear_Mountain_ID.csv'."""
    match = re.match(r"^(\d+)_", filename)
    return match.group(1) if match else filename.replace(".csv", "")


def normalize_station_csv(
    csv_path: Path,
    provenance: RunProvenance,
) -> pd.DataFrame:
    """Parse a SNOTEL per-station CSV into canonical daily wide-form."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if df.empty:
        return pd.DataFrame()

    # Drop rows where all met values are NaN
    met_cols = [c for c in df.columns if c in COLUMN_MAP]
    df = df.dropna(subset=met_cols, how="all")
    if df.empty:
        return pd.DataFrame()

    station_id = _parse_station_id(csv_path.name)

    n = len(df)
    out = pd.DataFrame()
    out["date"] = df.index
    out["station_key"] = [f"snotel:{station_id}"] * n
    out["source"] = "snotel"
    out["source_station_id"] = station_id
    out["day_basis"] = "local"

    for csv_col, (canon_name, _unit) in COLUMN_MAP.items():
        if csv_col in df.columns:
            vals = pd.to_numeric(df[csv_col], errors="coerce")
            if csv_col == "prec":
                # SNOTEL precip is cumulative — difference to get daily
                daily_prcp = vals.diff()
                # First day and resets (negative diffs) get NaN
                daily_prcp[daily_prcp < 0] = float("nan")
                out[canon_name] = daily_prcp.values
            else:
                out[canon_name] = vals.values

    out["qc_state"] = "pass"
    out["obs_count"] = 1
    out["ingest_run_id"] = provenance.run_id
    out["transform_version"] = provenance.transform_version
    out["raw_source_uri"] = str(csv_path)

    out = out.reset_index(drop=True)
    return out


# AWDB element code → (canonical name, unit)
_HOURLY_COLUMN_MAP = {
    "WTEQ": ("swe", "mm"),
    "SNWD": ("snow_depth", "mm"),
    "PREC": ("prcp", "mm"),  # accumulated — needs differencing
    "TOBS": ("tair", "degC"),
}


def normalize_station_parquet(
    parquet_path: Path,
    provenance: RunProvenance,
) -> pd.DataFrame:
    """Parse an AWDB hourly per-station parquet into canonical hourly wide-form.

    Input columns: datetime_utc, datetime_local, raw_tz_offset,
                   WTEQ (mm), SNWD (mm), PREC (mm, accumulated), TOBS (degC),
                   station_triplet, station_name, lat, lon, elev_ft
    """
    df = pd.read_parquet(parquet_path)
    if df.empty or "datetime_utc" not in df.columns:
        return pd.DataFrame()

    triplet = (
        df["station_triplet"].iloc[0] if "station_triplet" in df.columns else parquet_path.stem
    )
    station_id = triplet.split(":")[0] if ":" in str(triplet) else parquet_path.stem

    out = pd.DataFrame()
    out["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    out["station_key"] = f"snotel:{station_id}"
    out["source"] = "snotel"
    out["source_station_id"] = station_id

    for awdb_col, (canon_name, _unit) in _HOURLY_COLUMN_MAP.items():
        if awdb_col not in df.columns:
            continue
        vals = pd.to_numeric(df[awdb_col], errors="coerce")
        if awdb_col == "PREC":
            # Accumulated precip → difference to get hourly increment
            hourly_prcp = vals.diff()
            hourly_prcp[hourly_prcp < 0] = np.nan  # resets → NaN
            hourly_prcp.iloc[0] = np.nan  # first value has no predecessor
            out[canon_name] = hourly_prcp.values
        else:
            out[canon_name] = vals.values

    # Carry through station metadata
    for col in ("lat", "lon"):
        if col in df.columns:
            out[col] = df[col].values
    if "elev_ft" in df.columns:
        out["elev_m"] = pd.to_numeric(df["elev_ft"], errors="coerce").values * 0.3048

    out["qc_state"] = "pass"
    out["qc_reason_codes"] = ""
    out["ingest_run_id"] = provenance.run_id
    out["transform_version"] = provenance.transform_version
    out["raw_source_uri"] = str(parquet_path)

    return out


class SnotelAdapter(SourceAdapter):
    """SNOTEL source adapter.

    Auto-detects raw data format:
      - *.csv in raw_dir → legacy daily CSV path
      - *.parquet in raw_dir → hourly AWDB parquet path
    """

    source_name = "snotel"

    def __init__(self, raw_dir: str | Path = DEFAULT_RAW_DIR, **_kwargs):
        self.raw_dir = Path(raw_dir)
        self._is_hourly = any(self.raw_dir.glob("*.parquet"))

    def discover_keys(self, start, end) -> list[str]:
        if self._is_hourly:
            keys = []
            for f in sorted(self.raw_dir.glob("*.parquet")):
                if f.name == "station_inventory.parquet":
                    continue
                keys.append(f.stem)
            return keys
        # Legacy CSV
        return [f.stem for f in sorted(self.raw_dir.glob("*.csv"))]

    def fetch_raw(self, key: str, dest_dir: Path) -> Path:
        if self._is_hourly:
            return self.raw_dir / f"{key}.parquet"
        return self.raw_dir / f"{key}.csv"

    def normalize(self, raw_path: Path, provenance: RunProvenance) -> pd.DataFrame:
        if raw_path.suffix == ".parquet":
            return normalize_station_parquet(raw_path, provenance)
        return normalize_station_csv(raw_path, provenance)

    def normalize_key(self, key: str, provenance: RunProvenance, **kwargs) -> pd.DataFrame | None:
        if self._is_hourly:
            pq_path = self.raw_dir / f"{key}.parquet"
            if not pq_path.exists():
                return None
            df = normalize_station_parquet(pq_path, provenance)
        else:
            csv_path = self.raw_dir / f"{key}.csv"
            if not csv_path.exists():
                return None
            df = normalize_station_csv(csv_path, provenance)
        return df if not df.empty else None

    def output_filename(self, key: str) -> str:
        return f"{key}.parquet"
