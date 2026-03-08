"""SNOTEL source adapter — NRCS high-elevation snow and met stations.

Parses pre-downloaded per-station CSV files from /nas/climate/snotel/snotel_records/.
File naming: {site_id}_{name}_{state}.csv
Columns: (unnamed date index), swe, tmin, tmax, tavg, prec, rn, rh, ws
Units are already metric (degC, mm) — converted during the original NRCS download.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.sources.base import SourceAdapter

DEFAULT_RAW_DIR = "/nas/climate/snotel/snotel_records"

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


class SnotelAdapter(SourceAdapter):
    """SNOTEL source adapter."""

    source_name = "snotel"

    def __init__(self, raw_dir: str | Path = DEFAULT_RAW_DIR, **_kwargs):
        self.raw_dir = Path(raw_dir)

    def discover_keys(self, start, end) -> list[str]:
        """List station CSV files available in raw_dir.

        Keys are filenames without extension (e.g. '100_Bear_Mountain_ID').
        """
        keys = []
        for f in sorted(self.raw_dir.glob("*.csv")):
            keys.append(f.stem)
        return keys

    def fetch_raw(self, key: str, dest_dir: Path) -> Path:
        return self.raw_dir / f"{key}.csv"

    def normalize(self, raw_path: Path, provenance: RunProvenance) -> pd.DataFrame:
        return normalize_station_csv(raw_path, provenance)

    def normalize_key(self, key: str, provenance: RunProvenance, **kwargs) -> pd.DataFrame | None:
        csv_path = self.raw_dir / f"{key}.csv"
        if not csv_path.exists():
            return None
        df = normalize_station_csv(csv_path, provenance)
        return df if not df.empty else None

    def output_filename(self, key: str) -> str:
        return f"{key}.parquet"
