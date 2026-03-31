"""GHCNh source adapter — global hourly observations (replaces ISD).

Parses PSV (pipe-separated) station files from NCEI GHCNh v1beta.
One file per station containing full period-of-record.
Units: degC, m/s, degrees, hPa, mm — no scaling needed.
Per-observation Quality_Code flags for QC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.sources.base import SourceAdapter

DEFAULT_RAW_DIR = "/nas/climate/ghcnh"

# GHCNh variable columns → (canonical name, unit)
VARIABLE_MAP = {
    "temperature": ("tair", "degC"),
    "dew_point_temperature": ("td", "degC"),
    "wind_speed": ("wind", "m s-1"),
    "wind_direction": ("wind_dir", "deg"),
    "wind_gust": ("wind_gust", "m s-1"),
    "sea_level_pressure": ("slp", "hPa"),
    "station_level_pressure": ("psfc", "hPa"),
    "precipitation": ("prcp", "mm"),
    "relative_humidity": ("rh", "%"),
    "wet_bulb_temperature": ("tw", "degC"),
}

# Quality_Code values:
# 1=passed all QC, 2=not checked, 4=passed after correction,
# 5=translated (no QC applied), A=adjusted
# Codes that indicate bad data:
# 3=failed one test, 6=failed multiple tests, 7=suspect/erroneous
_BAD_QC_CODES = frozenset({"3", "6", "7"})

# Columns to read from the PSV (skip the many source/report_type columns)
_USE_COLS = [
    "Station_ID",
    "Station_name",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Latitude",
    "Longitude",
    "Elevation",
    "temperature",
    "temperature_Quality_Code",
    "dew_point_temperature",
    "dew_point_temperature_Quality_Code",
    "wind_speed",
    "wind_speed_Quality_Code",
    "wind_direction",
    "wind_direction_Quality_Code",
    "wind_gust",
    "wind_gust_Quality_Code",
    "sea_level_pressure",
    "sea_level_pressure_Quality_Code",
    "station_level_pressure",
    "station_level_pressure_Quality_Code",
    "precipitation",
    "precipitation_Quality_Code",
    "relative_humidity",
    "relative_humidity_Quality_Code",
    "wet_bulb_temperature",
    "wet_bulb_temperature_Quality_Code",
]


def normalize_station_psv(
    psv_path: Path,
    provenance: RunProvenance,
) -> pd.DataFrame:
    """Parse a GHCNh per-station PSV into canonical hourly wide-form."""
    # Read only the columns we need (PSV files can have 190+ columns)
    try:
        all_cols = pd.read_csv(psv_path, sep="|", nrows=0).columns.tolist()
    except Exception:
        return pd.DataFrame()

    use = [c for c in _USE_COLS if c in all_cols]
    df = pd.read_csv(psv_path, sep="|", usecols=use, dtype=str, low_memory=False)

    if df.empty:
        return pd.DataFrame()

    station_id = df["Station_ID"].iloc[0].strip()

    # Build datetime
    df["datetime_utc"] = pd.to_datetime(
        df["Year"]
        + "-"
        + df["Month"].str.zfill(2)
        + "-"
        + df["Day"].str.zfill(2)
        + " "
        + df["Hour"].str.zfill(2)
        + ":"
        + df["Minute"].str.zfill(2),
        errors="coerce",
        utc=True,
    )
    df = df.dropna(subset=["datetime_utc"])

    n = len(df)
    out = pd.DataFrame()
    out["datetime_utc"] = df["datetime_utc"].values
    out["station_key"] = ["ghcnh:" + station_id] * n
    out["source"] = "ghcnh"
    out["source_station_id"] = station_id
    out["lat"] = pd.to_numeric(df["Latitude"], errors="coerce").values
    out["lon"] = pd.to_numeric(df["Longitude"], errors="coerce").values
    out["elev_m"] = pd.to_numeric(df["Elevation"], errors="coerce").values

    # Map variables and check QC — per-variable state columns
    _STATE_RANK = {"pass": 0, "suspect": 1, "fail": 2}
    row_states = ["pass"] * n
    row_reasons: list[list[str]] = [[] for _ in range(n)]

    for ghcnh_var, (canon_name, _unit) in VARIABLE_MAP.items():
        if ghcnh_var not in df.columns:
            continue
        vals = pd.to_numeric(df[ghcnh_var], errors="coerce")
        var_states = ["pass"] * n
        var_reasons: list[list[str]] = [[] for _ in range(n)]

        # Check quality codes
        qc_col = f"{ghcnh_var}_Quality_Code"
        if qc_col in df.columns:
            for i, qc_val in enumerate(df[qc_col]):
                if pd.notna(qc_val) and str(qc_val).strip() in _BAD_QC_CODES:
                    vals.iloc[i] = np.nan  # null the bad variable value
                    var_states[i] = "fail"
                    var_reasons[i].append(f"{canon_name}:qc_{qc_val}")
                    # Update row-level summary
                    if _STATE_RANK["fail"] > _STATE_RANK[row_states[i]]:
                        row_states[i] = "fail"
                    row_reasons[i].append(f"{canon_name}:qc_{qc_val}")

        out[canon_name] = vals.values
        out[f"{canon_name}_qc_state"] = var_states
        out[f"{canon_name}_qc_reason_codes"] = [",".join(r) if r else "" for r in var_reasons]

    out["qc_state"] = row_states
    out["qc_reason_codes"] = [",".join(r) if r else "" for r in row_reasons]
    out["ingest_run_id"] = provenance.run_id
    out["transform_version"] = provenance.transform_version
    out["raw_source_uri"] = str(psv_path)

    return out


class GhcnhAdapter(SourceAdapter):
    """GHCNh source adapter — replaces ISD."""

    source_name = "ghcnh"

    def __init__(self, raw_dir: str | Path = DEFAULT_RAW_DIR, **_kwargs):
        self.raw_dir = Path(raw_dir)

    def discover_keys(self, start, end) -> list[str]:
        """List station PSV files in raw_dir.

        Keys are station IDs derived from filenames (e.g. 'USW00024153').
        Start/end are ignored since each file contains the full POR.
        """
        keys = []
        for f in sorted(self.raw_dir.glob("*.psv")):
            keys.append(f.stem)
        return keys

    def fetch_raw(self, key: str, dest_dir: Path) -> Path:
        return self.raw_dir / f"{key}.psv"

    def normalize(self, raw_path: Path, provenance: RunProvenance) -> pd.DataFrame:
        return normalize_station_psv(raw_path, provenance)

    def normalize_key(self, key: str, provenance: RunProvenance, **kwargs) -> pd.DataFrame | None:
        psv_path = self.raw_dir / f"{key}.psv"
        if not psv_path.exists():
            return None
        df = normalize_station_psv(psv_path, provenance)
        return df if not df.empty else None

    def output_filename(self, key: str) -> str:
        return f"{key}.parquet"
