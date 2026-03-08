"""GHCN-Daily source adapter — global daily precipitation and temperature.

Parses NCEI daily-summaries CSV files (one per station). The 2022 snapshot
at /nas/climate/ghcn/ghcn_daily_summaries_4FEB2022/ uses quoted CSV with
paired VALUE + VALUE_ATTRIBUTES columns. Values are in tenths (tenths of
degrees C for TMAX/TMIN, tenths of mm for PRCP, mm for SNOW/SNWD).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.sources.base import SourceAdapter

DEFAULT_RAW_DIR = "/nas/climate/ghcn/ghcn_daily_summaries_4FEB2022"

# GHCN-Daily element → (canonical name, scale factor, unit)
# Values in CSV are integers in tenths or mm
ELEMENT_MAP = {
    "TMAX": ("tmax", 0.1, "degC"),  # tenths of degC → degC
    "TMIN": ("tmin", 0.1, "degC"),
    "TAVG": ("tmean", 0.1, "degC"),
    "PRCP": ("prcp", 0.1, "mm"),  # tenths of mm → mm
    "SNOW": ("snow", 1.0, "mm"),  # mm
    "SNWD": ("snow_depth", 1.0, "mm"),  # mm
    "AWND": ("wind", 0.1, "m s-1"),  # tenths of m/s → m/s
    "WESD": ("swe", 0.1, "mm"),  # tenths of mm → mm
}

# Quality flag values that indicate bad data
_BAD_QC_FLAGS = frozenset(
    {
        "D",  # duplicate
        "I",  # internal consistency
        "K",  # streak/frequent value
        "L",  # multiday accumulation
        "N",  # climatological outlier
        "O",  # gap in record
        "R",  # lagged range
        "S",  # spatial consistency
        "T",  # temporal consistency
        "W",  # temperature too warm for snow
        "X",  # failed bounds
    }
)


def normalize_station_csv(
    csv_path: Path,
    provenance: RunProvenance,
) -> pd.DataFrame:
    """Parse a GHCN-Daily per-station CSV into canonical daily wide-form."""
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    if df.empty:
        return pd.DataFrame()

    station_id = df["STATION"].iloc[0].strip('"')
    lat = float(df["LATITUDE"].iloc[0])
    lon = float(df["LONGITUDE"].iloc[0])
    elev_raw = df["ELEVATION"].iloc[0]
    try:
        elev = (
            float(str(elev_raw).strip('"'))
            if pd.notna(elev_raw) and str(elev_raw).strip('"')
            else None
        )
    except (ValueError, TypeError):
        elev = None

    out = pd.DataFrame()
    out["station_key"] = "ghcnd:" + station_id
    out["source"] = "ghcnd"
    out["source_station_id"] = station_id
    out["date"] = pd.to_datetime(df["DATE"])
    out["day_basis"] = "local"
    out["lat"] = lat
    out["lon"] = lon
    if elev is not None:
        out["elev_m"] = elev

    qc_states = ["pass"] * len(df)
    qc_reasons = [""] * len(df)

    for element, (canon_name, scale, _unit) in ELEMENT_MAP.items():
        if element not in df.columns:
            continue

        vals = pd.to_numeric(df[element], errors="coerce")
        out[canon_name] = vals * scale

        # Check quality flags from the ATTRIBUTES column
        attr_col = f"{element}_ATTRIBUTES"
        if attr_col in df.columns:
            for i, attr in enumerate(df[attr_col]):
                if pd.isna(attr) or not isinstance(attr, str):
                    continue
                parts = attr.split(",")
                # ATTRIBUTES format: measurement_flag, quality_flag, source_flag[, obs_time]
                if len(parts) >= 2:
                    qflag = parts[1].strip()
                    if qflag in _BAD_QC_FLAGS:
                        qc_states[i] = "fail"
                        reason = f"{canon_name}:qflag_{qflag}"
                        if qc_reasons[i]:
                            qc_reasons[i] += f",{reason}"
                        else:
                            qc_reasons[i] = reason

    out["qc_state"] = qc_states
    out["qc_reason_codes"] = qc_reasons
    out["obs_count"] = 1
    out["ingest_run_id"] = provenance.run_id
    out["transform_version"] = provenance.transform_version
    out["raw_source_uri"] = str(csv_path)

    return out


class GhcndAdapter(SourceAdapter):
    """GHCN-Daily source adapter."""

    source_name = "ghcnd"

    def __init__(self, raw_dir: str | Path = DEFAULT_RAW_DIR, **_kwargs):
        self.raw_dir = Path(raw_dir)

    def discover_keys(self, start, end) -> list[str]:
        """List station IDs from CSV files in raw_dir."""
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
