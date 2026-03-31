"""ECCC hourly source adapter — Environment and Climate Change Canada.

Parses bulk-downloaded hourly CSV files from MSC Datamart.
File naming: climate_hourly_{PROV}_{CLIMATE_ID}_{YEAR}_P1H.csv
Encoding: Latin-1 (ISO-8859-1) — degree symbols in column headers.

Timestamps are Local Standard Time (LST); converted to UTC using a
province-derived offset. DST is never applied — ECCC uses LST year-round.

Key unit conversions:
  - Wind speed: km/h → m/s (÷ 3.6)
  - Wind direction: 10s of degrees → degrees (× 10)
  - Station pressure: kPa → Pa (× 1000)
  - Precip: mm (already correct, hourly interval not accumulated)

QC flags are single-character per variable. See _FLAG_MAP for mapping.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.sources.base import SourceAdapter

DEFAULT_RAW_DIR = "/nas/climate/eccc/hourly"

# Province → UTC offset (standard time, no DST).
# NT/NU vary by station longitude; handled in _utc_offset_for_station().
_PROVINCE_UTC_OFFSET: dict[str, float] = {
    "NL": -3.5,
    "NS": -4.0,
    "NB": -4.0,
    "PE": -4.0,
    "QC": -5.0,
    "ON": -5.0,  # default; western ON overridden by longitude
    "MB": -6.0,
    "SK": -6.0,
    "AB": -7.0,
    "BC": -8.0,
    "YT": -7.0,
    "NT": -7.0,  # default; varies by longitude
    "NU": -5.0,  # default; varies by longitude
}

# CSV column name → (canonical variable name, conversion function or None)
_COLUMN_MAP: dict[str, tuple[str, object]] = {
    "Temp (°C)": ("tair", None),
    "Dew Point Temp (°C)": ("td", None),
    "Rel Hum (%)": ("rh", None),
    "Precip. Amount (mm)": ("prcp", None),
    "Wind Dir (10s deg)": ("wind_dir", lambda x: x * 10.0),
    "Wind Spd (km/h)": ("wind", lambda x: x / 3.6),
    "Stn Press (kPa)": ("psfc", lambda x: x * 1000.0),
}

# Flag column suffix for each variable column
_FLAG_SUFFIX = " Flag"

# ECCC QC flag → obsmet qc_state
_FLAG_MAP: dict[str, str] = {
    "": "pass",
    "M": "fail",  # missing
    "N": "fail",  # temp missing, known > 0°C
    "Y": "fail",  # temp missing, known < 0°C
    "E": "suspect",  # estimated
    "A": "suspect",  # accumulated
    "C": "suspect",  # precip occurred, amount uncertain
    "L": "suspect",  # precip may or may not have occurred
    "B": "suspect",  # multiple occurrences and estimated
    "F": "suspect",  # accumulated and estimated
    "^": "suspect",  # incomplete
    "D": "suspect",  # subject to further QC
    "S": "pass",  # more than one occurrence
    "T": "pass",  # trace precip (< 0.2 mm) — value set to 0
    "†": "pass",  # not yet reviewed by NCA (provisional)
}


def _utc_offset_for_station(province: str, longitude: float | None) -> float:
    """Determine UTC offset from province, refined by longitude for ON/NT/NU."""
    offset = _PROVINCE_UTC_OFFSET.get(province, -6.0)

    if longitude is not None:
        if province == "ON" and longitude < -85.0:
            offset = -6.0  # western Ontario → Central
        elif province in ("NT", "NU"):
            if longitude < -102.0:
                offset = -7.0  # Mountain
            elif longitude < -85.0:
                offset = -6.0  # Central
            else:
                offset = -5.0  # Eastern

    return offset


def _parse_climate_id(filename: str) -> tuple[str, str]:
    """Extract (province, climate_id) from filename like climate_hourly_AB_3011240_1994_P1H.csv."""
    m = re.match(r"climate_hourly_([A-Z]{2})_(\w+)_\d{4}_P1H\.csv", filename)
    if m:
        return m.group(1), m.group(2)
    return "", filename.replace(".csv", "")


def normalize_hourly_csv(
    csv_path: Path,
    provenance: RunProvenance,
) -> pd.DataFrame:
    """Parse an ECCC hourly CSV into canonical hourly wide-form."""
    df = pd.read_csv(csv_path, encoding="latin-1", dtype=str, low_memory=False)

    if df.empty:
        return pd.DataFrame()

    # Parse timestamp
    time_col = "Date/Time (LST)"
    if time_col not in df.columns:
        return pd.DataFrame()

    province, climate_id = _parse_climate_id(csv_path.name)

    # Get coordinates for timezone refinement
    lon = None
    lat = None
    if "Longitude (x)" in df.columns:
        lon_vals = pd.to_numeric(df["Longitude (x)"], errors="coerce")
        lon = lon_vals.dropna().iloc[0] if lon_vals.notna().any() else None
    if "Latitude (y)" in df.columns:
        lat_vals = pd.to_numeric(df["Latitude (y)"], errors="coerce")
        lat = lat_vals.dropna().iloc[0] if lat_vals.notna().any() else None

    utc_offset = _utc_offset_for_station(province, lon)

    # Parse local time → UTC
    local_times = pd.to_datetime(df[time_col], errors="coerce")
    valid_mask = local_times.notna()
    if not valid_mask.any():
        return pd.DataFrame()

    df = df.loc[valid_mask].copy()
    local_times = local_times.loc[valid_mask]
    datetime_utc = local_times - pd.Timedelta(hours=utc_offset)
    datetime_utc = datetime_utc.dt.tz_localize("UTC")

    n = len(df)
    out = pd.DataFrame()
    out["datetime_utc"] = datetime_utc.values
    out["station_key"] = f"eccc:{climate_id}"
    out["source"] = "eccc"
    out["source_station_id"] = climate_id

    if lat is not None:
        out["lat"] = lat
    if lon is not None:
        out["lon"] = lon

    # Map variables with unit conversion
    qc_states = pd.Series("pass", index=out.index)
    qc_reasons: list[list[str]] = [[] for _ in range(n)]

    for csv_col, (canon_name, converter) in _COLUMN_MAP.items():
        if csv_col not in df.columns:
            continue

        vals = pd.to_numeric(df[csv_col], errors="coerce")
        flag_col = (
            csv_col.replace(" (°C)", "")
            .replace(" (%)", "")
            .replace(" (mm)", "")
            .replace(" (10s deg)", "")
            .replace(" (km/h)", "")
            .replace(" (km)", "")
            .replace(" (kPa)", "")
        )
        flag_col = flag_col + _FLAG_SUFFIX

        # Apply QC flags
        if flag_col in df.columns:
            flags = df[flag_col].fillna("").str.strip()
            for i, flag in enumerate(flags):
                if flag == "T" and canon_name == "prcp":
                    vals.iloc[i] = 0.0  # trace precip → 0
                mapped = _FLAG_MAP.get(flag, "suspect" if flag else "pass")
                if mapped == "fail":
                    vals.iloc[i] = np.nan
                if mapped != "pass":
                    qc_states.iloc[i] = max(
                        qc_states.iloc[i],
                        mapped,
                        key=lambda s: {"pass": 0, "suspect": 1, "fail": 2}[s],
                    )
                    qc_reasons[i].append(f"{canon_name}:{flag}")

        if converter is not None:
            vals = converter(vals)

        out[canon_name] = vals.values

    out["qc_state"] = qc_states.values
    out["qc_reason_codes"] = [",".join(r) for r in qc_reasons]
    out["ingest_run_id"] = provenance.run_id
    out["transform_version"] = provenance.transform_version
    out["raw_source_uri"] = str(csv_path)

    return out


class EcccAdapter(SourceAdapter):
    """ECCC hourly source adapter.

    Discovers per-station keys by grouping CSV files by Climate ID across
    all province subdirectories and years.
    """

    source_name = "eccc"

    def __init__(self, raw_dir: str | Path = DEFAULT_RAW_DIR, **_kwargs):
        self.raw_dir = Path(raw_dir)

    def discover_keys(self, start, end) -> list[str]:
        """List unique Climate IDs from hourly CSV filenames.

        Returns keys as '{PROVINCE}_{CLIMATE_ID}' (e.g., 'AB_3011240').
        Each key maps to all year-files for that station.
        """
        seen = set()
        keys = []
        for prov_dir in sorted(self.raw_dir.iterdir()):
            if not prov_dir.is_dir():
                continue
            for f in sorted(prov_dir.glob("climate_hourly_*.csv")):
                province, climate_id = _parse_climate_id(f.name)
                key = f"{province}_{climate_id}"
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        return keys

    def _station_files(self, key: str) -> list[Path]:
        """Find all year-files for a station key like 'AB_3011240'."""
        province, climate_id = key.split("_", 1)
        prov_dir = self.raw_dir / province
        if not prov_dir.exists():
            return []
        return sorted(prov_dir.glob(f"climate_hourly_{province}_{climate_id}_*_P1H.csv"))

    def fetch_raw(self, key: str, dest_dir: Path) -> Path:
        files = self._station_files(key)
        return files[0] if files else self.raw_dir / f"{key}.csv"

    def normalize(self, raw_path: Path, provenance: RunProvenance) -> pd.DataFrame:
        return normalize_hourly_csv(raw_path, provenance)

    def normalize_key(self, key: str, provenance: RunProvenance, **kwargs) -> pd.DataFrame | None:
        """Normalize all year-files for a station, concatenate into one DataFrame."""
        files = self._station_files(key)
        if not files:
            return None

        frames = []
        for f in files:
            df = normalize_hourly_csv(f, provenance)
            if not df.empty:
                frames.append(df)

        if not frames:
            return None

        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values("datetime_utc").drop_duplicates(
            subset=["datetime_utc"], keep="first"
        )
        return result.reset_index(drop=True)

    def output_filename(self, key: str) -> str:
        return f"{key}.parquet"
