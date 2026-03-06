"""MADIS source adapter — maps MADIS-native data to obsmet canonical schema.

Implements the SourceAdapter interface for MADIS mesonet observations.
Handles raw file discovery, download delegation, and normalization from
MADIS-native units/names to the obsmet canonical hourly observation schema.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.sources.base import SourceAdapter
from obsmet.sources.madis.extract import (
    QCR_REJECT_BITS,
    extract_day,
)

# --------------------------------------------------------------------------- #
# MADIS → canonical variable mapping
# --------------------------------------------------------------------------- #

# MADIS native names → obsmet canonical variable names
VARIABLE_MAP = {
    "temperature": "tair",
    "dewpoint": "td",
    "relHumidity": "rh",
    "windSpeed": "wind",
    "windDir": "wind_dir",
    "precipAccum": "prcp",
    "solarRadiation": "rsds_hourly",
}

# MADIS native units → obsmet canonical units
UNIT_MAP = {
    "temperature": ("K", "degC"),
    "dewpoint": ("K", "degC"),
    "relHumidity": ("percent", "percent"),
    "windSpeed": ("m s-1", "m s-1"),
    "windDir": ("deg", "deg"),
    "precipAccum": ("mm", "mm"),
    "solarRadiation": ("W m-2", "W m-2"),
}


def _kelvin_to_celsius(s: pd.Series) -> pd.Series:
    return s - 273.15


# Conversion functions keyed by MADIS variable name
CONVERTERS = {
    "temperature": _kelvin_to_celsius,
    "dewpoint": _kelvin_to_celsius,
}


# --------------------------------------------------------------------------- #
# Normalization to canonical long-form
# --------------------------------------------------------------------------- #


def normalize_to_canonical(
    df: pd.DataFrame,
    provenance: RunProvenance,
    *,
    raw_source_uri: str = "",
) -> pd.DataFrame:
    """Convert a MADIS-native extraction DataFrame to canonical hourly obs rows.

    Transforms wide-form MADIS DataFrame (one column per variable) into
    long-form canonical schema (one row per station-time-variable).

    Parameters
    ----------
    df : DataFrame from extract.extract_day() with MADIS-native columns.
    provenance : RunProvenance for this pipeline run.
    raw_source_uri : URI of the raw source file(s).

    Returns
    -------
    DataFrame conforming to OBS_HOURLY_SCHEMA columns.
    """
    records = []

    met_vars = list(VARIABLE_MAP.keys())
    dd_cols = {v: f"{v}DD" for v in met_vars if f"{v}DD" in df.columns}
    qcr_cols = {v: f"{v}QCR" for v in met_vars if f"{v}QCR" in df.columns}

    for _, row in df.iterrows():
        station_id = str(row.get("stationId", "")).strip()
        if not station_id:
            continue

        obs_time = row.get("observationTime")
        if pd.isna(obs_time):
            continue

        # Build station_key as source:id
        station_key = f"madis:{station_id}"

        for madis_var, canon_var in VARIABLE_MAP.items():
            value = row.get(madis_var)
            if pd.isna(value):
                continue

            # Apply unit conversion
            converter = CONVERTERS.get(madis_var)
            if converter is not None:
                value = float(converter(pd.Series([value])).iloc[0])
            else:
                value = float(value)

            _, canon_unit = UNIT_MAP[madis_var]

            # Collect native QC flags
            qc_native = {}
            if madis_var in dd_cols:
                dd_val = row.get(dd_cols[madis_var])
                if pd.notna(dd_val):
                    qc_native["dd"] = str(dd_val)
            if madis_var in qcr_cols:
                qcr_val = row.get(qcr_cols[madis_var])
                if pd.notna(qcr_val):
                    qc_native["qcr"] = int(qcr_val)

            # qc_state from the extraction layer
            qc_state = "pass" if row.get("qc_passed", False) else "fail"

            records.append(
                {
                    "station_key": station_key,
                    "source": "madis",
                    "source_station_id": station_id,
                    "datetime_utc": pd.Timestamp(obs_time, tz="UTC"),
                    "variable": canon_var,
                    "value": value,
                    "unit": canon_unit,
                    "qc_state": qc_state,
                    "qc_flags_native": json.dumps(qc_native) if qc_native else "",
                    "qc_rules_version": provenance.qaqc_rules_version,
                    "transform_version": provenance.transform_version,
                    "raw_source_uri": raw_source_uri,
                    "raw_file_hash": "",
                    "ingest_run_id": provenance.run_id,
                }
            )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def normalize_to_canonical_wide(
    df: pd.DataFrame,
    provenance: RunProvenance,
    *,
    raw_source_uri: str = "",
) -> pd.DataFrame:
    """Convert MADIS extraction to canonical form, keeping wide format.

    This is a faster alternative to normalize_to_canonical() that preserves
    the wide-form layout (one column per variable) while applying unit
    conversions and renaming to canonical variable names.  Useful for
    downstream daily aggregation that expects wide-form input.

    Returns a DataFrame with columns: station_key, source, source_station_id,
    datetime_utc, tair, td, rh, wind, wind_dir, prcp, rsds_hourly,
    plus DD/QCR columns and qc_passed.
    """
    out = pd.DataFrame()

    out["station_key"] = "madis:" + df["stationId"].astype(str).str.strip()
    out["source"] = "madis"
    out["source_station_id"] = df["stationId"].astype(str).str.strip()
    out["datetime_utc"] = pd.to_datetime(df["observationTime"], utc=True)
    out["lat"] = df["latitude"].values
    out["lon"] = df["longitude"].values
    out["elev_m"] = df["elevation"].values
    out["provider"] = df["dataProvider"].astype(str).str.strip()

    # Convert and rename met variables
    for madis_var, canon_var in VARIABLE_MAP.items():
        if madis_var in df.columns:
            values = pd.to_numeric(df[madis_var], errors="coerce")
            converter = CONVERTERS.get(madis_var)
            if converter is not None:
                values = converter(values)
            out[canon_var] = values.values

    # Carry through QC columns
    for col in df.columns:
        if col.endswith("DD") or col.endswith("QCR"):
            out[col] = df[col].values

    if "qc_passed" in df.columns:
        out["qc_passed"] = df["qc_passed"].values

    out["ingest_run_id"] = provenance.run_id
    out["transform_version"] = provenance.transform_version

    return out


# --------------------------------------------------------------------------- #
# Station metadata extraction
# --------------------------------------------------------------------------- #


def extract_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique station metadata from a MADIS extraction DataFrame.

    Parameters
    ----------
    df : Wide-form MADIS DataFrame (from extract_day or normalize_to_canonical_wide).

    Returns
    -------
    DataFrame with one row per station: station_key, source, source_station_id,
    lat, lon, elev_m, provider.
    """
    id_col = "source_station_id" if "source_station_id" in df.columns else "stationId"
    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lon" if "lon" in df.columns else "longitude"
    elev_col = "elev_m" if "elev_m" in df.columns else "elevation"
    prov_col = "provider" if "provider" in df.columns else "dataProvider"

    cols = [id_col, lat_col, lon_col, elev_col, prov_col]
    cols = [c for c in cols if c in df.columns]
    meta = df[cols].drop_duplicates(subset=[id_col]).copy()

    meta = meta.rename(
        columns={
            id_col: "source_station_id",
            lat_col: "lat",
            lon_col: "lon",
            elev_col: "elev_m",
            prov_col: "provider",
        }
    )
    meta["source"] = "madis"
    meta["station_key"] = "madis:" + meta["source_station_id"].astype(str)

    return meta.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SourceAdapter implementation
# --------------------------------------------------------------------------- #


class MadisAdapter(SourceAdapter):
    """MADIS mesonet source adapter."""

    source_name = "madis"

    def __init__(
        self,
        raw_dir: str | Path = "/nas/climate/madis/LDAD/mesonet/netCDF",
        bounds: tuple[float, float, float, float] | None = None,
        qcr_mask: int = QCR_REJECT_BITS,
    ):
        self.raw_dir = Path(raw_dir)
        self.bounds = bounds
        self.qcr_mask = qcr_mask

    def discover_keys(self, start, end) -> list[str]:
        """List YYYYMMDD day strings in [start, end]."""
        keys = []
        current = start
        while current <= end:
            keys.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return keys

    def fetch_raw(self, key: str, dest_dir: Path) -> Path:
        """Return path to raw directory for a given day key.

        Actual download is handled separately by download.download_day().
        This method just resolves the expected path.
        """
        return self.raw_dir

    def normalize(self, raw_path: Path, provenance: RunProvenance) -> pd.DataFrame:
        """Extract and normalize a single day.

        raw_path should be the directory containing .gz files.
        The day is determined from the provenance or must be passed
        via the key parameter in the pipeline.
        """
        raise NotImplementedError("Use normalize_key() with an explicit day_str instead.")

    def normalize_key(self, key: str, provenance: RunProvenance, **kwargs) -> pd.DataFrame | None:
        """Normalize a single MADIS day key via extract_and_normalize_day."""
        return self.extract_and_normalize_day(key, provenance, wide=True)

    def extract_and_normalize_day(
        self,
        day_str: str,
        provenance: RunProvenance,
        *,
        wide: bool = True,
    ) -> pd.DataFrame | None:
        """Full pipeline for one day: extract → QC → normalize.

        Parameters
        ----------
        day_str : YYYYMMDD date string.
        provenance : RunProvenance for this run.
        wide : If True, return wide-form canonical DataFrame.
               If False, return long-form (one row per observation-variable).

        Returns
        -------
        Normalized DataFrame or None if no data for the day.
        """
        df = extract_day(
            day_str,
            self.raw_dir,
            bounds=self.bounds,
            qcr_mask=self.qcr_mask,
        )
        if df is None or df.empty:
            return None

        uri = f"madis://{self.raw_dir}/{day_str}_*.gz"
        if wide:
            return normalize_to_canonical_wide(df, provenance, raw_source_uri=uri)
        else:
            return normalize_to_canonical(df, provenance, raw_source_uri=uri)
