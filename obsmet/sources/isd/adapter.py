"""ISD source adapter — DEPRECATED, use GHCNh instead.

ISD is superseded by GHCNh (Global Historical Climatology Network - Hourly),
which subsumes all ISD stations plus additional sources with aligned GHCN IDs.
This adapter is retained for backward compatibility with existing normalized
data but should not be used for new normalization runs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.sources.base import SourceAdapter
from obsmet.sources.isd.extract import apply_qc_mask, read_isd_file

warnings.warn(
    "ISD adapter is deprecated — use GHCNh (obsmet.sources.ghcnh) instead. "
    "ISD will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# --------------------------------------------------------------------------- #
# ISD → canonical variable mapping
# --------------------------------------------------------------------------- #

# ISD parsed names → obsmet canonical variable names
VARIABLE_MAP = {
    "air_temperature": "tair",
    "dew_point_temperature": "td",
    "wind_speed": "wind",
    "wind_direction": "wind_dir",
    "sea_level_pressure": "slp",
}

# ISD units are already scaled during parsing (÷10), so no further conversion
# needed. All values arrive as: °C, m/s, hPa, degrees.
UNIT_MAP = {
    "air_temperature": "degC",
    "dew_point_temperature": "degC",
    "wind_speed": "m s-1",
    "wind_direction": "deg",
    "sea_level_pressure": "Pa",
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
    """Convert ISD extraction DataFrame to canonical wide-form.

    Returns a DataFrame with columns: station_key, source, source_station_id,
    datetime_utc, lat, lon, elev_m, tair, td, wind, wind_dir, slp,
    plus per-variable QC columns and provenance fields.
    """
    out = pd.DataFrame()

    out["station_key"] = "isd:" + df["station_id"].astype(str)
    out["source"] = "isd"
    out["source_station_id"] = df["station_id"].astype(str)
    out["datetime_utc"] = df["datetime_utc"]
    out["lat"] = df["latitude"].values
    out["lon"] = df["longitude"].values
    out["elev_m"] = df["elevation"].values

    # Map variable names; pressure needs hPa → Pa conversion
    for isd_var, canon_var in VARIABLE_MAP.items():
        if isd_var in df.columns:
            vals = pd.to_numeric(df[isd_var], errors="coerce").values
            if isd_var == "sea_level_pressure":
                vals = vals * 100.0  # hPa → Pa
            out[canon_var] = vals

    # Carry through QC columns
    qc_cols = [c for c in df.columns if c.endswith("_qc")]
    for col in qc_cols:
        out[col] = df[col].values

    out["ingest_run_id"] = provenance.run_id
    out["transform_version"] = provenance.transform_version

    return out


def extract_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique station metadata from an ISD DataFrame."""
    id_col = "source_station_id" if "source_station_id" in df.columns else "station_id"
    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lon" if "lon" in df.columns else "longitude"
    elev_col = "elev_m" if "elev_m" in df.columns else "elevation"

    cols = [c for c in [id_col, lat_col, lon_col, elev_col] if c in df.columns]
    meta = df[cols].drop_duplicates(subset=[id_col]).copy()

    meta = meta.rename(
        columns={id_col: "source_station_id", lat_col: "lat", lon_col: "lon", elev_col: "elev_m"}
    )
    meta["source"] = "isd"
    meta["station_key"] = "isd:" + meta["source_station_id"].astype(str)

    return meta.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SourceAdapter implementation
# --------------------------------------------------------------------------- #


class IsdAdapter(SourceAdapter):
    """ISD source adapter."""

    source_name = "isd"

    def __init__(
        self,
        raw_dir: str | Path = "/nas/climate/isd/raw",
        apply_qc: bool = True,
    ):
        self.raw_dir = Path(raw_dir)
        self.apply_qc = apply_qc

    def discover_keys(self, start, end) -> list[str]:
        """List station-year file paths available in [start, end].

        Keys are relative paths like '2024/720538-00164-2024.gz'.
        """
        keys = []
        for year in range(start.year, end.year + 1):
            year_dir = self.raw_dir / str(year)
            if not year_dir.is_dir():
                continue
            for gz in sorted(year_dir.glob("*.gz")):
                keys.append(f"{year}/{gz.name}")
        return keys

    def fetch_raw(self, key: str, dest_dir: Path) -> Path:
        """Return path to raw file for a given key."""
        return self.raw_dir / key

    def normalize(self, raw_path: Path, provenance: RunProvenance) -> pd.DataFrame:
        """Parse a single ISD .gz file into canonical wide-form DataFrame."""
        df = read_isd_file(raw_path)
        if df.empty:
            return pd.DataFrame()

        if self.apply_qc:
            df = apply_qc_mask(df)

        uri = f"isd://{raw_path}"
        return normalize_to_canonical_wide(df, provenance, raw_source_uri=uri)

    def output_filename(self, key: str) -> str:
        """Derive parquet filename from ISD key like '2024/720538-00164-2024.gz'."""
        return key.replace("/", "_").replace(".gz", ".parquet")

    def normalize_file(
        self,
        path: Path | str,
        provenance: RunProvenance,
    ) -> pd.DataFrame | None:
        """Convenience: read, QC-mask, and normalize a single ISD file.

        Returns normalized DataFrame or None if no data.
        """
        path = Path(path)
        df = read_isd_file(path)
        if df.empty:
            return None

        if self.apply_qc:
            df = apply_qc_mask(df)

        uri = f"isd://{path}"
        return normalize_to_canonical_wide(df, provenance, raw_source_uri=uri)
