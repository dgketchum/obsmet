"""MADIS netCDF extraction and three-layer QC.

Ported from dads-mvp/process/obs/extract_madis_daily.py.

Reads gzip-compressed hourly netCDF files, extracts station/met/QC variables,
applies three-layer QC (DD flags, QCR bitmask, physical bounds), and returns
DataFrames in MADIS-native schema (Kelvin temps, native variable names).

The adapter module handles conversion to obsmet canonical schema.
"""

from __future__ import annotations

import gzip
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=xr.SerializationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")

# --------------------------------------------------------------------------- #
# QC constants
# --------------------------------------------------------------------------- #

ACCEPTABLE_DD = frozenset({"V", "S", "C", "G"})

# Bits: master(1) | validity(2) | temporal(16) | stat_spatial(32) | buddy(64)
# Excludes bit 4 (internal consistency = 8) — Td>T handled separately.
QCR_REJECT_BITS = 0b01110011  # = 115

BOUNDS = {
    "temperature": (223.15, 333.15),  # -50 to 60 °C in K
    "dewpoint": (205.15, 305.15),  # -68 to 32 °C in K
    "relHumidity": (2.0, 110.0),  # %
    "windSpeed": (0.0, 35.0),  # m/s
    "windDir": (0.0, 360.0),  # degrees
}

# Variables to extract from each netCDF file
ID_VARS = [
    "stationId",
    "latitude",
    "longitude",
    "elevation",
    "dataProvider",
    "observationTime",
]

MET_VARS = [
    "temperature",
    "dewpoint",
    "relHumidity",
    "windSpeed",
    "windDir",
    "precipAccum",
    "solarRadiation",
]

MET_WITH_DD = ["temperature", "dewpoint", "relHumidity", "windSpeed", "windDir"]

DD_VARS = [f"{v}DD" for v in MET_WITH_DD]
QCR_VARS = [f"{v}QCR" for v in MET_WITH_DD]


# --------------------------------------------------------------------------- #
# netCDF I/O
# --------------------------------------------------------------------------- #


def open_nc(f: str | Path) -> xr.Dataset | None:
    """Gzip-open a MADIS netCDF; scipy first, netcdf4 fallback."""
    f = str(f)
    temp_nc_file = None
    try:
        with gzip.open(f) as fp:
            ds = xr.open_dataset(fp, engine="scipy", cache=False)
    except OverflowError:
        return None
    except Exception:
        try:
            fd, temp_nc_file = tempfile.mkstemp(suffix=".nc")
            os.close(fd)
            with gzip.open(f, "rb") as f_in, open(temp_nc_file, "wb") as f_out:
                f_out.write(f_in.read())
            ds = xr.open_dataset(temp_nc_file, engine="netcdf4")
        except Exception:
            return None
        finally:
            if temp_nc_file and os.path.exists(temp_nc_file):
                os.remove(temp_nc_file)
    return ds


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract_hourly(
    ds: xr.Dataset,
    bounds: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame | None:
    """Extract one hourly dataset into a DataFrame with ID, met, DD, and QCR columns.

    Parameters
    ----------
    ds : xarray Dataset from a single MADIS hourly netCDF.
    bounds : Optional (west, south, east, north) spatial filter.

    Returns
    -------
    DataFrame with MADIS-native variable names and units, or None if empty.
    """
    if "recNum" not in ds.sizes or ds.sizes["recNum"] == 0:
        return None

    n = ds.sizes["recNum"]
    data: dict[str, np.ndarray | list] = {}

    for var in ID_VARS + MET_VARS + DD_VARS + QCR_VARS:
        if var not in ds:
            if var in MET_VARS or var in QCR_VARS:
                data[var] = np.full(n, np.nan)
            elif var in DD_VARS:
                data[var] = np.full(n, np.nan)
            else:
                data[var] = [""] * n
            continue

        if var == "observationTime":
            data[var] = pd.to_datetime(ds[var].values, errors="coerce")
        elif var in ("stationId", "dataProvider") or var in DD_VARS:
            raw = ds[var].values
            if raw.dtype.kind in ("S", "U", "O"):
                data[var] = [
                    v.decode().strip() if isinstance(v, bytes) else str(v).strip() for v in raw
                ]
            else:
                data[var] = [str(v).strip() for v in raw]
        elif var in QCR_VARS:
            data[var] = ds[var].values.astype(np.float64)
        else:
            data[var] = ds[var].values.astype(np.float64)

    df = pd.DataFrame(data)

    if bounds is not None:
        w, s, e, n_lat = bounds
        mask = (
            (df["latitude"] >= s)
            & (df["latitude"] < n_lat)
            & (df["longitude"] >= w)
            & (df["longitude"] < e)
        )
        df = df.loc[mask]

    return df if len(df) > 0 else None


# --------------------------------------------------------------------------- #
# Three-layer QC
# --------------------------------------------------------------------------- #


def apply_qc(
    df: pd.DataFrame,
    qcr_mask: int = QCR_REJECT_BITS,
) -> pd.DataFrame:
    """Three-layer QC: DD flags, QCR bitmask, physical bounds.

    NaNs individual variable values on failure; sets per-row ``qc_passed`` bool.
    Preserves native DD and QCR columns unmodified for provenance.
    """
    for var in MET_WITH_DD:
        dd_col = f"{var}DD"
        qcr_col = f"{var}QCR"

        # Layer 1: DD flag filter
        if dd_col in df.columns:
            bad_dd = ~df[dd_col].isin(ACCEPTABLE_DD)
            df.loc[bad_dd, var] = np.nan

        # Layer 2: QCR bitmask filter
        if qcr_col in df.columns:
            qcr = pd.to_numeric(df[qcr_col], errors="coerce").fillna(0).astype(np.int64)
            bad_qcr = (qcr & qcr_mask) > 0
            df.loc[bad_qcr, var] = np.nan

        # Layer 3: physical bounds
        if var in BOUNDS:
            lo, hi = BOUNDS[var]
            val = pd.to_numeric(df[var], errors="coerce")
            oob = (val < lo) | (val > hi)
            df.loc[oob, var] = np.nan

    core = ["temperature", "dewpoint", "relHumidity", "windSpeed", "windDir"]
    df["qc_passed"] = df[core].notna().any(axis=1)

    return df


# --------------------------------------------------------------------------- #
# Day-level extraction
# --------------------------------------------------------------------------- #


def extract_day(
    day_str: str,
    src_dir: str | Path,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    qcr_mask: int = QCR_REJECT_BITS,
) -> pd.DataFrame | None:
    """Extract and QC all 24 hourly files for a single day.

    Parameters
    ----------
    day_str : Date string YYYYMMDD.
    src_dir : Directory containing raw .gz files.
    bounds : Optional (west, south, east, north) spatial filter.
    qcr_mask : QCR reject bitmask.

    Returns
    -------
    QC'd DataFrame with all observations for the day, or None if no data.
    """
    src_dir = Path(src_dir)
    hours = [f"{day_str}_{h:02d}00.gz" for h in range(24)]
    frames = []

    for fname in hours:
        fpath = src_dir / fname
        if not fpath.exists():
            continue
        ds = open_nc(fpath)
        if ds is None:
            continue
        hdf = extract_hourly(ds, bounds)
        ds.close()
        if hdf is not None:
            frames.append(hdf)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = apply_qc(df, qcr_mask=qcr_mask)
    df = df.sort_values("stationId").reset_index(drop=True)

    return df
