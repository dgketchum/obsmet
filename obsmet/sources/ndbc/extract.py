"""NDBC standard meteorological data file parser.

Parses NDBC space-delimited text files (gzip-compressed for historical,
plain text for latest) into pandas DataFrames with hourly observations.

Ported from dads-mvp/extract/met_data/obs/ndbc_download.py.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import pandas as pd

from obsmet.sources.ndbc.download import STDMET_COLUMNS

# Missing/NA values used by NDBC
NA_VALUES = {"99", "99.0", "999", "999.0", "9999", "9999.0"}

# Rename map: NDBC raw column names → friendly names
RENAME_MAP = {
    "WDIR": "wind_dir",
    "WSPD": "wind_speed",
    "GST": "wind_gust",
    "WVHT": "wave_height",
    "DPD": "dominant_wave_period",
    "APD": "average_wave_period",
    "MWD": "mean_wave_dir",
    "PRES": "pressure",
    "ATMP": "air_temp",
    "WTMP": "water_temp",
    "DEWP": "dewpoint",
    "VIS": "visibility",
    "TIDE": "tide",
}

# Columns to coerce to numeric after loading
_NUMERIC_COLS = list(RENAME_MAP.values())


def read_stdmet_file(path: Path | str) -> pd.DataFrame:
    """Read a single NDBC stdmet file into a DataFrame.

    Handles both .txt.gz (historical) and .txt (latest) formats.

    Returns DataFrame with datetime_utc index and renamed columns.
    """
    path = Path(path)

    # Read file content
    if path.suffix == ".gz" or path.name.endswith(".txt.gz"):
        with gzip.open(path, "rt", errors="replace") as f:
            text = f.read()
    else:
        text = path.read_text(errors="replace")

    if not text.strip():
        return pd.DataFrame()

    lines = text.strip().split("\n")

    # Detect header line
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip("#").strip()
        if "YY" in stripped and "MM" in stripped:
            header_line = stripped
            data_start = i + 1
            # Skip units line if present (starts with # and contains "yr" or "deg")
            if data_start < len(lines):
                next_line = lines[data_start].lstrip("#").strip()
                if next_line and (next_line.startswith("yr") or next_line.startswith("deg")):
                    data_start += 1
            break

    if header_line:
        col_names = header_line.split()
    else:
        col_names = STDMET_COLUMNS[:18]

    # Read data
    data_text = "\n".join(lines[data_start:])
    try:
        df = pd.read_csv(
            io.StringIO(data_text),
            sep=r"\s+",
            names=col_names,
            na_values=list(NA_VALUES),
            dtype=str,
            header=None,
        )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Handle YYYY vs YY column
    year_col = "YYYY" if "YYYY" in df.columns else "YY"
    if year_col not in df.columns:
        return pd.DataFrame()

    # Add mm column if missing (default 0)
    if "mm" not in df.columns:
        df["mm"] = "0"

    # Build datetime
    date_parts = pd.DataFrame(
        {
            "year": pd.to_numeric(df[year_col], errors="coerce"),
            "month": pd.to_numeric(df["MM"], errors="coerce"),
            "day": pd.to_numeric(df["DD"], errors="coerce"),
            "hour": pd.to_numeric(df["hh"], errors="coerce"),
            "minute": pd.to_numeric(df["mm"], errors="coerce"),
        }
    )

    # Handle 2-digit years
    mask_2d = date_parts["year"] < 100
    date_parts.loc[mask_2d & (date_parts["year"] >= 70), "year"] += 1900
    date_parts.loc[mask_2d & (date_parts["year"] < 70), "year"] += 2000

    df["datetime_utc"] = pd.to_datetime(date_parts, errors="coerce", utc=True)
    df = df.dropna(subset=["datetime_utc"])

    # Rename meteorological columns
    df = df.rename(columns=RENAME_MAP)

    # Coerce numeric columns
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop date-part columns
    drop_cols = [c for c in [year_col, "YY", "YYYY", "MM", "DD", "hh", "mm"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Sort and dedup
    df = df.sort_values("datetime_utc").drop_duplicates(subset=["datetime_utc"], keep="first")

    return df.reset_index(drop=True)


def read_station_files(station_dir: Path | str, station_id: str) -> pd.DataFrame:
    """Read all stdmet files for a station and concatenate.

    Looks for both historical ({sid}h{year}.txt.gz) and latest ({sid}_latest.txt)
    files in the station directory.
    """
    station_dir = Path(station_dir)
    sid = station_id.lower()

    frames = []

    # Historical files
    for gz in sorted(station_dir.glob(f"{sid}h*.txt.gz")):
        df = read_stdmet_file(gz)
        if not df.empty:
            frames.append(df)

    # Latest file
    latest = station_dir / f"{sid}_latest.txt"
    if latest.exists():
        df = read_stdmet_file(latest)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("datetime_utc").drop_duplicates(
        subset=["datetime_utc"], keep="first"
    )

    return combined.reset_index(drop=True)
