"""ISD fixed-width record parser and batch loader.

Parses NOAA Integrated Surface Database (ISD) files, which use a fixed-width
format with mandatory fields in positions 0-104 and optional additional data
sections after position 105.

Ported from pyisd/src/isd/record.py and io.py — only the parser internals,
not taken as a dependency.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Sentinel / missing values
# --------------------------------------------------------------------------- #

SENTINELS = {
    "latitude": "+99999",
    "longitude": "+999999",
    "report_type": "99999",
    "elevation": "+9999",
    "call_letters": "99999",
    "wind_direction": "999",
    "wind_speed": "9999",
    "ceiling": "99999",
    "visibility": "999999",
    "air_temperature": "+9999",
    "dew_point_temperature": "+9999",
    "sea_level_pressure": "99999",
}

# Quality codes considered good
GOOD_QC = {"1", "5"}

# Minimum line length (mandatory control + data section)
_MIN_LINE_LEN = 105


# --------------------------------------------------------------------------- #
# Single-record parser
# --------------------------------------------------------------------------- #


def parse_line(line: str) -> dict | None:
    """Parse a single ISD fixed-width line into a dict.

    Returns None if the line is too short or unparseable.
    """
    if len(line) < _MIN_LINE_LEN:
        return None

    def _opt(s, sentinel, transform=None):
        if s == sentinel:
            return None
        return transform(s) if transform else s

    rec = {
        "usaf_id": line[4:10],
        "ncei_id": line[10:15],
        "year": int(line[15:19]),
        "month": int(line[19:21]),
        "day": int(line[21:23]),
        "hour": int(line[23:25]),
        "minute": int(line[25:27]),
        "latitude": _opt(line[28:34], "+99999", lambda s: float(s) / 1000),
        "longitude": _opt(line[34:41], "+999999", lambda s: float(s) / 1000),
        "elevation": _opt(line[46:51], "+9999", float),
        "wind_direction": _opt(line[60:63], "999", int),
        "wind_direction_qc": line[63],
        "wind_speed": _opt(line[65:69], "9999", lambda s: float(s) / 10),
        "wind_speed_qc": line[69],
        "air_temperature": _opt(line[87:92], "+9999", lambda s: float(s) / 10),
        "air_temperature_qc": line[92],
        "dew_point_temperature": _opt(line[93:98], "+9999", lambda s: float(s) / 10),
        "dew_point_temperature_qc": line[98],
        "sea_level_pressure": _opt(line[99:104], "99999", lambda s: float(s) / 10),
        "sea_level_pressure_qc": line[104],
    }
    return rec


# --------------------------------------------------------------------------- #
# File-level reader
# --------------------------------------------------------------------------- #


def read_isd_file(path: Path | str) -> pd.DataFrame:
    """Read an ISD .gz file into a DataFrame.

    Each row is one observation record with parsed fields, datetime, and
    per-variable quality codes. Units are already scaled during parsing:
    - air_temperature, dew_point_temperature: °C (÷10 from raw)
    - wind_speed: m/s (÷10 from raw)
    - sea_level_pressure: hPa (÷10 from raw)
    - latitude, longitude: decimal degrees (÷1000 from raw)
    """
    path = Path(path)
    records = []

    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            rec = parse_line(line)
            if rec is not None:
                records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Build datetime column
    df["datetime_utc"] = pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": df["day"],
            "hour": df["hour"],
            "minute": df["minute"],
        },
        errors="coerce",
        utc=True,
    )

    # Drop rows with invalid datetimes
    df = df.dropna(subset=["datetime_utc"])

    # Build station key
    df["station_id"] = df["usaf_id"] + "-" + df["ncei_id"]

    return df


def apply_qc_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Mask values whose quality code is not in GOOD_QC.

    Sets the variable value to NaN if its QC code is not in {"1", "5"}.
    """
    df = df.copy()
    qc_pairs = [
        ("air_temperature", "air_temperature_qc"),
        ("dew_point_temperature", "dew_point_temperature_qc"),
        ("wind_speed", "wind_speed_qc"),
        ("wind_direction", "wind_direction_qc"),
        ("sea_level_pressure", "sea_level_pressure_qc"),
    ]
    for var_col, qc_col in qc_pairs:
        if var_col in df.columns and qc_col in df.columns:
            bad = ~df[qc_col].isin(GOOD_QC)
            df.loc[bad, var_col] = np.nan

    return df
