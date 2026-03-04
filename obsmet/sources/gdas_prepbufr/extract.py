"""GDAS PrepBUFR extraction — ADPSFC/SFCSHP surface obs from BUFR files.

Uses py-ncepbufr to decode PrepBUFR files and extract surface observations
(message types ADPSFC and SFCSHP). Handles both NR (2009+) and WO40
(pre-2009) tar archive formats.

Ported from GDASApp/scripts/gdas_extract_obs.py.
"""

from __future__ import annotations

import re
import tarfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MSG_TYPES = ("ADPSFC", "SFCSHP")

# BUFR missing value sentinel — values > this are treated as missing
MISSING = 1e10

# BUFR mnemonics
HDR_MNEMONICS = "SID XOB YOB ELV TYP DHR"
OBS_MNEMONICS = "POB PQM TOB TQM QOB QQM UOB VOB WQM ZOB ZQM"
SST_MNEMONICS = "SST1 SSTQM"


def _is_missing(v) -> bool:
    """Check if a BUFR value is missing."""
    return np.isnan(v) or v > MISSING


def _decode_station_id(sid_raw) -> str | None:
    """Decode BUFR station ID from float64-packed 8-char ASCII."""
    if np.ma.is_masked(sid_raw):
        return None
    try:
        sid = sid_raw.tobytes().decode("ascii", errors="ignore").strip()
        sid = "".join(c for c in sid if c.isprintable()).strip()
        return sid if sid else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Single BUFR file extraction
# --------------------------------------------------------------------------- #


def extract_bufr_file(
    bufr_path: Path | str,
    cycle_hour: int,
    date_str: str,
) -> pd.DataFrame:
    """Extract surface observations from a single PrepBUFR file.

    Parameters
    ----------
    bufr_path : Path to the PrepBUFR file.
    cycle_hour : Analysis cycle hour (0, 6, 12, or 18).
    date_str : Date string YYYYMMDD.

    Returns
    -------
    DataFrame with columns: station_id, latitude, longitude, elevation,
    obs_type, datetime_utc, cycle, msg_type, pressure, pressure_qm,
    temperature, temperature_qm, specific_humidity, humidity_qm,
    u_wind, v_wind, wind_qm, height, height_qm, sst, sst_qm.
    """
    import ncepbufr

    bufr_path = Path(bufr_path)
    records = []

    cycle_dt = datetime.strptime(f"{date_str}{cycle_hour:02d}", "%Y%m%d%H")

    bufr = ncepbufr.open(str(bufr_path))
    while bufr.advance() == 0:
        msg_type = bufr.msg_type.strip()
        if msg_type not in MSG_TYPES:
            continue

        while bufr.load_subset() == 0:
            try:
                hdr = bufr.read_subset(HDR_MNEMONICS)
            except Exception:
                continue

            sid = _decode_station_id(hdr[0, 0])
            if sid is None:
                continue

            xob = float(hdr[1, 0])
            yob = float(hdr[2, 0])
            elv = float(hdr[3, 0])
            typ = float(hdr[4, 0])
            dhr = float(hdr[5, 0])

            if _is_missing(xob) or _is_missing(yob):
                continue

            # Normalize longitude: 0-360 → -180 to 180
            lon = xob - 360.0 if xob > 180.0 else xob
            lat = yob
            elv = np.nan if _is_missing(elv) else elv
            obs_type = int(typ) if not _is_missing(typ) else None

            # Observation datetime
            if _is_missing(dhr):
                obs_dt = cycle_dt
            else:
                obs_dt = cycle_dt + timedelta(hours=dhr)

            try:
                obs = bufr.read_subset(OBS_MNEMONICS)
            except Exception:
                continue

            def _val(arr, idx):
                v = float(arr[idx, 0])
                return np.nan if _is_missing(v) else v

            def _qm(arr, idx):
                v = float(arr[idx, 0])
                return int(v) if not _is_missing(v) else None

            pob = _val(obs, 0)
            pqm = _qm(obs, 1)
            tob = _val(obs, 2)
            tqm = _qm(obs, 3)
            qob = _val(obs, 4)
            qqm = _qm(obs, 5)
            uob = _val(obs, 6)
            vob = _val(obs, 7)
            wqm = _qm(obs, 8)
            zob = _val(obs, 9)
            zqm = _qm(obs, 10)

            # Unit conversions
            pressure = pob * 100.0 if not np.isnan(pob) else np.nan  # hPa → Pa
            q = qob * 1e-6 if not np.isnan(qob) else np.nan  # mg/kg → kg/kg

            # SST (optional)
            sst_val = np.nan
            sst_qm = None
            try:
                sst_data = bufr.read_subset(SST_MNEMONICS)
                sv = float(sst_data[0, 0])
                if not _is_missing(sv):
                    sst_val = sv
                sq = float(sst_data[1, 0])
                if not _is_missing(sq):
                    sst_qm = int(sq)
            except Exception:
                pass

            records.append(
                {
                    "station_id": sid,
                    "latitude": lat,
                    "longitude": lon,
                    "elevation": elv,
                    "obs_type": obs_type,
                    "datetime_utc": pd.Timestamp(obs_dt, tz="UTC"),
                    "cycle": cycle_hour,
                    "msg_type": msg_type,
                    "pressure": pressure,
                    "pressure_qm": pqm,
                    "temperature": tob,
                    "temperature_qm": tqm,
                    "specific_humidity": q,
                    "humidity_qm": qqm,
                    "u_wind": uob,
                    "v_wind": vob,
                    "wind_qm": wqm,
                    "height": zob,
                    "height_qm": zqm,
                    "sst": sst_val,
                    "sst_qm": sst_qm,
                }
            )

    bufr.close()

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Tar archive extraction
# --------------------------------------------------------------------------- #


def _extract_nr_tar(tar_path: Path, tmp_dir: Path) -> list[tuple[Path, int]]:
    """Extract NR-format tar (flat, 2009+). Returns list of (bufr_path, cycle_hour)."""
    results = []
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(tmp_dir, filter="data")
        for member in tf.getnames():
            m = re.search(r"t(\d{2})z", member)
            if m:
                cycle = int(m.group(1))
                results.append((tmp_dir / member, cycle))
    return results


def _extract_wo40_tar(tar_path: Path, tmp_dir: Path) -> list[tuple[Path, int]]:
    """Extract WO40-format tar (nested, pre-2009). Returns list of (bufr_path, cycle_hour)."""
    results = []
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(tmp_dir, filter="data")
        for member in tf.getnames():
            m = re.search(r"prepbufr\.gdas\.(\d{10})\.", member)
            if m:
                cycle = int(m.group(1)[-2:])
                results.append((tmp_dir / member, cycle))
    return results


def extract_day(tar_path: Path | str, date_str: str) -> pd.DataFrame:
    """Extract all surface obs from a daily PrepBUFR tar archive.

    Handles both NR (.nr.tar.gz) and WO40 (.wo40.tar.gz) formats.

    Parameters
    ----------
    tar_path : Path to the daily tar.gz archive.
    date_str : Date string YYYYMMDD.

    Returns
    -------
    DataFrame with all surface observations for the day.
    """
    tar_path = Path(tar_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        if ".nr." in tar_path.name:
            bufr_files = _extract_nr_tar(tar_path, tmp)
        else:
            bufr_files = _extract_wo40_tar(tar_path, tmp)

        frames = []
        for bufr_path, cycle_hour in bufr_files:
            if bufr_path.exists():
                df = extract_bufr_file(bufr_path, cycle_hour, date_str)
                if not df.empty:
                    frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).sort_values("datetime_utc")
