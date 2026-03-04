"""Time semantics policy (plan section 10).

Enforces explicit time handling: UTC timestamps for hourly records,
explicit day_basis for daily aggregation.  No implicit day definitions.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

import numpy as np
import pandas as pd

from obsmet.core.provenance import RunProvenance


class DayBasis(str, Enum):
    """Supported daily aggregation bases (plan decision D2: UTC-only for v1)."""

    UTC = "utc"
    LOCAL = "local"


def ensure_utc(ts: pd.Timestamp | dt.datetime) -> pd.Timestamp:
    """Ensure a timestamp is tz-aware UTC.

    Raises ValueError if the timestamp is naive (no tzinfo).
    """
    if isinstance(ts, dt.datetime) and not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"Naive timestamp not allowed: {ts!r}. Attach UTC tz first.")
    return ts.tz_convert("UTC")


def assign_utc_date(datetime_utc: pd.Series) -> pd.Series:
    """Derive the UTC date from a UTC datetime series."""
    return datetime_utc.dt.date


def hourly_coverage(
    datetime_utc: pd.Series,
    date: dt.date,
    *,
    required_hours: int = 18,
) -> dict:
    """Compute hourly coverage stats for a single UTC day.

    Returns dict with keys: obs_count, meets_threshold, morning_ok, afternoon_ok.
    """
    day_start = pd.Timestamp(date, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)
    mask = (datetime_utc >= day_start) & (datetime_utc < day_end)
    hours_present = datetime_utc[mask].dt.hour.unique()
    obs_count = int(mask.sum())

    morning_hours = set(range(6, 12))
    afternoon_hours = set(range(12, 18))
    hours_set = set(hours_present)

    return {
        "obs_count": obs_count,
        "meets_threshold": len(hours_present) >= required_hours,
        "morning_ok": len(morning_hours & hours_set) >= 4,
        "afternoon_ok": len(afternoon_hours & hours_set) >= 4,
    }


def aggregate_daily(
    values: pd.Series,
    datetime_utc: pd.Series,
    variable: str,
    *,
    day_basis: DayBasis = DayBasis.UTC,
) -> pd.DataFrame:
    """Aggregate hourly values to daily records.

    Groups by UTC date and computes min/max/mean as appropriate for the
    variable type.

    Returns a DataFrame with columns: date, day_basis, value, obs_count.
    """
    if day_basis != DayBasis.UTC:
        raise NotImplementedError("Only UTC day basis supported in v1.")

    df = pd.DataFrame({"datetime_utc": datetime_utc, "value": values})
    df["date"] = df["datetime_utc"].dt.date
    grouped = df.groupby("date")["value"]

    if variable in ("tmax",):
        agg = grouped.max()
    elif variable in ("tmin",):
        agg = grouped.min()
    elif variable in ("prcp",):
        agg = grouped.sum()
    elif variable in ("rsds",):
        # Convert mean instantaneous W/m² to daily MJ/m²/day
        agg = grouped.mean() * 86400.0 / 1e6
    elif variable in ("wind_dir",):
        agg = grouped.apply(circular_mean_deg)
    else:
        agg = grouped.mean()

    result = agg.reset_index()
    result.columns = ["date", "value"]
    result["day_basis"] = day_basis.value
    result["obs_count"] = grouped.count().values

    return result


# --------------------------------------------------------------------------- #
# Circular mean for angular variables
# --------------------------------------------------------------------------- #


def circular_mean_deg(angles: pd.Series) -> float:
    """Compute the circular (angular) mean of a series of degree values.

    Uses the atan2(mean(sin), mean(cos)) approach which correctly handles
    wraparound (e.g. averaging 350° and 10° gives 0°, not 180°).

    Parameters
    ----------
    angles : Series of angles in degrees.

    Returns
    -------
    Mean angle in degrees [0, 360).
    """
    angles = angles.dropna()
    if angles.empty:
        return np.nan

    rad = np.deg2rad(angles.values)
    mean_sin = np.mean(np.sin(rad))
    mean_cos = np.mean(np.cos(rad))
    mean_angle = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360
    return float(mean_angle)


# --------------------------------------------------------------------------- #
# Daily aggregation engine (wide-form)
# --------------------------------------------------------------------------- #

# Sources that support hourly → daily aggregation
DAILY_SOURCES = ["madis", "isd", "gdas", "ndbc"]

# Variables and their aggregation method for daily rollup from hourly data
DAILY_AGG_MAP = {
    "tair": "mean",
    "td": "mean",
    "rh": "mean",
    "wind": "mean",
    "wind_dir": "circular_mean",
    "slp": "mean",
    "prcp": "sum",
}


def aggregate_daily_wide(
    df: pd.DataFrame,
    provenance: RunProvenance,
    *,
    agg_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate wide-form hourly observations to daily records.

    Input: wide-form hourly DataFrame with station_key, datetime_utc,
    and variable columns.
    Output: DataFrame conforming to OBS_DAILY_CORE_SCHEMA + DAILY_METRIC_FIELDS.

    Parameters
    ----------
    df : Wide-form hourly DataFrame.
    provenance : RunProvenance for this pipeline run.
    agg_map : Override aggregation map. Defaults to DAILY_AGG_MAP.
    """
    if agg_map is None:
        agg_map = DAILY_AGG_MAP

    if df.empty or "datetime_utc" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["date"] = df["datetime_utc"].dt.date

    if "station_key" not in df.columns:
        return pd.DataFrame()

    groups = df.groupby(["station_key", "date"])

    daily_records = []
    for (stn, day), grp in groups:
        cov = hourly_coverage(grp["datetime_utc"], day)
        cov_str = (
            f"n={cov['obs_count']}"
            f",thresh={'Y' if cov['meets_threshold'] else 'N'}"
            f",am={'Y' if cov['morning_ok'] else 'N'}"
            f",pm={'Y' if cov['afternoon_ok'] else 'N'}"
        )

        rec = {
            "station_key": stn,
            "date": day,
            "day_basis": "utc",
            "obs_count": len(grp),
            "coverage_flags": cov_str,
            "qc_state": "pass",
            "qc_rules_version": provenance.qaqc_rules_version,
            "transform_version": provenance.transform_version,
            "ingest_run_id": provenance.run_id,
        }

        for var, agg_type in agg_map.items():
            if var not in grp.columns:
                continue
            vals = pd.to_numeric(grp[var], errors="coerce").dropna()
            if vals.empty:
                continue
            if agg_type == "mean":
                rec[var] = float(vals.mean())
            elif agg_type == "sum":
                rec[var] = float(vals.sum())
            elif agg_type == "circular_mean":
                rec[var] = circular_mean_deg(vals)

        # Tmax/tmin/tmean from tair
        if "tair" in grp.columns:
            tair_vals = pd.to_numeric(grp["tair"], errors="coerce").dropna()
            if not tair_vals.empty:
                rec["tmax"] = float(tair_vals.max())
                rec["tmin"] = float(tair_vals.min())
                rec["tmean"] = float(tair_vals.mean())

        # Convert rsds_hourly W/m² to rsds MJ/m²/day
        if "rsds_hourly" in grp.columns:
            rsds_vals = pd.to_numeric(grp["rsds_hourly"], errors="coerce").dropna()
            if not rsds_vals.empty:
                rec["rsds"] = float(rsds_vals.mean()) * 86400.0 / 1e6

        daily_records.append(rec)

    if not daily_records:
        return pd.DataFrame()

    result = pd.DataFrame(daily_records)
    result["date"] = pd.to_datetime(result["date"])
    return result
