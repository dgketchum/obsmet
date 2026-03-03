"""Time semantics policy (plan section 10).

Enforces explicit time handling: UTC timestamps for hourly records,
explicit day_basis for daily aggregation.  No implicit day definitions.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

import pandas as pd


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
    else:
        agg = grouped.mean()

    result = agg.reset_index()
    result.columns = ["date", "value"]
    result["day_basis"] = day_basis.value
    result["obs_count"] = grouped.count().values

    return result
