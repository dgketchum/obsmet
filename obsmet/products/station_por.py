"""Station period-of-record (POR) pivot product.

Reads normalized hourly parquets, aggregates to daily, applies Tier 2 temporal QC,
and writes one parquet per station.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import traceback
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

from obsmet.core.manifest import Manifest
from obsmet.core.provenance import RunProvenance
from obsmet.core.time_policy import DAILY_AGG_MAP, aggregate_daily_wide, required_hours_for_source
from obsmet.qaqc.rules.temporal import (
    MonthlyZScoreRule,
    RHDriftRule,
    RsPeriodRatioRule,
)

logger = logging.getLogger(__name__)

_STATE_PRECEDENCE = {"fail": 2, "suspect": 1, "pass": 0}

# Variables excluded from z-score (zero-inflated / skewed / circular / redundant)
_ZSCORE_SKIP_VARS = {"prcp", "wind", "wind_dir", "tair"}  # tair is redundant with tmean

# Physical upper bound for daily precipitation (mm) — approx. world record
_DAILY_PRCP_MAX_MM = 610.0


def _worst_state(a: str, b: str) -> str:
    return a if _STATE_PRECEDENCE.get(a, 0) >= _STATE_PRECEDENCE.get(b, 0) else b


def _apply_tier2_qc(
    station_df: pd.DataFrame,
    variable_columns: list[str],
    *,
    rso: np.ndarray | None = None,
) -> pd.DataFrame:
    """Apply Tier 2 temporal QC to a single station's daily DataFrame.

    Runs MonthlyZScoreRule (excluding precip) on each variable column,
    RHDriftRule on rhmax/rhmin with correction, RsPeriodRatioRule on rsds
    with correction (if Rso provided), and a physical upper bound on daily
    precipitation.

    Merges with existing qc_state (worst wins).

    Parameters
    ----------
    rso : np.ndarray, optional
        365-element clear-sky array (MJ/m²/day, same units as rsds) for
        Rs period-ratio correction.  From RSUN raster or ASCE flat-earth.
    """
    zscore_rule = MonthlyZScoreRule()
    rh_drift_rule = RHDriftRule()
    rs_ratio_rule = RsPeriodRatioRule()

    if "date" not in station_df.columns:
        return station_df

    dates = pd.to_datetime(station_df["date"])

    tier2_state = pd.Series("pass", index=station_df.index)
    tier2_reasons: list[list[str]] = [[] for _ in range(len(station_df))]

    for col in variable_columns:
        if col not in station_df.columns:
            continue

        vals = pd.to_numeric(station_df[col], errors="coerce")

        # Monthly z-score (skip zero-inflated / skewed variables)
        if col not in _ZSCORE_SKIP_VARS:
            z_states = zscore_rule.check_series(vals, dates)
            for i, (idx, st) in enumerate(z_states.items()):
                if st != "pass":
                    pos = station_df.index.get_loc(idx)
                    tier2_state.iloc[pos] = _worst_state(tier2_state.iloc[pos], st)
                    tier2_reasons[pos].append(f"zscore_{col}")

        # TODO: stuck sensor detection belongs in hourly normalize

    # Daily precip upper bound
    if "prcp" in station_df.columns:
        prcp_vals = pd.to_numeric(station_df["prcp"], errors="coerce")
        prcp_extreme = prcp_vals > _DAILY_PRCP_MAX_MM
        for idx in station_df.index[prcp_extreme]:
            pos = station_df.index.get_loc(idx)
            tier2_state.iloc[pos] = _worst_state(tier2_state.iloc[pos], "fail")
            tier2_reasons[pos].append("prcp_exceeds_daily_max")

    # RH drift (yearly percentile correction) — use rhmax/rhmin if available
    station_df = station_df.copy()
    if "rh" in station_df.columns:
        if "rhmax" in station_df.columns and "rhmin" in station_df.columns:
            rhmax = pd.to_numeric(station_df["rhmax"], errors="coerce")
            rhmin = pd.to_numeric(station_df["rhmin"], errors="coerce")
        else:
            rhmax = pd.to_numeric(station_df["rh"], errors="coerce")
            rhmin = rhmax.copy()
        years = dates.dt.year
        _, corr_rhmax, corr_rhmin, _ = rh_drift_rule.correct_series(rhmax, rhmin, years)
        # rh_drift corrects but never flags — agweather never NaN's due to RH drift

        # Store corrected values (originals stay in base columns)
        rh_mean = pd.to_numeric(station_df["rh"], errors="coerce").values
        valid_orig = rhmax.values > 0
        correction_ratio = np.ones(len(rhmax))
        correction_ratio[valid_orig] = corr_rhmax[valid_orig] / rhmax.values[valid_orig]
        station_df["rh_corrected"] = np.where(np.isnan(rh_mean), np.nan, rh_mean * correction_ratio)
        station_df["rh_corrected"] = np.clip(station_df["rh_corrected"], 0, 100)
        station_df["rhmax_corrected"] = corr_rhmax
        station_df["rhmin_corrected"] = corr_rhmin

    # Rs period-ratio correction (requires RSUN Rso)
    if rso is not None and "rsds" in station_df.columns:
        rs_vals = pd.to_numeric(station_df["rsds"], errors="coerce")
        doy = dates.dt.dayofyear
        rs_states, corr_rs = rs_ratio_rule.correct_series(rs_vals, rso, doy)
        for idx, st in rs_states.items():
            if st != "pass":
                pos = station_df.index.get_loc(idx)
                tier2_state.iloc[pos] = _worst_state(tier2_state.iloc[pos], st)
                tier2_reasons[pos].append("rs_period_ratio")

        # Store corrected Rs (original stays in rsds)
        station_df["rsds_corrected"] = corr_rs

    # Merge with existing qc_state
    if "qc_state" in station_df.columns:
        existing = station_df["qc_state"].fillna("pass")
        merged = [_worst_state(e, t) for e, t in zip(existing, tier2_state)]
        station_df["qc_state"] = merged
    else:
        station_df["qc_state"] = tier2_state.values

    # Merge reason codes
    existing_reasons = station_df.get("qc_reason_codes", pd.Series("", index=station_df.index))
    existing_reasons = existing_reasons.fillna("")
    new_reasons = []
    for old, tier2 in zip(existing_reasons, tier2_reasons):
        parts = [old] if old else []
        parts.extend(tier2)
        new_reasons.append(",".join(parts))
    station_df["qc_reason_codes"] = new_reasons

    return station_df


_N_BUCKETS = 100
_GDAS_STAGE_DIRNAME = "_gdas_hourly_stage"
# Daily variable columns for Tier 2 QC in station_por. These are the post-aggregation
# column names (tmax/tmin/tmean derived from tair, rsds derived from rsds_hourly).
# Sources not listed here fall through to _VARIABLE_COLUMNS (hourly names), which
# only works for daily-native sources where hourly and daily names are the same.
_STATION_POR_VARIABLE_COLUMNS: dict[str, list[str]] = {
    "madis": ["tmax", "tmin", "tmean", "td", "rh", "wind", "wind_dir", "prcp", "rsds"],
    "ghcnh": ["tmax", "tmin", "tmean", "td", "wind", "wind_dir", "slp", "psfc", "prcp", "rh"],
    "gdas": ["tmean", "td", "psfc"],
    "ndbc": ["tmax", "tmin", "tmean", "td", "wind", "wind_dir", "slp"],
    "snotel": ["tmax", "tmin", "tmean", "prcp", "swe", "snow_depth"],
    "eccc": ["tmax", "tmin", "tmean", "td", "rh", "wind", "wind_dir", "prcp", "psfc"],
}


def _bucket_id(station_key: str, n_buckets: int = _N_BUCKETS) -> int:
    """Deterministic hash bucket for a station key."""
    return hash(station_key) % n_buckets


def _safe_station_filename(station_key: str) -> str:
    """Map a station key to a filesystem-safe parquet stem."""
    return station_key.replace(":", "_").replace("/", "_")


def _build_failure_record(
    *,
    source: str,
    station_key: str,
    bucket_id: int,
    exc: Exception,
) -> dict[str, str | int]:
    """Capture enough context to debug a failed station without halting the build."""
    return {
        "source": source,
        "phase": "pass2_station",
        "station_key": station_key,
        "bucket_id": bucket_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def _build_file_failure_record(
    *,
    source: str,
    input_file: Path,
    exc: Exception,
) -> dict[str, str]:
    """Capture pass-1 file read failures so missing daily chunks are visible."""
    return {
        "source": source,
        "phase": "pass1_read_daily",
        "input_file": str(input_file),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def _build_qc_skip_record(
    *,
    source: str,
    station_key: str,
    bucket_id: int,
    qc_name: str,
    reason: str,
    details: str = "",
) -> dict[str, str | int]:
    """Record successful stations where a QC step was skipped and why."""
    return {
        "source": source,
        "station_key": station_key,
        "bucket_id": bucket_id,
        "qc_name": qc_name,
        "reason": reason,
        "details": details,
    }


def _station_por_variable_columns(source: str, default_columns: list[str]) -> list[str]:
    """Return source-specific Tier 2 variable targets for station POR."""
    return _STATION_POR_VARIABLE_COLUMNS.get(source, default_columns)


# Variables that produce daily output via aggregation or derivation.
# Only these are considered when deciding whether a row has usable data.
_AGGREGATABLE_HOURLY_VARS = frozenset(list(DAILY_AGG_MAP.keys()) + ["rsds_hourly"])


def _drop_failed_hourly_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter QC failures: per-variable nulling when available, row-drop as fallback.

    When per-variable QC columns (``{var}_qc_state``) are present, nulls only
    the specific variables that failed while preserving good variables on the
    same row. Drops rows only when all *aggregatable* data columns are NaN
    after nulling (ancillary fields like wind_gust do not keep a row alive).

    When per-variable columns are absent (legacy normalized data), falls back
    to dropping entire rows where ``qc_state == "fail"``.
    """
    if df.empty or "qc_state" not in df.columns:
        return df

    # Check for per-variable QC columns
    var_qc_cols = [c for c in df.columns if c.endswith("_qc_state")]
    if not var_qc_cols:
        # Legacy fallback: row-level drop
        states = df["qc_state"].fillna("pass")
        return df.loc[states != "fail"].copy()

    # Per-variable nulling: null only variables where {var}_qc_state == "fail"
    df = df.copy()
    for qc_col in var_qc_cols:
        var_name = qc_col.removesuffix("_qc_state")
        if var_name not in df.columns:
            continue
        fail_mask = df[qc_col].fillna("pass") == "fail"
        if fail_mask.any():
            df.loc[fail_mask, var_name] = np.nan

    # Drop rows where ALL aggregatable columns are NaN (no usable daily output)
    agg_cols = [c for c in _AGGREGATABLE_HOURLY_VARS if c in df.columns]
    if agg_cols:
        all_nan = df[agg_cols].isna().all(axis=1)
        df = df.loc[~all_nan]

    return df


def _prepare_gdas_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare GDAS hourly rows for daily aggregation.

    Derives Tier-2 input columns if missing and collapses duplicate
    station/timestamp rows so coverage reflects unique observations rather than
    variable bundles or repeated cycle snapshots.
    """
    from obsmet.sources.gdas_prepbufr.adapter import (
        _collapse_duplicate_timestamps,
        _dewpoint_from_specific_humidity,
        _wind_from_uv,
    )

    if df.empty or "station_key" not in df.columns or "datetime_utc" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

    if "td" not in df.columns and {"q", "psfc"}.issubset(df.columns):
        df["td"] = _dewpoint_from_specific_humidity(df["q"].to_numpy(), df["psfc"].to_numpy())

    if ("wind" not in df.columns or "wind_dir" not in df.columns) and {"u", "v"}.issubset(
        df.columns
    ):
        wind, wind_dir = _wind_from_uv(df["u"].to_numpy(), df["v"].to_numpy())
        if "wind" not in df.columns:
            df["wind"] = wind
        if "wind_dir" not in df.columns:
            df["wind_dir"] = wind_dir

    return _collapse_duplicate_timestamps(df)


def _resolve_station_rso(
    grp: pd.DataFrame,
    station_key: str,
    *,
    use_rsun_raster: bool,
    station_coords: dict[str, tuple[float, float]],
    rsun_path: str | None,
) -> tuple[np.ndarray | None, dict[str, str] | None]:
    """Resolve the clear-sky radiation series used by Rs tier-2 QC."""
    rsun_error = ""
    asce_error = ""

    if use_rsun_raster and station_key in station_coords:
        lon, lat = station_coords[station_key]
        try:
            from obsmet.products.rsun import extract_station_rsun

            return extract_station_rsun(lon, lat, str(rsun_path)), None
        except Exception as exc:
            rsun_error = f"{type(exc).__name__}: {exc}"

    has_lat = "lat" in grp.columns
    has_elev = "elev_m" in grp.columns
    if has_lat and has_elev:
        lat_val = pd.to_numeric(grp["lat"], errors="coerce").dropna()
        elev_val = pd.to_numeric(grp["elev_m"], errors="coerce").dropna()
        if not lat_val.empty and not elev_val.empty:
            try:
                from obsmet.products.rsun import compute_rso_asce

                return compute_rso_asce(float(lat_val.iloc[0]), float(elev_val.iloc[0])), None
            except Exception as exc:
                asce_error = f"{type(exc).__name__}: {exc}"

    details: list[str] = []
    if use_rsun_raster:
        if station_key not in station_coords:
            details.append("station coords missing from station index")
        elif rsun_error:
            details.append(f"RSUN lookup failed: {rsun_error}")

    lat_val = pd.to_numeric(grp["lat"], errors="coerce").dropna() if has_lat else pd.Series()
    elev_val = pd.to_numeric(grp["elev_m"], errors="coerce").dropna() if has_elev else pd.Series()
    if not has_lat or not has_elev:
        missing = []
        if not has_lat:
            missing.append("lat")
        if not has_elev:
            missing.append("elev_m")
        details.append(f"missing columns: {', '.join(missing)}")
    else:
        if lat_val.empty:
            details.append("lat contains only null/invalid values")
        if elev_val.empty:
            details.append("elev_m contains only null/invalid values")
        if asce_error:
            details.append(f"ASCE fallback failed: {asce_error}")

    reason = "rso_unavailable"
    if use_rsun_raster and rsun_error and asce_error:
        reason = "rsun_and_asce_failed"
    elif use_rsun_raster and station_key not in station_coords:
        reason = "station_index_missing_coords"
    elif asce_error:
        reason = "asce_failed"
    elif not has_lat or not has_elev:
        reason = "missing_lat_or_elev"
    elif lat_val.empty:
        reason = "invalid_lat"
    elif elev_val.empty:
        reason = "invalid_elev_m"

    return None, {"reason": reason, "details": "; ".join(details)}


def _process_station_group(
    station_key: str,
    grp: pd.DataFrame,
    *,
    variable_columns: list[str],
    out_dir: Path,
    source: str,
    bucket_id: int,
    use_rsun_raster: bool,
    station_coords: dict[str, tuple[float, float]],
    rsun_path: str | None,
    min_por_days: int,
) -> tuple[int | None, dict[str, str | int] | None, dict[str, str | int] | None]:
    """QC and write one station parquet, returning row count or failure details."""
    try:
        grp = grp.sort_values("date").reset_index(drop=True)

        if "date" in grp.columns:
            grp = grp.drop_duplicates(subset=["date"], keep="first")

        if min_por_days > 0:
            obs_count = pd.to_numeric(grp.get("obs_count", pd.Series(dtype=float)), errors="coerce")
            required_hours = required_hours_for_source(source)
            sufficient_days = int((obs_count >= required_hours).sum())
            if sufficient_days < min_por_days:
                return (
                    None,
                    None,
                    _build_qc_skip_record(
                        source=source,
                        station_key=station_key,
                        bucket_id=bucket_id,
                        qc_name="min_por_days",
                        reason="insufficient_por_days",
                        details=(
                            f"sufficient_days={sufficient_days},"
                            f"required_days={min_por_days},"
                            f"required_hours={required_hours}"
                        ),
                    ),
                )

        rso = None
        rs_qc_skip = None
        if "rsds" in grp.columns:
            rso, skip_info = _resolve_station_rso(
                grp,
                station_key,
                use_rsun_raster=use_rsun_raster,
                station_coords=station_coords,
                rsun_path=rsun_path,
            )
            if rso is None and skip_info is not None:
                rs_qc_skip = _build_qc_skip_record(
                    source=source,
                    station_key=station_key,
                    bucket_id=bucket_id,
                    qc_name="rs_period_ratio",
                    reason=skip_info["reason"],
                    details=skip_info["details"],
                )
        grp = _apply_tier2_qc(grp, variable_columns, rso=rso)

        out_path = out_dir / f"{_safe_station_filename(station_key)}.parquet"
        grp.to_parquet(out_path, index=False, compression="snappy")
        return len(grp), None, rs_qc_skip
    except Exception as exc:
        return (
            None,
            _build_failure_record(
                source=source,
                station_key=station_key,
                bucket_id=bucket_id,
                exc=exc,
            ),
            None,
        )


def _write_failure_report(
    out_dir: Path,
    source: str,
    run_id: str,
    failures: list[dict[str, str | int]],
    qc_skips: list[dict[str, str | int]],
) -> Path:
    """Write a source-scoped JSON report beside the station_por source directory."""
    report_path = out_dir.parent / f"station_por_failures_{source}.json"
    payload = {
        "source": source,
        "run_id": run_id,
        "out_dir": str(out_dir),
        "failure_count": len(failures),
        "failures": failures,
        "qc_skip_count": len(qc_skips),
        "qc_skips": qc_skips,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return report_path


# ------------------------------------------------------------------ #
# Pass 1 worker: read one day file, aggregate to daily
# ------------------------------------------------------------------ #


def _aggregate_one_file(args: tuple) -> dict[str, object]:
    """Worker: read a normalized parquet, aggregate to daily, return DataFrame."""
    pf_str, prov_dict, start_date, end_date = args
    pf = Path(pf_str)

    # Reconstruct provenance (can't pickle RunProvenance across processes easily)
    prov = RunProvenance(**prov_dict)

    try:
        df = pd.read_parquet(pf)
    except Exception as exc:
        return {
            "daily": None,
            "failure": _build_file_failure_record(
                source=str(prov_dict.get("source", "")),
                input_file=pf,
                exc=exc,
            ),
        }

    if df.empty or "station_key" not in df.columns:
        return {"daily": None, "failure": None}

    if "datetime_utc" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
        if start_date is not None:
            df = df[df["datetime_utc"].dt.date >= start_date]
        if end_date is not None:
            df = df[df["datetime_utc"].dt.date <= end_date]
        if df.empty:
            return {"daily": None, "failure": None}

    df = _drop_failed_hourly_rows(df)
    if df.empty:
        return {"daily": None, "failure": None}

    daily = aggregate_daily_wide(df, prov)
    return {"daily": daily if not daily.empty else None, "failure": None}


# ------------------------------------------------------------------ #
# Pass 1 worker: passthrough for daily-native sources (RAWS)
# ------------------------------------------------------------------ #

# Sources whose normalized parquets are already at daily resolution.
# SNOTEL was here but moved to hourly after AWDB adapter integration.
_DAILY_NATIVE_SOURCES = {"raws"}


def _derive_station_key(source: str, filepath: Path) -> tuple[str, str]:
    """Derive (station_key, source_station_id) from filename for sources with NaN metadata."""
    import re

    stem = filepath.stem
    if source == "snotel":
        m = re.match(r"^(\d+)_", stem)
        station_id = m.group(1) if m else stem
        return f"snotel:{station_id}", station_id
    return f"{source}:{stem}", stem


def _passthrough_daily_file(args: tuple) -> dict[str, object]:
    """Worker: read a daily-native per-station parquet, prepare for Pass 2.

    Unlike _aggregate_one_file (which aggregates hourly→daily), this passes
    through already-daily data with minimal transforms:
    - Drop rows where qc_state == "fail"
    - Ensure station_key/source/source_station_id are populated (derive from filename if NaN)
    - Ensure obs_count exists (default 1)
    - Apply date range filters
    - RAWS-specific: rename rh_max→rhmax, rh_min→rhmin; convert rsds kWh/m²→MJ/m²/day
    """
    pf_str, source, start_date, end_date = args
    pf = Path(pf_str)

    try:
        df = pd.read_parquet(pf)
    except Exception as exc:
        return {
            "daily": None,
            "failure": _build_file_failure_record(source=source, input_file=pf, exc=exc),
        }

    if df.empty or "date" not in df.columns:
        return {"daily": None, "failure": None}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed")

    # Derive station_key from filename when NaN (SNOTEL normalized data has all-NaN metadata)
    station_key, station_id = _derive_station_key(source, pf)
    if "station_key" not in df.columns or df["station_key"].isna().all():
        df["station_key"] = station_key
    if "source" not in df.columns or df["source"].isna().all():
        df["source"] = source
    if "source_station_id" not in df.columns or df["source_station_id"].isna().all():
        df["source_station_id"] = station_id

    # Date range filter
    if start_date is not None:
        df = df[df["date"].dt.date >= start_date]
    if end_date is not None:
        df = df[df["date"].dt.date <= end_date]
    if df.empty:
        return {"daily": None, "failure": None}

    df = _drop_failed_hourly_rows(df)
    if df.empty:
        return {"daily": None, "failure": None}

    if "obs_count" not in df.columns:
        df["obs_count"] = 1
    if "qc_state" not in df.columns:
        df["qc_state"] = "pass"
    if "day_basis" not in df.columns:
        df["day_basis"] = "local"
    if "coverage_flags" not in df.columns:
        df["coverage_flags"] = "n=1,thresh=Y,am=N,pm=N"
    if "qc_rules_version" not in df.columns:
        df["qc_rules_version"] = "0.1.0"

    # RAWS-specific transforms
    if source == "raws":
        if "rh_max" in df.columns and "rhmax" not in df.columns:
            df = df.rename(columns={"rh_max": "rhmax"})
        if "rh_min" in df.columns and "rhmin" not in df.columns:
            df = df.rename(columns={"rh_min": "rhmin"})
        # Convert rsds from kWh/m² to MJ/m²/day (1 kWh = 3.6 MJ)
        if "rsds" in df.columns:
            df["rsds"] = pd.to_numeric(df["rsds"], errors="coerce") * 3.6

    return {"daily": df if not df.empty else None, "failure": None}


def _stage_gdas_hourly_file(args: tuple) -> dict[str, object]:
    """Worker: read one GDAS normalized parquet and stage hourly rows by bucket."""
    pf_str, prov_dict, start_date, end_date, stage_dir_str, n_buckets = args
    pf = Path(pf_str)
    stage_dir = Path(stage_dir_str)

    try:
        df = pd.read_parquet(pf)
    except Exception as exc:
        return {
            "failure": _build_file_failure_record(
                source=str(prov_dict.get("source", "")),
                input_file=pf,
                exc=exc,
            ),
            "bucket_files": 0,
            "rows": 0,
        }

    if df.empty or "station_key" not in df.columns or "datetime_utc" not in df.columns:
        return {"failure": None, "bucket_files": 0, "rows": 0}

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    if start_date is not None:
        df = df[df["datetime_utc"].dt.date >= start_date]
    if end_date is not None:
        df = df[df["datetime_utc"].dt.date <= end_date]
    if df.empty:
        return {"failure": None, "bucket_files": 0, "rows": 0}

    df = _drop_failed_hourly_rows(df)
    if df.empty:
        return {"failure": None, "bucket_files": 0, "rows": 0}

    df = _prepare_gdas_hourly(df)
    if df.empty:
        return {"failure": None, "bucket_files": 0, "rows": 0}

    df["_bucket"] = df["station_key"].apply(_bucket_id, n_buckets=n_buckets)

    bucket_files = 0
    rows = 0
    for bid, grp in df.groupby("_bucket"):
        bucket_dir = stage_dir / f"bucket_{int(bid):03d}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        out_path = bucket_dir / f"{pf.stem}.parquet"
        grp.drop(columns=["_bucket"]).to_parquet(out_path, index=False, compression="snappy")
        bucket_files += 1
        rows += len(grp)

    return {"failure": None, "bucket_files": bucket_files, "rows": rows}


# ------------------------------------------------------------------ #
# Pass 2 worker: process a list of stations from one bucket
# ------------------------------------------------------------------ #


def _process_bucket(args: tuple) -> dict[str, object]:
    """Worker: process all stations in a bucket — QC + write parquets."""
    (
        bucket_df_bytes,
        bucket_id,
        variable_columns,
        out_dir_str,
        source,
        _run_id,
        use_rsun_raster,
        station_coords,
        rsun_path,
        min_por_days,
    ) = args

    out_dir = Path(out_dir_str)
    bucket_df = pd.read_parquet(io.BytesIO(bucket_df_bytes))
    stats: dict[str, int] = {}
    failures: list[dict[str, str | int]] = []
    qc_skips: list[dict[str, str | int]] = []

    for station_key, grp in bucket_df.groupby("station_key"):
        station_key = str(station_key)
        row_count, failure, rs_qc_skip = _process_station_group(
            station_key,
            grp,
            variable_columns=variable_columns,
            out_dir=out_dir,
            source=source,
            bucket_id=bucket_id,
            use_rsun_raster=use_rsun_raster,
            station_coords=station_coords,
            rsun_path=rsun_path,
            min_por_days=min_por_days,
        )
        if failure is not None:
            failures.append(failure)
            continue
        if rs_qc_skip is not None:
            qc_skips.append(rs_qc_skip)
        if row_count is not None:
            stats[station_key] = row_count

    return {"stats": stats, "failures": failures, "qc_skips": qc_skips}


def _process_gdas_bucket(args: tuple) -> dict[str, object]:
    """Worker: stitch GDAS hourly shards for one bucket, aggregate, QC, and write."""
    (
        bucket_dir_str,
        bucket_id,
        prov_dict,
        variable_columns,
        out_dir_str,
        source,
        use_rsun_raster,
        station_coords,
        rsun_path,
        min_por_days,
    ) = args

    bucket_dir = Path(bucket_dir_str)
    out_dir = Path(out_dir_str)
    hourly_files = sorted(bucket_dir.glob("*.parquet"))
    if not hourly_files:
        return {"stats": {}, "failures": [], "qc_skips": []}

    prov = RunProvenance(**prov_dict)
    frames = [pd.read_parquet(pf) for pf in hourly_files]
    hourly = pd.concat(frames, ignore_index=True)
    hourly = _prepare_gdas_hourly(hourly)
    if hourly.empty:
        return {"stats": {}, "failures": [], "qc_skips": []}

    daily = aggregate_daily_wide(hourly, prov)
    if daily.empty:
        return {"stats": {}, "failures": [], "qc_skips": []}

    stats: dict[str, int] = {}
    failures: list[dict[str, str | int]] = []
    qc_skips: list[dict[str, str | int]] = []

    for station_key, grp in daily.groupby("station_key"):
        station_key = str(station_key)
        row_count, failure, rs_qc_skip = _process_station_group(
            station_key,
            grp,
            variable_columns=variable_columns,
            out_dir=out_dir,
            source=source,
            bucket_id=bucket_id,
            use_rsun_raster=use_rsun_raster,
            station_coords=station_coords,
            rsun_path=rsun_path,
            min_por_days=min_por_days,
        )
        if failure is not None:
            failures.append(failure)
            continue
        if rs_qc_skip is not None:
            qc_skips.append(rs_qc_skip)
        if row_count is not None:
            stats[station_key] = row_count

    return {"stats": stats, "failures": failures, "qc_skips": qc_skips}


def _build_station_por_gdas(
    source: str,
    parquet_files: list[Path],
    out_dir: Path,
    provenance: RunProvenance,
    *,
    start_date,
    end_date,
    variable_columns: list[str],
    n_buckets: int,
    station_index_path: Path | str | None,
    rsun_path: Path | str | None,
    min_por_days: int,
    workers: int,
) -> dict[str, int]:
    """Build GDAS station POR using stitched hourly rows across file boundaries."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    use_parallel = workers > 1
    n_files = len(parquet_files)

    prov_dict = {
        "run_id": provenance.run_id,
        "schema_version": provenance.schema_version,
        "qaqc_rules_version": provenance.qaqc_rules_version,
        "crosswalk_version": provenance.crosswalk_version,
        "transform_version": provenance.transform_version,
        "source": provenance.source,
        "command": provenance.command,
    }

    stage_dir = out_dir / _GDAS_STAGE_DIRNAME
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Pass 1 (GDAS): reading %d files, staging hourly rows by bucket (%d workers)",
        n_files,
        workers,
    )

    failures: list[dict[str, str | int]] = []
    qc_skips: list[dict[str, str | int]] = []
    staged_bucket_files = 0
    staged_rows = 0

    if use_parallel:
        task_args = [
            (str(pf), prov_dict, start_date, end_date, str(stage_dir), n_buckets)
            for pf in parquet_files
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_stage_gdas_hourly_file, a): i for i, a in enumerate(task_args)}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                result = future.result()
                failure = result["failure"]
                if failure is not None:
                    failures.append(failure)
                staged_bucket_files += int(result["bucket_files"])
                staged_rows += int(result["rows"])
                if done_count % 500 == 0:
                    logger.info("  GDAS pass 1: %d/%d files done", done_count, n_files)
    else:
        for i, pf in enumerate(parquet_files):
            result = _stage_gdas_hourly_file(
                (str(pf), prov_dict, start_date, end_date, str(stage_dir), n_buckets)
            )
            failure = result["failure"]
            if failure is not None:
                failures.append(failure)
            staged_bucket_files += int(result["bucket_files"])
            staged_rows += int(result["rows"])
            if (i + 1) % 200 == 0:
                logger.info("  GDAS pass 1: %d/%d files", i + 1, n_files)

    if staged_bucket_files == 0:
        shutil.rmtree(stage_dir, ignore_errors=True)
        report_path = _write_failure_report(out_dir, source, provenance.run_id, failures, qc_skips)
        logger.warning("No GDAS hourly data staged from %d files", n_files)
        logger.warning("Station POR report written to %s", report_path)
        return {}

    logger.info(
        "GDAS pass 1 complete: %d bucket files, %d hourly rows staged",
        staged_bucket_files,
        staged_rows,
    )

    use_rsun_raster = rsun_path is not None and station_index_path is not None
    station_coords: dict[str, tuple[float, float]] = {}
    if use_rsun_raster:
        idx_path = Path(station_index_path)
        if idx_path.exists():
            idx_df = pd.read_parquet(idx_path)
            for _, row in idx_df.iterrows():
                key = row.get("canonical_id") or row.get("station_key")
                if key and pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
                    station_coords[str(key)] = (float(row["lon"]), float(row["lat"]))
            logger.info("Loaded %d station coords for RSUN Rso lookup", len(station_coords))
        else:
            use_rsun_raster = False

    logger.info("Pass 2 (GDAS): stitching buckets and applying QC (%d workers)", workers)

    stats: dict[str, int] = {}
    bucket_dirs = [
        stage_dir / f"bucket_{bid:03d}"
        for bid in range(n_buckets)
        if (stage_dir / f"bucket_{bid:03d}").exists()
    ]

    if use_parallel:
        bucket_tasks = [
            (
                str(bucket_dir),
                int(bucket_dir.name.split("_")[-1]),
                prov_dict,
                variable_columns,
                str(out_dir),
                source,
                use_rsun_raster,
                station_coords,
                str(rsun_path) if rsun_path else None,
                min_por_days,
            )
            for bucket_dir in bucket_dirs
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_gdas_bucket, t): i for i, t in enumerate(bucket_tasks)}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                bucket_result = future.result()
                stats.update(bucket_result["stats"])
                failures.extend(bucket_result["failures"])
                qc_skips.extend(bucket_result["qc_skips"])
                if done_count % 10 == 0:
                    logger.info(
                        "  GDAS pass 2: %d/%d buckets done, %d stations, %d failures, %d QC skips",
                        done_count,
                        len(bucket_tasks),
                        len(stats),
                        len(failures),
                        len(qc_skips),
                    )
    else:
        manifest_path = out_dir / "manifest.parquet"
        manifest = Manifest(manifest_path, source=source)

        for bucket_dir in bucket_dirs:
            bucket_result = _process_gdas_bucket(
                (
                    str(bucket_dir),
                    int(bucket_dir.name.split("_")[-1]),
                    prov_dict,
                    variable_columns,
                    str(out_dir),
                    source,
                    use_rsun_raster,
                    station_coords,
                    str(rsun_path) if rsun_path else None,
                    min_por_days,
                )
            )
            stats.update(bucket_result["stats"])
            failures.extend(bucket_result["failures"])
            qc_skips.extend(bucket_result["qc_skips"])
            for sk in bucket_result["stats"]:
                manifest.update(sk, "done", run_id=provenance.run_id)
            if len(stats) and len(stats) % 500 == 0:
                logger.info(
                    "  GDAS pass 2: %d stations written, %d failures, %d QC skips",
                    len(stats),
                    len(failures),
                    len(qc_skips),
                )

        manifest.flush()

    if use_parallel:
        manifest_path = out_dir / "manifest.parquet"
        manifest = Manifest(manifest_path, source=source)
        for sk in stats:
            manifest.update(sk, "done", run_id=provenance.run_id)
        manifest.flush()

    shutil.rmtree(stage_dir, ignore_errors=True)

    report_path = _write_failure_report(out_dir, source, provenance.run_id, failures, qc_skips)
    if failures:
        logger.warning(
            "GDAS station POR complete: %d stations, %d failures, %d QC skips (%s)",
            len(stats),
            len(failures),
            len(qc_skips),
            report_path,
        )
    else:
        logger.info(
            "GDAS station POR complete: %d stations, 0 failures, %d QC skips (%s)",
            len(stats),
            len(qc_skips),
            report_path,
        )
    return stats


def build_station_por(
    source: str,
    norm_dir: Path | str,
    out_dir: Path | str,
    provenance: RunProvenance,
    *,
    start_date=None,
    end_date=None,
    variable_columns: list[str] | None = None,
    n_buckets: int = _N_BUCKETS,
    station_index_path: Path | str | None = None,
    rsun_path: Path | str | None = None,
    min_por_days: int = 0,
    workers: int = 1,
) -> dict[str, int]:
    """Build station POR parquets from normalized data.

    Two-pass approach:
      Pass 1: Read day files (parallel), aggregate to daily, collect in memory.
      Pass 2: Hash-bucket by station, process buckets in parallel (QC + write).

    With workers=1, runs single-threaded (original behavior).
    With workers>1, uses ProcessPoolExecutor for both passes.

    Returns dict of station_key → row_count.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from obsmet.qaqc.engines.pipeline import _VARIABLE_COLUMNS

    norm_dir = Path(norm_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if variable_columns is None:
        variable_columns = _station_por_variable_columns(source, _VARIABLE_COLUMNS.get(source, []))

    parquet_files = sorted(norm_dir.glob("*.parquet"))
    parquet_files = [p for p in parquet_files if p.name != "manifest.parquet"]

    # Pre-filter by filename date when start/end are specified (YYYYMMDD stems)
    if start_date is not None or end_date is not None:
        filtered = []
        for p in parquet_files:
            try:
                file_date = _date(int(p.stem[:4]), int(p.stem[4:6]), int(p.stem[6:8]))
                if start_date is not None and file_date < start_date:
                    continue
                if end_date is not None and file_date > end_date:
                    continue
                filtered.append(p)
            except (ValueError, IndexError):
                filtered.append(p)  # non-date stems pass through unchanged
        parquet_files = filtered

    if not parquet_files:
        logger.warning("No parquet files found in %s", norm_dir)
        return {}

    if source == "gdas":
        return _build_station_por_gdas(
            source,
            parquet_files,
            out_dir,
            provenance,
            start_date=start_date,
            end_date=end_date,
            variable_columns=variable_columns,
            n_buckets=n_buckets,
            station_index_path=station_index_path,
            rsun_path=rsun_path,
            min_por_days=min_por_days,
            workers=workers,
        )

    n_files = len(parquet_files)
    use_parallel = workers > 1
    is_daily_native = source in _DAILY_NATIVE_SOURCES

    # Serialize provenance for worker pickling (hourly sources only)
    prov_dict = {
        "run_id": provenance.run_id,
        "schema_version": provenance.schema_version,
        "qaqc_rules_version": provenance.qaqc_rules_version,
        "crosswalk_version": provenance.crosswalk_version,
        "transform_version": provenance.transform_version,
        "source": provenance.source,
        "command": provenance.command,
    }

    # Select worker function based on source type
    if is_daily_native:
        worker_fn = _passthrough_daily_file

        def _make_args(pf):
            return (str(pf), source, start_date, end_date)
    else:
        worker_fn = _aggregate_one_file

        def _make_args(pf):
            return (str(pf), prov_dict, start_date, end_date)

    # ------------------------------------------------------------------ #
    # Pass 1: read files → daily (aggregate or passthrough) → collect
    # ------------------------------------------------------------------ #
    logger.info(
        "Pass 1: reading %d files%s (%d workers)",
        n_files,
        " (daily passthrough)" if is_daily_native else ", aggregating to daily",
        workers,
    )

    all_daily: list[pd.DataFrame] = []
    failures: list[dict[str, str | int]] = []
    qc_skips: list[dict[str, str | int]] = []

    if use_parallel:
        task_args = [_make_args(pf) for pf in parquet_files]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker_fn, a): i for i, a in enumerate(task_args)}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                result = future.result()
                failure = result["failure"]
                if failure is not None:
                    failures.append(failure)
                daily = result["daily"]
                if daily is not None:
                    all_daily.append(daily)
                if done_count % 500 == 0:
                    logger.info("  pass 1: %d/%d files done", done_count, n_files)
    else:
        for i, pf in enumerate(parquet_files):
            result = worker_fn(_make_args(pf))
            failure = result["failure"]
            if failure is not None:
                failures.append(failure)
            daily = result["daily"]
            if daily is not None:
                all_daily.append(daily)
            if (i + 1) % 200 == 0:
                logger.info("  pass 1: %d/%d files", i + 1, n_files)

    if not all_daily:
        report_path = _write_failure_report(out_dir, source, provenance.run_id, failures, qc_skips)
        logger.warning("No daily data produced from %d files", n_files)
        logger.warning("Station POR report written to %s", report_path)
        return {}

    logger.info("Pass 1 complete: %d files → %d daily chunks", n_files, len(all_daily))

    # Concat all daily data in memory
    daily_all = pd.concat(all_daily, ignore_index=True)
    del all_daily
    logger.info("Total daily rows: %d", len(daily_all))

    # ------------------------------------------------------------------ #
    # Pass 2: hash-bucket → parallel QC + write
    # ------------------------------------------------------------------ #
    logger.info("Pass 2: bucketing %d stations, applying QC (%d workers)", n_buckets, workers)

    # Assign buckets
    daily_all["_bucket"] = daily_all["station_key"].apply(_bucket_id, n_buckets=n_buckets)

    # Rso source setup
    use_rsun_raster = rsun_path is not None and station_index_path is not None
    station_coords: dict[str, tuple[float, float]] = {}
    if use_rsun_raster:
        idx_path = Path(station_index_path)
        if idx_path.exists():
            idx_df = pd.read_parquet(idx_path)
            for _, row in idx_df.iterrows():
                key = row.get("canonical_id") or row.get("station_key")
                if key and pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
                    station_coords[str(key)] = (float(row["lon"]), float(row["lat"]))
            logger.info("Loaded %d station coords for RSUN Rso lookup", len(station_coords))
        else:
            use_rsun_raster = False

    stats: dict[str, int] = {}

    if use_parallel:
        # Serialize each bucket to bytes for worker transfer
        bucket_tasks = []
        for bid in range(n_buckets):
            bucket_df = daily_all[daily_all["_bucket"] == bid].drop(columns=["_bucket"])
            if bucket_df.empty:
                continue
            buf = io.BytesIO()
            bucket_df.to_parquet(buf, index=False)
            bucket_tasks.append(
                (
                    buf.getvalue(),
                    bid,
                    variable_columns,
                    str(out_dir),
                    source,
                    provenance.run_id,
                    use_rsun_raster,
                    station_coords,
                    str(rsun_path) if rsun_path else None,
                    min_por_days,
                )
            )

        del daily_all
        logger.info("Serialized %d non-empty buckets, submitting to pool", len(bucket_tasks))

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_bucket, t): i for i, t in enumerate(bucket_tasks)}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                bucket_result = future.result()
                stats.update(bucket_result["stats"])
                failures.extend(bucket_result["failures"])
                qc_skips.extend(bucket_result["qc_skips"])
                if done_count % 10 == 0:
                    logger.info(
                        "  pass 2: %d/%d buckets done, %d stations, %d failures, %d QC skips",
                        done_count,
                        len(bucket_tasks),
                        len(stats),
                        len(failures),
                        len(qc_skips),
                    )

        del bucket_tasks
    else:
        # Single-threaded: process inline
        manifest_path = out_dir / "manifest.parquet"
        manifest = Manifest(manifest_path, source=source)

        for bid in range(n_buckets):
            bucket_df = daily_all[daily_all["_bucket"] == bid].drop(columns=["_bucket"])
            if bucket_df.empty:
                continue

            for station_key, grp in bucket_df.groupby("station_key"):
                station_key = str(station_key)
                row_count, failure, rs_qc_skip = _process_station_group(
                    station_key,
                    grp,
                    variable_columns=variable_columns,
                    out_dir=out_dir,
                    source=source,
                    bucket_id=bid,
                    use_rsun_raster=use_rsun_raster,
                    station_coords=station_coords,
                    rsun_path=str(rsun_path) if rsun_path else None,
                    min_por_days=min_por_days,
                )
                if failure is not None:
                    failures.append(failure)
                    continue
                if rs_qc_skip is not None:
                    qc_skips.append(rs_qc_skip)
                if row_count is not None:
                    manifest.update(station_key, "done", run_id=provenance.run_id)
                    stats[station_key] = row_count

            if (bid + 1) % 10 == 0:
                logger.info(
                    "  pass 2: %d/%d buckets, %d stations written, %d failures, %d QC skips",
                    bid + 1,
                    n_buckets,
                    len(stats),
                    len(failures),
                    len(qc_skips),
                )

        manifest.flush()
        del daily_all

    # Write manifest for parallel path (workers can't share Manifest object)
    if use_parallel:
        manifest_path = out_dir / "manifest.parquet"
        manifest = Manifest(manifest_path, source=source)
        for sk in stats:
            manifest.update(sk, "done", run_id=provenance.run_id)
        manifest.flush()

    report_path = _write_failure_report(out_dir, source, provenance.run_id, failures, qc_skips)
    if failures:
        logger.warning(
            "Station POR complete: %d stations, %d failures, %d QC skips (%s)",
            len(stats),
            len(failures),
            len(qc_skips),
            report_path,
        )
    else:
        logger.info(
            "Station POR complete: %d stations, 0 failures, %d QC skips (%s)",
            len(stats),
            len(qc_skips),
            report_path,
        )
    return stats
