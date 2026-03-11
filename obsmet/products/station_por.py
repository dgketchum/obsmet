"""Station period-of-record (POR) pivot product.

Reads normalized hourly parquets, aggregates to daily, applies Tier 2 temporal QC,
and writes one parquet per station.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from obsmet.core.manifest import Manifest
from obsmet.core.provenance import RunProvenance
from obsmet.core.time_policy import aggregate_daily_wide
from obsmet.qaqc.rules.temporal import (
    DewpointTemperatureRule,
    MonthlyZScoreRule,
    RHDriftRule,
    RsPeriodRatioRule,
    StuckSensorRule,
)

logger = logging.getLogger(__name__)

_STATE_PRECEDENCE = {"fail": 2, "suspect": 1, "pass": 0}

# Variables excluded from z-score (zero-inflated / skewed distributions)
_ZSCORE_SKIP_VARS = {"prcp"}

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

    Runs MonthlyZScoreRule (excluding precip) and StuckSensorRule on each
    variable column, DewpointTemperatureRule cross-variable (td vs tmin,
    gated on obs_count >= 18), RHDriftRule on rhmax/rhmin with correction,
    RsPeriodRatioRule on rsds with correction (if Rso provided), and a
    physical upper bound on daily precipitation.

    Merges with existing qc_state (worst wins).

    Parameters
    ----------
    rso : np.ndarray, optional
        365-element clear-sky array (MJ/m²/day, same units as rsds) for
        Rs period-ratio correction.  From RSUN raster or ASCE flat-earth.
    """
    zscore_rule = MonthlyZScoreRule()
    stuck_rule = StuckSensorRule(min_run_length=5)  # daily threshold
    td_rule = DewpointTemperatureRule(tolerance=2.0)
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

        # Monthly z-score (skip zero-inflated variables like precip)
        if col not in _ZSCORE_SKIP_VARS:
            z_states = zscore_rule.check_series(vals, dates)
            for i, (idx, st) in enumerate(z_states.items()):
                if st != "pass":
                    pos = station_df.index.get_loc(idx)
                    tier2_state.iloc[pos] = _worst_state(tier2_state.iloc[pos], st)
                    tier2_reasons[pos].append(f"zscore_{col}")

        # Stuck sensor (all variables)
        s_states = stuck_rule.check_series(vals)
        for idx, st in s_states.items():
            if st != "pass":
                pos = station_df.index.get_loc(idx)
                tier2_state.iloc[pos] = _worst_state(tier2_state.iloc[pos], st)
                tier2_reasons[pos].append(f"stuck_{col}")

    # Daily precip upper bound
    if "prcp" in station_df.columns:
        prcp_vals = pd.to_numeric(station_df["prcp"], errors="coerce")
        prcp_extreme = prcp_vals > _DAILY_PRCP_MAX_MM
        for idx in station_df.index[prcp_extreme]:
            pos = station_df.index.get_loc(idx)
            tier2_state.iloc[pos] = _worst_state(tier2_state.iloc[pos], "fail")
            tier2_reasons[pos].append("prcp_exceeds_daily_max")

    # Dewpoint-temperature daily cross-check (gated on obs_count >= 18)
    if "td" in station_df.columns and "tmin" in station_df.columns:
        td_vals = pd.to_numeric(station_df["td"], errors="coerce")
        tmin_vals = pd.to_numeric(station_df["tmin"], errors="coerce")
        obs_count = pd.to_numeric(
            station_df.get("obs_count", pd.Series(24, index=station_df.index)),
            errors="coerce",
        )
        # Mask low-coverage days so check_daily sees NaN → returns "pass"
        coverage_ok = obs_count >= 18
        masked_td = td_vals.where(coverage_ok)
        masked_tmin = tmin_vals.where(coverage_ok)
        dt_states = td_rule.check_daily(masked_td, masked_tmin)
        for idx, st in dt_states.items():
            if st != "pass":
                pos = station_df.index.get_loc(idx)
                tier2_state.iloc[pos] = _worst_state(tier2_state.iloc[pos], st)
                tier2_reasons[pos].append("td_exceeds_tmin_daily")

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
        rh_states, corr_rhmax, corr_rhmin, _ = rh_drift_rule.correct_series(rhmax, rhmin, years)
        for idx, st in rh_states.items():
            if st != "pass":
                pos = station_df.index.get_loc(idx)
                tier2_state.iloc[pos] = _worst_state(tier2_state.iloc[pos], st)
                tier2_reasons[pos].append("rh_drift")

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


def _bucket_id(station_key: str, n_buckets: int = _N_BUCKETS) -> int:
    """Deterministic hash bucket for a station key."""
    return hash(station_key) % n_buckets


# ------------------------------------------------------------------ #
# Pass 1 worker: read one day file, aggregate to daily
# ------------------------------------------------------------------ #


def _aggregate_one_file(args: tuple) -> pd.DataFrame | None:
    """Worker: read a normalized parquet, aggregate to daily, return DataFrame."""
    pf_str, prov_dict, start_date, end_date = args
    pf = Path(pf_str)

    # Reconstruct provenance (can't pickle RunProvenance across processes easily)
    prov = RunProvenance(**prov_dict)

    try:
        df = pd.read_parquet(pf)
    except Exception:
        return None

    if df.empty or "station_key" not in df.columns:
        return None

    if "datetime_utc" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
        if start_date is not None:
            df = df[df["datetime_utc"].dt.date >= start_date]
        if end_date is not None:
            df = df[df["datetime_utc"].dt.date <= end_date]
        if df.empty:
            return None

    daily = aggregate_daily_wide(df, prov)
    return daily if not daily.empty else None


# ------------------------------------------------------------------ #
# Pass 2 worker: process a list of stations from one bucket
# ------------------------------------------------------------------ #


def _process_bucket(args: tuple) -> dict[str, int]:
    """Worker: process all stations in a bucket — QC + write parquets."""
    (
        bucket_df_bytes,
        variable_columns,
        out_dir_str,
        source,
        run_id,
        use_rsun_raster,
        station_coords,
        rsun_path,
        min_por_days,
    ) = args

    out_dir = Path(out_dir_str)
    bucket_df = pd.read_parquet(io.BytesIO(bucket_df_bytes))
    stats: dict[str, int] = {}

    for station_key, grp in bucket_df.groupby("station_key"):
        station_key = str(station_key)
        grp = grp.sort_values("date").reset_index(drop=True)

        if "date" in grp.columns:
            grp = grp.drop_duplicates(subset=["date"], keep="first")

        # Minimum POR filter
        if min_por_days > 0:
            obs_count = pd.to_numeric(grp.get("obs_count", pd.Series(dtype=float)), errors="coerce")
            sufficient_days = int((obs_count >= 18).sum())
            if sufficient_days < min_por_days:
                continue

        # Compute Rso for Rs QC
        rso = None
        if use_rsun_raster and station_key in station_coords:
            lon, lat = station_coords[station_key]
            try:
                from obsmet.products.rsun import extract_station_rsun

                rso = extract_station_rsun(lon, lat, str(rsun_path))
            except Exception:
                pass

        if rso is None and "lat" in grp.columns and "elev_m" in grp.columns:
            lat_val = pd.to_numeric(grp["lat"], errors="coerce").dropna()
            elev_val = pd.to_numeric(grp["elev_m"], errors="coerce").dropna()
            if not lat_val.empty and not elev_val.empty:
                try:
                    from obsmet.products.rsun import compute_rso_asce

                    rso = compute_rso_asce(float(lat_val.iloc[0]), float(elev_val.iloc[0]))
                except Exception:
                    pass

        grp = _apply_tier2_qc(grp, variable_columns, rso=rso)

        safe_name = station_key.replace(":", "_").replace("/", "_")
        out_path = out_dir / f"{safe_name}.parquet"
        grp.to_parquet(out_path, index=False, compression="snappy")
        stats[station_key] = len(grp)

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
        variable_columns = _VARIABLE_COLUMNS.get(source, [])

    parquet_files = sorted(norm_dir.glob("*.parquet"))
    parquet_files = [p for p in parquet_files if p.name != "manifest.parquet"]

    if not parquet_files:
        logger.warning("No parquet files found in %s", norm_dir)
        return {}

    n_files = len(parquet_files)
    use_parallel = workers > 1

    # Serialize provenance for worker pickling
    prov_dict = {
        "run_id": provenance.run_id,
        "schema_version": provenance.schema_version,
        "qaqc_rules_version": provenance.qaqc_rules_version,
        "crosswalk_version": provenance.crosswalk_version,
        "transform_version": provenance.transform_version,
        "source": provenance.source,
        "command": provenance.command,
    }

    # ------------------------------------------------------------------ #
    # Pass 1: read day files → aggregate to daily → collect in memory
    # ------------------------------------------------------------------ #
    logger.info(
        "Pass 1: reading %d files, aggregating to daily (%d workers)",
        n_files,
        workers,
    )

    all_daily: list[pd.DataFrame] = []

    if use_parallel:
        task_args = [(str(pf), prov_dict, start_date, end_date) for pf in parquet_files]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_aggregate_one_file, a): i for i, a in enumerate(task_args)}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                result = future.result()
                if result is not None:
                    all_daily.append(result)
                if done_count % 500 == 0:
                    logger.info("  pass 1: %d/%d files done", done_count, n_files)
    else:
        for i, pf in enumerate(parquet_files):
            result = _aggregate_one_file((str(pf), prov_dict, start_date, end_date))
            if result is not None:
                all_daily.append(result)
            if (i + 1) % 200 == 0:
                logger.info("  pass 1: %d/%d files", i + 1, n_files)

    if not all_daily:
        logger.warning("No daily data produced from %d files", n_files)
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
                bucket_stats = future.result()
                stats.update(bucket_stats)
                if done_count % 10 == 0:
                    logger.info(
                        "  pass 2: %d/%d buckets done, %d stations",
                        done_count,
                        len(bucket_tasks),
                        len(stats),
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
                grp = grp.sort_values("date").reset_index(drop=True)

                if "date" in grp.columns:
                    grp = grp.drop_duplicates(subset=["date"], keep="first")

                if min_por_days > 0:
                    obs_count = pd.to_numeric(
                        grp.get("obs_count", pd.Series(dtype=float)), errors="coerce"
                    )
                    sufficient_days = int((obs_count >= 18).sum())
                    if sufficient_days < min_por_days:
                        continue

                rso = None
                if use_rsun_raster and station_key in station_coords:
                    lon, lat = station_coords[station_key]
                    try:
                        from obsmet.products.rsun import extract_station_rsun

                        rso = extract_station_rsun(lon, lat, str(rsun_path))
                    except Exception:
                        pass

                if rso is None and "lat" in grp.columns and "elev_m" in grp.columns:
                    lat_val = pd.to_numeric(grp["lat"], errors="coerce").dropna()
                    elev_val = pd.to_numeric(grp["elev_m"], errors="coerce").dropna()
                    if not lat_val.empty and not elev_val.empty:
                        try:
                            from obsmet.products.rsun import compute_rso_asce

                            rso = compute_rso_asce(float(lat_val.iloc[0]), float(elev_val.iloc[0]))
                        except Exception:
                            pass

                grp = _apply_tier2_qc(grp, variable_columns, rso=rso)

                safe_name = station_key.replace(":", "_").replace("/", "_")
                out_path = out_dir / f"{safe_name}.parquet"
                grp.to_parquet(out_path, index=False, compression="snappy")
                manifest.update(station_key, "done", run_id=provenance.run_id)
                stats[station_key] = len(grp)

            if (bid + 1) % 10 == 0:
                logger.info(
                    "  pass 2: %d/%d buckets, %d stations written",
                    bid + 1,
                    n_buckets,
                    len(stats),
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

    logger.info("Station POR complete: %d stations", len(stats))
    return stats
