"""Station period-of-record (POR) pivot product.

Reads normalized hourly parquets, aggregates to daily, applies Tier 2 temporal QC,
and writes one parquet per station.
"""

from __future__ import annotations

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


def _worst_state(a: str, b: str) -> str:
    return a if _STATE_PRECEDENCE.get(a, 0) >= _STATE_PRECEDENCE.get(b, 0) else b


def _apply_tier2_qc(
    station_df: pd.DataFrame,
    variable_columns: list[str],
    *,
    rso: np.ndarray | None = None,
) -> pd.DataFrame:
    """Apply Tier 2 temporal QC to a single station's daily DataFrame.

    Runs MonthlyZScoreRule and StuckSensorRule on each variable column,
    DewpointTemperatureRule cross-variable (td vs tmin),
    RHDriftRule on rhmax/rhmin, and RsPeriodRatioRule on rsds (if Rso provided).
    Merges with existing qc_state (worst wins).

    Parameters
    ----------
    rso : np.ndarray, optional
        365-element RSUN clear-sky array (W/m²) for Rs period-ratio correction.
    """
    zscore_rule = MonthlyZScoreRule()
    stuck_rule = StuckSensorRule(min_run_length=5)  # daily threshold
    td_rule = DewpointTemperatureRule()
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

        # Monthly z-score
        z_states = zscore_rule.check_series(vals, dates)
        for i, (idx, st) in enumerate(z_states.items()):
            if st != "pass":
                tier2_state.iloc[station_df.index.get_loc(idx)] = _worst_state(
                    tier2_state.iloc[station_df.index.get_loc(idx)], st
                )
                pos = station_df.index.get_loc(idx)
                tier2_reasons[pos].append(f"zscore_{col}")

        # Stuck sensor
        s_states = stuck_rule.check_series(vals)
        for idx, st in s_states.items():
            if st != "pass":
                pos = station_df.index.get_loc(idx)
                tier2_state.iloc[pos] = _worst_state(tier2_state.iloc[pos], st)
                tier2_reasons[pos].append(f"stuck_{col}")

    # Dewpoint-temperature daily cross-check
    if "td" in station_df.columns and "tmin" in station_df.columns:
        td_vals = pd.to_numeric(station_df["td"], errors="coerce")
        tmin_vals = pd.to_numeric(station_df["tmin"], errors="coerce")
        dt_states = td_rule.check_daily(td_vals, tmin_vals)
        for idx, st in dt_states.items():
            if st != "pass":
                pos = station_df.index.get_loc(idx)
                tier2_state.iloc[pos] = _worst_state(tier2_state.iloc[pos], st)
                tier2_reasons[pos].append("td_exceeds_tmin_daily")

    # RH drift (yearly percentile correction)
    if "rh" in station_df.columns:
        rhmax = pd.to_numeric(station_df["rh"], errors="coerce")
        rhmin = rhmax.copy()  # MADIS reports single RH; use as both max/min
        years = dates.dt.year
        rh_states = rh_drift_rule.check_series(rhmax, rhmin, years)
        for idx, st in rh_states.items():
            if st != "pass":
                pos = station_df.index.get_loc(idx)
                tier2_state.iloc[pos] = _worst_state(tier2_state.iloc[pos], st)
                tier2_reasons[pos].append("rh_drift")

    # Rs period-ratio correction (requires RSUN Rso)
    if rso is not None and "rsds" in station_df.columns:
        rs_vals = pd.to_numeric(station_df["rsds"], errors="coerce")
        doy = dates.dt.dayofyear
        rs_states = rs_ratio_rule.check_series(rs_vals, rso, doy)
        for idx, st in rs_states.items():
            if st != "pass":
                pos = station_df.index.get_loc(idx)
                tier2_state.iloc[pos] = _worst_state(tier2_state.iloc[pos], st)
                tier2_reasons[pos].append("rs_period_ratio")

    # Merge with existing qc_state
    station_df = station_df.copy()
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
) -> dict[str, int]:
    """Build station POR parquets from normalized data.

    Two-pass bucketed approach to bound memory usage:
      Pass 1: Read each day file, aggregate to daily, spill rows into
              N temp bucket parquets (hash on station_key).
      Pass 2: Read each bucket, groupby station, apply Tier 2 QC,
              write final per-station parquets.

    Peak memory ~ max(one day file, one bucket ≈ total_daily / N).

    Returns dict of station_key → row_count.
    """
    import shutil
    import tempfile

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

    # ------------------------------------------------------------------ #
    # Pass 1: read day files → aggregate to daily → spill to buckets
    # ------------------------------------------------------------------ #
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"station_por_{source}_"))
    logger.info("Pass 1: spilling daily rows to %d buckets in %s", n_buckets, tmp_dir)

    # Accumulate rows per bucket in memory, flush each bucket periodically
    bucket_frames: dict[int, list[pd.DataFrame]] = {i: [] for i in range(n_buckets)}
    bucket_row_counts: dict[int, int] = {i: 0 for i in range(n_buckets)}
    _BUCKET_FLUSH_ROWS = 50_000  # flush a bucket to disk when it exceeds this many rows
    n_files = len(parquet_files)

    for i, pf in enumerate(parquet_files):
        try:
            df = pd.read_parquet(pf)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", pf, exc)
            continue

        if df.empty:
            continue

        # Filter by date range
        if "datetime_utc" in df.columns:
            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
            if start_date is not None:
                df = df[df["datetime_utc"].dt.date >= start_date]
            if end_date is not None:
                df = df[df["datetime_utc"].dt.date <= end_date]
            if df.empty:
                continue

        if "station_key" not in df.columns:
            continue

        # Aggregate this file's hourly data to daily
        daily = aggregate_daily_wide(df, provenance)
        if daily.empty:
            continue

        # Distribute daily rows to hash buckets
        daily["_bucket"] = daily["station_key"].apply(_bucket_id, n_buckets=n_buckets)
        for bid, grp in daily.groupby("_bucket"):
            grp = grp.drop(columns=["_bucket"])
            bucket_frames[bid].append(grp)
            bucket_row_counts[bid] += len(grp)

            # Flush bucket to disk if it's grown large
            if bucket_row_counts[bid] >= _BUCKET_FLUSH_ROWS:
                _spill_bucket(bid, bucket_frames[bid], tmp_dir)
                bucket_frames[bid] = []
                bucket_row_counts[bid] = 0

        if (i + 1) % 200 == 0:
            logger.info("  pass 1: %d/%d files", i + 1, n_files)

    # Final spill of remaining in-memory bucket data
    for bid in range(n_buckets):
        if bucket_frames[bid]:
            _spill_bucket(bid, bucket_frames[bid], tmp_dir)
    del bucket_frames, bucket_row_counts

    logger.info("Pass 1 complete: %d files processed", n_files)

    # ------------------------------------------------------------------ #
    # Pass 2: read each bucket → groupby station → QC → write final
    # ------------------------------------------------------------------ #
    logger.info("Pass 2: reading buckets, applying QC, writing per-station parquets")

    manifest_path = out_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source=source)
    stats: dict[str, int] = {}

    for bid in range(n_buckets):
        bucket_files = sorted(tmp_dir.glob(f"bucket_{bid:04d}_*.parquet"))
        if not bucket_files:
            continue

        frames = []
        for bf in bucket_files:
            try:
                frames.append(pd.read_parquet(bf))
            except Exception as exc:
                logger.warning("Failed to read bucket file %s: %s", bf, exc)

        if not frames:
            continue

        bucket_df = pd.concat(frames, ignore_index=True)
        del frames

        for station_key, grp in bucket_df.groupby("station_key"):
            station_key = str(station_key)
            grp = grp.sort_values("date").reset_index(drop=True)

            # Deduplicate in case of overlapping day files
            if "date" in grp.columns:
                grp = grp.drop_duplicates(subset=["date"], keep="first")

            # Apply Tier 2 QC
            grp = _apply_tier2_qc(grp, variable_columns)

            safe_name = station_key.replace(":", "_").replace("/", "_")
            out_path = out_dir / f"{safe_name}.parquet"
            grp.to_parquet(out_path, index=False, compression="snappy")
            manifest.update(station_key, "done", run_id=provenance.run_id)
            stats[station_key] = len(grp)

        del bucket_df

        if (bid + 1) % 10 == 0:
            logger.info(
                "  pass 2: %d/%d buckets, %d stations written", bid + 1, n_buckets, len(stats)
            )

    manifest.flush()

    # Clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Station POR complete: %d stations", len(stats))

    return stats


def _spill_bucket(bucket_id: int, frames: list[pd.DataFrame], tmp_dir: Path) -> None:
    """Write accumulated frames for one bucket to a temp parquet file."""
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    # Use a sequence number to allow multiple spills per bucket
    existing = list(tmp_dir.glob(f"bucket_{bucket_id:04d}_*.parquet"))
    seq = len(existing)
    out = tmp_dir / f"bucket_{bucket_id:04d}_{seq:04d}.parquet"
    df.to_parquet(out, index=False, compression="snappy")
