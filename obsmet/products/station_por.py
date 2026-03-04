"""Station period-of-record (POR) pivot product.

Reads normalized hourly parquets, aggregates to daily, applies Tier 2 temporal QC,
and writes one parquet per station.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from obsmet.core.manifest import Manifest
from obsmet.core.provenance import RunProvenance
from obsmet.core.time_policy import aggregate_daily_wide
from obsmet.qaqc.rules.temporal import (
    DewpointTemperatureRule,
    MonthlyZScoreRule,
    StuckSensorRule,
)

logger = logging.getLogger(__name__)

_STATE_PRECEDENCE = {"fail": 2, "suspect": 1, "pass": 0}


def _worst_state(a: str, b: str) -> str:
    return a if _STATE_PRECEDENCE.get(a, 0) >= _STATE_PRECEDENCE.get(b, 0) else b


def _apply_tier2_qc(
    station_df: pd.DataFrame,
    variable_columns: list[str],
) -> pd.DataFrame:
    """Apply Tier 2 temporal QC to a single station's daily DataFrame.

    Runs MonthlyZScoreRule and StuckSensorRule on each variable column,
    plus DewpointTemperatureRule cross-variable (td vs tmin).
    Merges with existing qc_state (worst wins).
    """
    zscore_rule = MonthlyZScoreRule()
    stuck_rule = StuckSensorRule(min_run_length=5)  # daily threshold
    td_rule = DewpointTemperatureRule()

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


def build_station_por(
    source: str,
    norm_dir: Path | str,
    out_dir: Path | str,
    provenance: RunProvenance,
    *,
    start_date=None,
    end_date=None,
    variable_columns: list[str] | None = None,
) -> dict[str, int]:
    """Build station POR parquets from normalized hourly data.

    Reads all parquets from norm_dir, aggregates to daily, applies Tier 2 QC,
    writes one parquet per station to out_dir.

    Returns dict of station_key → row_count.
    """
    from obsmet.qaqc.engines.pipeline import _VARIABLE_COLUMNS

    norm_dir = Path(norm_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if variable_columns is None:
        variable_columns = _VARIABLE_COLUMNS.get(source, [])

    # Read all normalized parquets
    parquet_files = sorted(norm_dir.glob("*.parquet"))
    parquet_files = [p for p in parquet_files if p.name != "manifest.parquet"]

    if not parquet_files:
        logger.warning("No parquet files found in %s", norm_dir)
        return {}

    frames = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            frames.append(df)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", pf, exc)

    if not frames:
        return {}

    all_data = pd.concat(frames, ignore_index=True)

    # Filter by date range
    if "datetime_utc" in all_data.columns:
        all_data["datetime_utc"] = pd.to_datetime(all_data["datetime_utc"], utc=True)
        if start_date is not None:
            all_data = all_data[all_data["datetime_utc"].dt.date >= start_date]
        if end_date is not None:
            all_data = all_data[all_data["datetime_utc"].dt.date <= end_date]

    if all_data.empty or "station_key" not in all_data.columns:
        return {}

    # Aggregate to daily
    daily = aggregate_daily_wide(all_data, provenance)
    if daily.empty:
        return {}

    # Manifest for resume
    manifest_path = out_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source=source)

    stats: dict[str, int] = {}
    for station_key, grp in daily.groupby("station_key"):
        station_key = str(station_key)
        grp = grp.sort_values("date").reset_index(drop=True)

        # Apply Tier 2 QC
        grp = _apply_tier2_qc(grp, variable_columns)

        out_path = out_dir / f"{station_key}.parquet"
        grp.to_parquet(out_path, index=False, compression="snappy")
        manifest.update(station_key, "done", run_id=provenance.run_id)
        stats[station_key] = len(grp)

    manifest.flush()
    return stats
