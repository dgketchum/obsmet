"""Validation of obsmet MADIS QC against the CONUS-AgWeather v1 data release.

Workflow:
  1. parse_agweather_log  — extract per-variable correction decisions from each log
  2. load_agweather_excel — standardize agweather daily data to obsmet units/names
  3. build_comparison_dataset — join agweather to station_por for all matched pairs
  4. flag_agreement_summary — confusion matrix per Tier-2 rule
  5. correction_comparison   — Rs/RH correction magnitude comparison
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MATCHES_CSV = Path("/nas/climate/agweather/agweather_madis_matches.csv")
DEFAULT_AGW_BASE = Path("/nas/climate/agweather/CONUS-AgWeather_v1")
DEFAULT_POR_BASE = Path("/mnt/mco_nas1/shared/obsmet/products/station_por/madis")
DEFAULT_OUT_DIR = Path("/nas/climate/agweather")

# W/m² (daily mean) → MJ/m²/day
RS_CONV = 86400 / 1e6  # = 0.0864

# Map our qc_reason_codes to the agweather variable(s) they govern
RULE_TO_AGW_VARS: dict[str, list[str]] = {
    # zscore_tmax/tmin/tmean omitted — we call agweather's own modified_z_score_outlier_detection
    # directly, so the algorithm is identical. Disagreement is driven by agweather's manual
    # correction intervals (sensor failure periods) layered on top of their z-score, not by
    # algorithmic differences. Comparison against their final output is not meaningful.
    "zscore_td": ["td"],
    "zscore_rh": ["rh", "rhmax", "rhmin"],
    "zscore_prcp": ["prcp"],
    "zscore_wind": ["wind"],
    "td_exceeds_tmin_daily": ["td", "tmin"],
    "stuck_sensor": ["tmax", "tmin", "td", "rh", "wind", "prcp", "rsds"],
    "rs_period_ratio": ["rsds"],
}

ALL_RULES = list(RULE_TO_AGW_VARS.keys())

# Agweather → obsmet column mapping (after unit conversion)
AGW_COL_MAP = {
    "TMax (C)": "tmax",
    "TAvg (C)": "tmean",
    "TMin (C)": "tmin",
    "TDew (C)": "td",
    "RHMax (%)": "rhmax",
    "RHAvg (%)": "rh",
    "RHMin (%)": "rhmin",
    "Rs (w/m2)": "rsds",
    "Optimized TR Rs (w/m2)": "rsds_corrected",
    "Uz at 2m (m/s)": "wind",
    "Precipitation (mm)": "prcp",
}

# obsmet variables we include in comparison
COMPARE_VARS = [
    "tmax",
    "tmin",
    "tmean",
    "td",
    "rh",
    "rhmax",
    "rhmin",
    "rsds",
    "rsds_corrected",
    "wind",
    "prcp",
]


# ---------------------------------------------------------------------------
# 1. Log parser
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"Now correcting (.+?) (?:and|at) ", re.IGNORECASE)
_ZSCORE_RE = re.compile(r"(\d+) outliers were removed on variable (\w+)")
_INTERVAL_RE = re.compile(r"Selected correction interval started at (\d+) and ended at (\d+)")
_NULL_RE = re.compile(r"Observations within the interval were set to nan")
_RS_PARAMS_RE = re.compile(r"period length was (\d+), and correction sample size was (\d+)")
_RS_DESPIKE_RE = re.compile(r"(\d+) data points were removed as part of the despiking process")
_RS_CLIP_RE = re.compile(r"(\d+) Rs data points were clipped to")
_RH_CORR_RE = re.compile(
    r"Year-based RH correction.*RHMax had (\d+) points exceed 100.*RHMin had (\d+) points",
    re.IGNORECASE,
)
_MISSING_DATES_RE = re.compile(r"had (\d+) missing date entries")


def parse_agweather_log(log_path: Path) -> dict:
    """Parse an agweather-qaqc log file into a structured summary dict.

    Returns
    -------
    dict with keys:
        missing_dates: int
        zscore_removals: dict[variable_label, int]
        manual_nulled_intervals: list[dict] with keys variable, start, end
        rs_period: int | None
        rs_sample_size: int | None
        rs_despiked: int
        rs_clipped: int
        rh_rhmax_exceed_100: int
        rh_rhmin_exceed_100: int
    """
    text = log_path.read_text(errors="replace")
    lines = text.splitlines()

    result: dict = {
        "missing_dates": 0,
        "zscore_removals": {},
        "manual_nulled_intervals": [],
        "rs_period": None,
        "rs_sample_size": None,
        "rs_despiked": 0,
        "rs_clipped": 0,
        "rh_rhmax_exceed_100": 0,
        "rh_rhmin_exceed_100": 0,
    }

    m = _MISSING_DATES_RE.search(text)
    if m:
        result["missing_dates"] = int(m.group(1))

    current_section = "unknown"
    pending_intervals: list[tuple[int, int]] = []

    for line in lines:
        sec_m = _SECTION_RE.search(line)
        if sec_m:
            current_section = sec_m.group(1).strip()
            pending_intervals = []
            continue

        z = _ZSCORE_RE.search(line)
        if z:
            count, var = int(z.group(1)), z.group(2)
            result["zscore_removals"][var] = result["zscore_removals"].get(var, 0) + count
            continue

        iv = _INTERVAL_RE.search(line)
        if iv:
            pending_intervals.append((int(iv.group(1)), int(iv.group(2))))
            continue

        if _NULL_RE.search(line) and pending_intervals:
            for start, end in pending_intervals:
                result["manual_nulled_intervals"].append(
                    {"variable": current_section, "start": start, "end": end}
                )
            pending_intervals = []
            continue

        rs_p = _RS_PARAMS_RE.search(line)
        if rs_p:
            result["rs_period"] = int(rs_p.group(1))
            result["rs_sample_size"] = int(rs_p.group(2))
            continue

        ds = _RS_DESPIKE_RE.search(line)
        if ds:
            result["rs_despiked"] += int(ds.group(1))
            continue

        cl = _RS_CLIP_RE.search(line)
        if cl:
            result["rs_clipped"] += int(cl.group(1))
            continue

        rh = _RH_CORR_RE.search(line)
        if rh:
            result["rh_rhmax_exceed_100"] += int(rh.group(1))
            result["rh_rhmin_exceed_100"] += int(rh.group(2))

    return result


# ---------------------------------------------------------------------------
# 2. Load agweather Excel
# ---------------------------------------------------------------------------


def load_agweather_excel(xlsx_path: Path) -> pd.DataFrame:
    """Load one agweather station Excel, standardize columns and units.

    Rs/Rso columns are converted from W/m² (daily mean) to MJ/m²/day.
    Returns a DataFrame indexed by date with obsmet-style column names.
    """
    df = pd.read_excel(xlsx_path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    rename = {k: v for k, v in AGW_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Unit conversion: W/m² → MJ/m²/day
    for col in ("rsds", "rsds_corrected"):
        if col in df.columns:
            df[col] = df[col] * RS_CONV

    # Keep only columns we care about
    keep = ["date"] + [c for c in COMPARE_VARS if c in df.columns]
    df = df[keep].copy()
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    return df


# ---------------------------------------------------------------------------
# 3. Load station_por
# ---------------------------------------------------------------------------


def load_station_por(por_path: Path) -> pd.DataFrame:
    """Load a MADIS station_por parquet and return indexed by date."""
    df = pd.read_parquet(por_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    return df


# ---------------------------------------------------------------------------
# 4. Compare one pair
# ---------------------------------------------------------------------------


def compare_pair(
    agw_df: pd.DataFrame,
    por_df: pd.DataFrame,
    *,
    agw_station: str,
    madis_key: str,
) -> pd.DataFrame:
    """Join agweather and station_por on date and compute per-variable agreement.

    Returns a tidy DataFrame with one row per (date × variable) in the overlap
    window. All operations are vectorised — no Python-level date loops.

    Columns:
        agw_station, madis_key, date, variable,
        agw_value, our_value,
        agw_null     — agweather has NaN (their QC removed it or data absent)
        our_missing  — our value is NaN (input missing, not QC flag)
        our_flagged  — our qc_state is fail or suspect
        our_qc_state — raw qc_state string
        our_reasons  — full qc_reason_codes string
        var_reasons  — subset of our_reasons that govern this variable
    """
    overlap = agw_df.index.intersection(por_df.index)
    if len(overlap) == 0:
        return pd.DataFrame()

    agw = agw_df.loc[overlap]
    por = por_df.loc[overlap]

    por_state = (
        por["qc_state"].fillna("pass")
        if "qc_state" in por.columns
        else pd.Series("pass", index=overlap)
    )
    por_reasons = (
        por["qc_reason_codes"].fillna("")
        if "qc_reason_codes" in por.columns
        else pd.Series("", index=overlap)
    )
    our_flagged_mask = por_state.isin(["fail", "suspect"])

    # Build a lookup: rule → frozenset of variables it governs
    rule_var_sets: dict[str, frozenset] = {r: frozenset(vs) for r, vs in RULE_TO_AGW_VARS.items()}

    var_frames: list[pd.DataFrame] = []
    for var in COMPARE_VARS:
        agw_vals = agw[var] if var in agw.columns else pd.Series(np.nan, index=overlap)
        por_vals = por[var] if var in por.columns else pd.Series(np.nan, index=overlap)

        agw_null = agw_vals.isna()
        our_missing = por_vals.isna()

        # Vectorised var_reasons: for each row, keep only reason codes that govern var
        def _filter_reasons(reasons_str: str) -> str:
            codes = [c.strip() for c in reasons_str.split(",") if c.strip()]
            relevant = [c for c in codes if var in rule_var_sets.get(c, frozenset())]
            return ",".join(relevant)

        var_reasons = por_reasons.map(_filter_reasons)

        frame = pd.DataFrame(
            {
                "agw_station": agw_station,
                "madis_key": madis_key,
                "date": overlap,
                "variable": var,
                "agw_value": agw_vals.values,
                "our_value": por_vals.values,
                "agw_null": agw_null.values,
                "our_missing": our_missing.values,
                "our_flagged": our_flagged_mask.values,
                "our_qc_state": por_state.values,
                "our_reasons": por_reasons.values,
                "var_reasons": var_reasons.values,
            }
        )
        var_frames.append(frame)

    return pd.concat(var_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Build full comparison dataset
# ---------------------------------------------------------------------------


def _por_date_range(por_path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Read only the date column of a station_por parquet to get POR bounds."""
    try:
        df = pd.read_parquet(por_path, columns=["date"])
        dates = pd.to_datetime(df["date"]).dropna()
        if dates.empty:
            return None, None
        return dates.min(), dates.max()
    except Exception:
        return None, None


def _overlap_days_from_rows(
    row: pd.Series,
    por_start: pd.Timestamp | None,
    por_end: pd.Timestamp | None,
) -> int:
    """Compute overlap days between a matches-table row and station_por POR."""
    if por_start is None or por_end is None:
        return 0
    try:
        agw_start = pd.Timestamp(row["agw_start"])
        agw_end = pd.Timestamp(row["agw_end"])
    except Exception:
        return 0
    start = max(por_start, agw_start)
    end = min(por_end, agw_end)
    return max(0, (end - start).days)


def build_comparison_dataset(
    matches_csv: Path = DEFAULT_MATCHES_CSV,
    agw_base: Path = DEFAULT_AGW_BASE,
    por_base: Path = DEFAULT_POR_BASE,
    out_path: Path | None = None,
    *,
    max_dist_m: float = 500.0,
    max_elev_diff_m: float = 50.0,
    min_overlap_days: int = 365,
    limit: int | None = None,
) -> pd.DataFrame:
    """Build the full paired comparison dataset for all high-confidence matches.

    Parameters
    ----------
    matches_csv      : path to agweather_madis_matches.csv
    agw_base         : root of CONUS-AgWeather_v1 (standardized_data/ and log_files/)
    por_base         : root of madis station_por parquet files
    out_path         : if set, write comparison parquet here
    max_dist_m       : maximum distance filter (default 500 m)
    max_elev_diff_m  : maximum elevation difference filter (default 50 m)
    min_overlap_days : skip pairs with fewer overlapping days (default 365)
    limit            : if set, process only the first N qualifying pairs (for testing)
    """
    matches = pd.read_csv(matches_csv)
    matches["elev_diff"] = (matches["agw_elev_m"] - matches["madis_elev_m"]).abs()
    matches = matches[
        (matches["dist_m"] <= max_dist_m) & (matches["elev_diff"] <= max_elev_diff_m)
    ].reset_index(drop=True)

    # Pre-filter by overlap: read only date columns from station_por
    logger.info(
        "Pre-filtering %d pairs by overlap (min %d days) ...", len(matches), min_overlap_days
    )
    qualifying: list[int] = []
    for i, row in matches.iterrows():
        sid = str(row["madis_key"]).replace("madis:", "")
        por_path = por_base / f"madis_{sid}.parquet"
        if not por_path.exists():
            continue
        por_start, por_end = _por_date_range(por_path)
        days = _overlap_days_from_rows(row, por_start, por_end)
        if days >= min_overlap_days:
            qualifying.append(i)

    matches = matches.loc[qualifying].reset_index(drop=True)
    logger.info("Qualifying pairs after overlap filter: %d", len(matches))

    # Sort by overlap descending so best pairs run first
    matches = matches.sort_values("dist_m").reset_index(drop=True)

    if limit is not None:
        matches = matches.head(limit)

    logger.info("Building comparison dataset: %d station pairs", len(matches))

    xlsx_dir = agw_base / "standardized_data"
    log_dir = agw_base / "log_files"
    frames: list[pd.DataFrame] = []
    log_records: list[dict] = []
    skipped = 0
    n_total = len(matches)

    for idx, row in matches.iterrows():
        agw_station = row["agw_station"]
        madis_key = str(row["madis_key"])
        station_id = madis_key.replace("madis:", "")

        xlsx_path = xlsx_dir / f"{agw_station}_data.xlsx"
        por_path = por_base / f"madis_{station_id}.parquet"
        log_path = log_dir / f"{agw_station}_qaqc_log.txt"

        if not xlsx_path.exists():
            logger.warning("Missing agweather xlsx: %s", xlsx_path)
            skipped += 1
            continue
        if not por_path.exists():
            logger.warning("Missing station_por: %s", por_path)
            skipped += 1
            continue

        if (idx + 1) % 50 == 0 or idx == 0:
            logger.info("  %d/%d — %s ↔ %s", idx + 1, n_total, agw_station, madis_key)

        try:
            agw_df = load_agweather_excel(xlsx_path)
            por_df = load_station_por(por_path)
            pair_df = compare_pair(agw_df, por_df, agw_station=agw_station, madis_key=madis_key)
        except Exception as exc:
            logger.warning("Failed to compare %s ↔ %s: %s", agw_station, madis_key, exc)
            skipped += 1
            continue

        if pair_df.empty:
            logger.info("  No overlapping dates: %s ↔ %s", agw_station, madis_key)
            skipped += 1
            continue

        frames.append(pair_df)

        if log_path.exists():
            try:
                log_parsed = parse_agweather_log(log_path)
                log_parsed["agw_station"] = agw_station
                log_parsed["madis_key"] = madis_key
                log_records.append(log_parsed)
            except Exception as exc:
                logger.warning("Log parse failed %s: %s", log_path.name, exc)

    logger.info("Comparison complete: %d pairs built, %d skipped", len(frames), skipped)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path, index=False)
        logger.info("Wrote comparison dataset: %s (%d rows)", out_path, len(result))

        # Also write parsed logs
        if log_records:
            log_df = _flatten_log_records(log_records)
            log_out = out_path.parent / "agweather_log_summary.parquet"
            log_df.to_parquet(log_out, index=False)
            logger.info("Wrote log summary: %s (%d stations)", log_out, len(log_df))

    return result


def _flatten_log_records(records: list[dict]) -> pd.DataFrame:
    """Flatten log parse results into a tabular DataFrame."""
    rows = []
    for rec in records:
        row = {
            "agw_station": rec["agw_station"],
            "madis_key": rec["madis_key"],
            "missing_dates": rec["missing_dates"],
            "rs_period": rec["rs_period"],
            "rs_sample_size": rec["rs_sample_size"],
            "rs_despiked": rec["rs_despiked"],
            "rs_clipped": rec["rs_clipped"],
            "rh_rhmax_exceed_100": rec["rh_rhmax_exceed_100"],
            "rh_rhmin_exceed_100": rec["rh_rhmin_exceed_100"],
            "manual_nulled_count": len(rec["manual_nulled_intervals"]),
        }
        for var, count in rec["zscore_removals"].items():
            row[f"zscore_{var.lower()}"] = count
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


# ---------------------------------------------------------------------------
# 6. Flag agreement summary
# ---------------------------------------------------------------------------


def flag_agreement_summary(
    comp_df: pd.DataFrame,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Compute per-rule confusion matrix across all station-date-variable rows.

    For each Tier-2 rule, reports:
      - n_our_flag   : days where our reason code fired for this variable
      - n_agw_null   : agweather has NaN for this variable (their QC removed it)
      - tp            : we flag AND agweather null (agreement — both removed)
      - fp            : we flag AND agweather has value (we over-flagged)
      - fn            : we pass AND agweather null (we under-flagged / missed)
      - tn            : both pass (agreement — both kept)
      - fp_rate       : fp / n_our_flag (fraction of our flags that are over-flags)
      - fn_rate       : fn / n_agw_null (fraction of their removals we missed)
    """
    records = []

    for rule, agw_vars in RULE_TO_AGW_VARS.items():
        for var in agw_vars:
            subset = comp_df[comp_df["variable"] == var].copy()
            if subset.empty:
                continue

            # Our flag for this rule+variable: reason code appears in var_reasons
            our_flag = subset["var_reasons"].str.contains(rule, na=False)
            agw_null = subset["agw_null"]

            tp = int((our_flag & agw_null).sum())
            fp = int((our_flag & ~agw_null).sum())
            fn = int((~our_flag & agw_null).sum())
            tn = int((~our_flag & ~agw_null).sum())
            n_our_flag = tp + fp
            n_agw_null = tp + fn

            records.append(
                {
                    "rule": rule,
                    "variable": var,
                    "n_our_flag": n_our_flag,
                    "n_agw_null": n_agw_null,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "fp_rate": fp / n_our_flag if n_our_flag > 0 else np.nan,
                    "fn_rate": fn / n_agw_null if n_agw_null > 0 else np.nan,
                    "precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
                    "recall": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
                }
            )

    result = pd.DataFrame(records).sort_values(["fp_rate", "rule"], ascending=[False, True])

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        logger.info("Wrote flag agreement summary: %s", out_path)

    return result


# ---------------------------------------------------------------------------
# 7. Correction comparison
# ---------------------------------------------------------------------------


def correction_comparison(
    comp_df: pd.DataFrame,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Compare correction magnitudes between agweather and obsmet.

    For days where both datasets have non-null raw AND corrected values,
    computes bias and MAE of our correction vs theirs for Rs and RH.
    """
    records = []

    MAX_RSDS = 100.0  # MJ/m²/day — well above any real daily total

    for var, corrected_var in [("rsds", "rsds_corrected")]:
        raw_rows = comp_df[comp_df["variable"] == var].set_index(
            ["agw_station", "madis_key", "date"]
        )
        corr_rows = comp_df[comp_df["variable"] == corrected_var].set_index(
            ["agw_station", "madis_key", "date"]
        )

        raw_rows = raw_rows[raw_rows["our_value"].abs() <= MAX_RSDS]
        corr_rows = corr_rows[corr_rows["our_value"].abs() <= MAX_RSDS]

        common_idx = raw_rows.index.intersection(corr_rows.index)
        if common_idx.empty:
            continue

        raw_rows = raw_rows.loc[common_idx]
        corr_rows = corr_rows.loc[common_idx]

        both_raw_valid = raw_rows["agw_value"].notna() & raw_rows["our_value"].notna()
        both_corr_valid = corr_rows["agw_value"].notna() & corr_rows["our_value"].notna()
        valid = both_raw_valid & both_corr_valid

        if valid.sum() == 0:
            continue

        agw_correction = corr_rows.loc[valid, "agw_value"] - raw_rows.loc[valid, "agw_value"]
        our_correction = corr_rows.loc[valid, "our_value"] - raw_rows.loc[valid, "our_value"]

        diff = our_correction - agw_correction
        records.append(
            {
                "variable": var,
                "n": int(valid.sum()),
                "mean_agw_correction": float(agw_correction.mean()),
                "mean_our_correction": float(our_correction.mean()),
                "correction_bias": float(diff.mean()),
                "correction_mae": float(diff.abs().mean()),
                "correction_corr": float(agw_correction.corr(our_correction)),
            }
        )

    result = pd.DataFrame(records)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        logger.info("Wrote correction comparison: %s", out_path)

    return result


# ---------------------------------------------------------------------------
# 8. Full run
# ---------------------------------------------------------------------------


def run_agweather_validation(
    matches_csv: Path = DEFAULT_MATCHES_CSV,
    agw_base: Path = DEFAULT_AGW_BASE,
    por_base: Path = DEFAULT_POR_BASE,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    max_dist_m: float = 500.0,
    max_elev_diff_m: float = 50.0,
    min_overlap_days: int = 365,
    limit: int | None = None,
) -> dict[str, int | str]:
    """Run the full agweather validation pipeline and write artifacts."""
    comp_path = out_dir / "comparison_dataset.parquet"
    agreement_path = out_dir / "flag_agreement_by_rule.csv"
    correction_path = out_dir / "correction_comparison.csv"

    comp_df = build_comparison_dataset(
        matches_csv=matches_csv,
        agw_base=agw_base,
        por_base=por_base,
        out_path=comp_path,
        max_dist_m=max_dist_m,
        max_elev_diff_m=max_elev_diff_m,
        min_overlap_days=min_overlap_days,
        limit=limit,
    )

    if comp_df.empty:
        logger.warning("Empty comparison dataset — no output written")
        return {"status": "empty"}

    agreement_df = flag_agreement_summary(comp_df, out_path=agreement_path)
    correction_df = correction_comparison(comp_df, out_path=correction_path)

    return {
        "status": "ok",
        "pairs_compared": comp_df[["agw_station", "madis_key"]].drop_duplicates().__len__(),
        "total_rows": len(comp_df),
        "flag_agreement_rows": len(agreement_df),
        "correction_rows": len(correction_df),
        "outputs": str(out_dir),
    }
