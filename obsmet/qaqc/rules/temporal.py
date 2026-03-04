"""Tier 2: Temporal QC rules for station-level time-series checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from obsmet.qaqc.rules.base import QCResult, QCRule, QCTier


# --------------------------------------------------------------------------- #
# Monthly Z-Score Rule
# --------------------------------------------------------------------------- #


@dataclass
class MonthlyZScoreRule(QCRule):
    """Flag outliers using modified z-score per station-variable-month.

    Uses MAD (median absolute deviation) for robust outlier detection.
    Modified z = 0.6745 * (x - median) / MAD.
    |z| > 3.5 → suspect, |z| > 7.0 → fail.
    """

    tier: QCTier = QCTier.TEMPORAL
    name: str = "monthly_zscore"
    version: str = "1"
    suspect_threshold: float = 3.5
    fail_threshold: float = 7.0
    min_obs_per_month: int = 90
    min_obs_total: int = 90

    def check(self, value: float, **context) -> QCResult:
        """Single-value check (requires monthly_median and monthly_mad in context)."""
        median = context.get("monthly_median")
        mad = context.get("monthly_mad")
        if median is None or mad is None or mad == 0:
            return QCResult(
                state="pass",
                reason="zscore_insufficient_stats",
                tier=self.tier,
                rule_name=self.name,
            )
        z = 0.6745 * abs(value - median) / mad
        if z > self.fail_threshold:
            return QCResult(
                state="fail",
                reason="zscore_extreme",
                tier=self.tier,
                rule_name=self.name,
                detail=f"modified_z={z:.1f}",
            )
        if z > self.suspect_threshold:
            return QCResult(
                state="suspect",
                reason="zscore_outlier",
                tier=self.tier,
                rule_name=self.name,
                detail=f"modified_z={z:.1f}",
            )
        return QCResult(state="pass", reason="zscore_ok", tier=self.tier, rule_name=self.name)

    def check_series(
        self,
        series: pd.Series,
        dates: pd.Series,
    ) -> pd.Series:
        """Vectorized check on a station's time series.

        Returns Series of state strings aligned with input index.
        """
        result = pd.Series("pass", index=series.index)
        valid = series.dropna()
        if len(valid) < self.min_obs_total:
            return result

        months = (
            dates.loc[valid.index].dt.month
            if hasattr(dates.dt, "month")
            else pd.to_datetime(dates.loc[valid.index]).dt.month
        )

        for month in months.unique():
            mask = months == month
            vals = valid[mask]
            if len(vals) < self.min_obs_per_month:
                # Fall back to annual stats
                med = valid.median()
                mad = np.median(np.abs(valid - med))
            else:
                med = vals.median()
                mad = np.median(np.abs(vals - med))

            if mad == 0:
                continue

            z = 0.6745 * np.abs(vals - med) / mad
            result.loc[vals.index[z > self.fail_threshold]] = "fail"
            suspect_mask = (z > self.suspect_threshold) & (z <= self.fail_threshold)
            result.loc[vals.index[suspect_mask]] = "suspect"

        return result


# --------------------------------------------------------------------------- #
# Isolated Observation Rule
# --------------------------------------------------------------------------- #


@dataclass
class IsolatedObsRule(QCRule):
    """Flag observations with no valid neighbor within ±gap_hours.

    Designed for hourly data; not meaningful for daily series.
    """

    tier: QCTier = QCTier.TEMPORAL
    name: str = "isolated_obs"
    version: str = "1"
    gap_hours: int = 6

    def check(self, value: float, **context) -> QCResult:
        """Single-value check (requires prev_gap_hours and next_gap_hours in context)."""
        prev_gap = context.get("prev_gap_hours")
        next_gap = context.get("next_gap_hours")
        if prev_gap is None or next_gap is None:
            return QCResult(
                state="pass",
                reason="isolation_no_context",
                tier=self.tier,
                rule_name=self.name,
            )
        if prev_gap > self.gap_hours and next_gap > self.gap_hours:
            return QCResult(
                state="suspect",
                reason="isolated_obs",
                tier=self.tier,
                rule_name=self.name,
                detail=f"gaps: prev={prev_gap:.1f}h next={next_gap:.1f}h",
            )
        return QCResult(state="pass", reason="not_isolated", tier=self.tier, rule_name=self.name)

    def check_series(self, datetime_utc: pd.Series) -> pd.Series:
        """Vectorized isolation check on sorted datetime series.

        Returns Series of state strings aligned with input index.
        """
        result = pd.Series("pass", index=datetime_utc.index)
        if len(datetime_utc) < 2:
            return result

        dt = pd.to_datetime(datetime_utc).sort_values()
        diffs = dt.diff()
        gap_hours = diffs.dt.total_seconds() / 3600.0

        # Forward and backward gaps
        fwd = gap_hours.shift(-1)  # gap to next obs
        bwd = gap_hours  # gap from prev obs

        isolated = (bwd > self.gap_hours) & (fwd > self.gap_hours)
        # First obs: only check forward gap
        if pd.notna(fwd.iloc[0]) and fwd.iloc[0] > self.gap_hours:
            isolated.iloc[0] = True
        # Last obs: only check backward gap
        if pd.notna(bwd.iloc[-1]) and bwd.iloc[-1] > self.gap_hours:
            isolated.iloc[-1] = True

        result.loc[isolated] = "suspect"
        return result


# --------------------------------------------------------------------------- #
# Dewpoint-Temperature Daily Rule
# --------------------------------------------------------------------------- #


@dataclass
class DewpointTemperatureRule(QCRule):
    """Daily check: flag days where mean Td > Tmin + tolerance.

    Stricter than hourly DewpointConsistencyRule (which checks Td ≤ Tair + 0.5).
    """

    tier: QCTier = QCTier.TEMPORAL
    name: str = "dewpoint_temperature_daily"
    version: str = "1"
    tolerance: float = 1.0

    def check(self, value: float, **context) -> QCResult:
        """Single-value check where value is daily mean td, context has tmin."""
        tmin = context.get("tmin")
        if tmin is None:
            return QCResult(
                state="pass",
                reason="td_daily_no_tmin",
                tier=self.tier,
                rule_name=self.name,
            )
        if value > tmin + self.tolerance:
            return QCResult(
                state="fail",
                reason="td_exceeds_tmin_daily",
                tier=self.tier,
                rule_name=self.name,
                detail=f"td_mean={value:.1f} > tmin={tmin:.1f} + {self.tolerance}",
            )
        return QCResult(state="pass", reason="td_tmin_ok", tier=self.tier, rule_name=self.name)

    def check_daily(self, td: pd.Series, tmin: pd.Series) -> pd.Series:
        """Vectorized daily check: flag where mean Td > Tmin + tolerance.

        Both series should be aligned (same index = dates).
        Returns Series of state strings.
        """
        result = pd.Series("pass", index=td.index)
        both_valid = td.notna() & tmin.notna()
        exceeds = both_valid & (td > tmin + self.tolerance)
        result.loc[exceeds] = "fail"
        return result


# --------------------------------------------------------------------------- #
# Stuck Sensor Rule
# --------------------------------------------------------------------------- #


@dataclass
class StuckSensorRule(QCRule):
    """Flag runs of N+ identical non-zero values (stuck sensor).

    Zero values are excluded from stuck detection (e.g., no precip is normal).
    """

    tier: QCTier = QCTier.TEMPORAL
    name: str = "stuck_sensor"
    version: str = "1"
    min_run_length: int = 12

    def check(self, value: float, **context) -> QCResult:
        """Single-value check (requires run_length in context)."""
        run_length = context.get("run_length", 1)
        if value == 0:
            return QCResult(
                state="pass", reason="stuck_zero_exempt", tier=self.tier, rule_name=self.name
            )
        if run_length >= self.min_run_length:
            return QCResult(
                state="suspect",
                reason="stuck_sensor",
                tier=self.tier,
                rule_name=self.name,
                detail=f"run_length={run_length}",
            )
        return QCResult(state="pass", reason="not_stuck", tier=self.tier, rule_name=self.name)

    def check_series(self, series: pd.Series) -> pd.Series:
        """Vectorized stuck sensor check.

        Returns Series of state strings aligned with input index.
        """
        result = pd.Series("pass", index=series.index)
        valid = series.dropna()
        if valid.empty:
            return result

        # Exclude zeros
        nonzero = valid[valid != 0]
        if nonzero.empty:
            return result

        # Find runs of consecutive equal values
        shifted = nonzero.shift(1)
        group_id = (nonzero != shifted).cumsum()
        run_lengths = group_id.map(group_id.value_counts())

        stuck = run_lengths >= self.min_run_length
        result.loc[stuck.index[stuck]] = "suspect"
        return result
