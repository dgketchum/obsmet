"""Tier 2: Temporal QC rules for station-level time-series checks.

Uses agweather-qaqc (Dunkerly et al., 2024) for core algorithms:
- Modified z-score outlier detection per station-variable-month
- RH yearly percentile drift correction
- Rs period-ratio correction with spike detection
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agweatherqaqc.qaqc_functions import (
    modified_z_score_outlier_detection,
    rh_yearly_percentile_corr,
    rs_period_ratio_corr,
)

from obsmet.qaqc.rules.base import QCResult, QCRule, QCTier


# --------------------------------------------------------------------------- #
# Monthly Z-Score Rule
# --------------------------------------------------------------------------- #


@dataclass
class MonthlyZScoreRule(QCRule):
    """Flag outliers using modified z-score per station-variable-month.

    Delegates to agweather-qaqc's ``modified_z_score_outlier_detection``
    (Iglewicz & Hoaglin, 1993; threshold = 3.5).
    Adds a second fail tier at z > 7.0.
    """

    tier: QCTier = QCTier.TEMPORAL
    name: str = "monthly_zscore"
    version: str = "2"
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

        Uses agweather-qaqc's modified_z_score_outlier_detection per month
        for the 3.5 suspect tier, then computes z-scores for the 7.0 fail tier.
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
                continue

            data = vals.values.copy()

            # agweather-qaqc flags |z| > 3.5 by setting values to NaN
            cleaned, _ = modified_z_score_outlier_detection(data)
            flagged_35 = ~np.isnan(data) & np.isnan(cleaned)
            result.loc[vals.index[flagged_35]] = "suspect"

            # Extend to fail tier (z > 7.0)
            med = np.nanmedian(data)
            mad = np.nanmedian(np.abs(data - med))
            if mad == 0:
                continue
            z = 0.6745 * np.abs(data - med) / mad
            result.loc[vals.index[z > self.fail_threshold]] = "fail"

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
    tolerance: float = 3.0

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


# --------------------------------------------------------------------------- #
# RH Drift Rule (agweather-qaqc yearly percentile correction)
# --------------------------------------------------------------------------- #


@dataclass
class RHDriftRule(QCRule):
    """Detect humidity sensor drift via yearly percentile correction.

    Wraps agweather-qaqc's ``rh_yearly_percentile_corr`` which assumes that
    RHmax should reach ~100% at least once per year in agricultural areas.
    The multiplicative correction factor per year indicates sensor drift;
    days in years with large corrections are flagged.
    """

    tier: QCTier = QCTier.TEMPORAL
    name: str = "rh_drift"
    version: str = "1"
    percentage: int = 1
    suspect_corr_factor: float = 1.05
    fail_corr_factor: float = 1.15

    def check(self, value: float, **context) -> QCResult:
        """Single-value interface (not meaningful for this rule)."""
        return QCResult(
            state="pass", reason="rh_drift_no_context", tier=self.tier, rule_name=self.name
        )

    def _compute_rh_correction(
        self,
        rhmax: pd.Series,
        rhmin: pd.Series,
        years: pd.Series,
    ) -> tuple[pd.Series, np.ndarray, np.ndarray, dict[int, float]]:
        """Shared RH correction logic.

        Returns (state_series, corrected_rhmax, corrected_rhmin, year_factors).
        """
        result = pd.Series("pass", index=rhmax.index)
        rh_max_arr = rhmax.values.astype(np.float64).copy()
        rh_min_arr = rhmin.values.astype(np.float64).copy()
        year_arr = years.values.astype(np.int32)

        n = len(rh_max_arr)
        if n < 365:
            return result, rh_max_arr, rh_min_arr, {}

        log_buf = io.StringIO()
        with redirect_stdout(io.StringIO()):
            corr_rhmax, corr_rhmin = rh_yearly_percentile_corr(
                log_buf, 0, n, rh_max_arr, rh_min_arr, year_arr, self.percentage
            )

        # Clip corrected values to [0, 100]
        corr_rhmax = np.clip(corr_rhmax, 0, 100)
        corr_rhmin = np.clip(corr_rhmin, 0, 100)

        # Compute per-year correction factor: corr / original
        year_factors: dict[int, float] = {}
        unique_years = np.unique(year_arr)
        for yr in unique_years:
            yr_mask = year_arr == yr
            orig_vals = rh_max_arr[yr_mask]
            corr_vals = corr_rhmax[yr_mask]

            valid = ~np.isnan(orig_vals) & ~np.isnan(corr_vals) & (orig_vals > 0)
            if not np.any(valid):
                continue

            factors = corr_vals[valid] / orig_vals[valid]
            mean_factor = float(np.nanmean(factors))
            year_factors[int(yr)] = mean_factor

            yr_idx = rhmax.index[yr_mask]
            if mean_factor >= self.fail_corr_factor or mean_factor <= (1 / self.fail_corr_factor):
                result.loc[yr_idx] = "fail"
            elif mean_factor >= self.suspect_corr_factor or mean_factor <= (
                1 / self.suspect_corr_factor
            ):
                result.loc[yr_idx] = "suspect"

        return result, corr_rhmax, corr_rhmin, year_factors

    def check_series(
        self,
        rhmax: pd.Series,
        rhmin: pd.Series,
        years: pd.Series,
    ) -> pd.Series:
        """Flag days in years where RH sensor drift exceeds thresholds.

        Parameters
        ----------
        rhmax, rhmin : pd.Series
            Daily max/min relative humidity (aligned index).
        years : pd.Series
            Year for each observation (aligned index).
        """
        states, _, _, _ = self._compute_rh_correction(rhmax, rhmin, years)
        return states

    def correct_series(
        self,
        rhmax: pd.Series,
        rhmin: pd.Series,
        years: pd.Series,
    ) -> tuple[pd.Series, np.ndarray, np.ndarray, dict[int, float]]:
        """Flag drift AND return corrected RH arrays.

        Returns (states, corrected_rhmax, corrected_rhmin, year_factors).
        """
        return self._compute_rh_correction(rhmax, rhmin, years)


# --------------------------------------------------------------------------- #
# Rs Period Ratio Rule (agweather-qaqc period-ratio correction)
# --------------------------------------------------------------------------- #

_DEFAULT_RSUN_PATH = os.path.join("/nas", "dads", "mvp", "rsun_pnw_1km.tif")


@dataclass
class RsPeriodRatioRule(QCRule):
    """Detect solar radiation spikes and drift via period-ratio correction.

    Wraps agweather-qaqc's ``rs_period_ratio_corr`` which compares observed Rs
    to clear-sky Rso in configurable windows, identifying spikes via a two-rule
    test (2% correction-factor sensitivity + 75 W/m² absolute excess).

    Rso is extracted from pre-computed RSUN terrain-corrected rasters rather
    than the flat-earth refet calculation used natively by agweather-qaqc.
    """

    tier: QCTier = QCTier.TEMPORAL
    name: str = "rs_period_ratio"
    version: str = "1"
    period: int = 60
    sample_size: int = 6

    def check(self, value: float, **context) -> QCResult:
        """Single-value interface (not meaningful for this rule)."""
        return QCResult(
            state="pass", reason="rs_ratio_no_context", tier=self.tier, rule_name=self.name
        )

    def _compute_rs_correction(
        self,
        rs: pd.Series,
        rso: np.ndarray,
        doy: pd.Series,
    ) -> tuple[pd.Series, np.ndarray]:
        """Shared Rs correction logic.

        Returns (state_series, corrected_rs_array).
        """
        result = pd.Series("pass", index=rs.index)
        rs_arr = rs.values.astype(np.float64).copy()
        n = len(rs_arr)
        if n < self.period:
            return result, rs_arr

        # Build per-day Rso from DOY lookup
        rso_arr = rso[(doy.values.astype(int) - 1) % 365].astype(np.float64)

        # Cap values exceeding 2 × peak Rso to prevent float overflow in correction
        rso_max = float(np.nanmax(rso_arr)) if rso_arr.size else 50.0
        rs_arr = np.where(rs_arr > 2.0 * rso_max, np.nan, rs_arr)
        rs_arr[rs_arr == 0] = np.nan  # Zero rsds encodes no valid daytime obs, not zero radiation

        corr_input = rs_arr
        rso_input = rso_arr
        corr_end = n

        # agweather-qaqc drops the last full correction period when the series
        # length is an exact multiple of ``period``. Padding with a trailing NaN
        # forces its final-partial-period branch and keeps the output aligned.
        if n % self.period == 0:
            corr_input = np.append(rs_arr, np.nan)
            rso_input = np.append(rso_arr, np.nan)
            corr_end = n + 1

        log_buf = io.StringIO()
        with redirect_stdout(io.StringIO()):
            corr_rs, _ = rs_period_ratio_corr(
                log_buf,
                0,
                corr_end,
                corr_input,
                rso_input,
                self.sample_size,
                self.period,
            )
        corr_rs = np.asarray(corr_rs[:n], dtype=np.float64)

        # Flag only days where the removed value exceeded clear-sky Rso —
        # a strict physical violation. CF-instability removals (rs ≤ rso) are
        # artifacts of cloudy or short-record periods, not measurement errors.
        was_valid = ~np.isnan(rs_arr)
        now_nan = np.isnan(corr_rs)
        removed = was_valid & now_nan
        overestimate = rs_arr > rso_arr
        result.loc[rs.index[removed & overestimate]] = "fail"

        return result, corr_rs

    def check_series(
        self,
        rs: pd.Series,
        rso: np.ndarray,
        doy: pd.Series,
    ) -> pd.Series:
        """Flag days where Rs correction indicates spikes or severe drift.

        Parameters
        ----------
        rs : pd.Series
            Daily observed solar radiation (aligned index). Must be same
            units as rso (typically MJ/m²/day).
        rso : np.ndarray
            365-element clear-sky array indexed by DOY-1 (same units as rs).
        doy : pd.Series
            Day-of-year for each observation (1-365, aligned index).
        """
        states, _ = self._compute_rs_correction(rs, rso, doy)
        return states

    def correct_series(
        self,
        rs: pd.Series,
        rso: np.ndarray,
        doy: pd.Series,
    ) -> tuple[pd.Series, np.ndarray]:
        """Flag drift AND return corrected Rs array.

        Returns (states, corrected_rs_array).
        """
        return self._compute_rs_correction(rs, rso, doy)
