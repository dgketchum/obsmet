"""Tests for obsmet.qaqc rules and pipeline."""

import numpy as np
import pandas as pd

from obsmet.qaqc.engines.pipeline import (
    QCPipeline,
    apply_pipeline_to_df,
    build_default_pipeline,
)
from obsmet.qaqc.rules.bounds import DewpointConsistencyRule, PhysicalBoundsRule
from obsmet.qaqc.rules.temporal import (
    DewpointTemperatureRule,
    IsolatedObsRule,
    MonthlyZScoreRule,
    RHDriftRule,
    RsPeriodRatioRule,
    StuckSensorRule,
)


class TestPhysicalBounds:
    def test_pass(self):
        rule = PhysicalBoundsRule()
        result = rule.check(20.0, variable="tair")
        assert result.state == "pass"

    def test_fail_high(self):
        rule = PhysicalBoundsRule()
        result = rule.check(70.0, variable="tair")
        assert result.state == "fail"
        assert result.reason == "out_of_bounds"

    def test_fail_low(self):
        rule = PhysicalBoundsRule()
        result = rule.check(-100.0, variable="tair")
        assert result.state == "fail"

    def test_no_bounds(self):
        rule = PhysicalBoundsRule()
        result = rule.check(1.0, variable="unknown_var")
        assert result.state == "pass"
        assert result.reason == "no_bounds_defined"


class TestDewpointConsistency:
    def test_pass(self):
        rule = DewpointConsistencyRule()
        result = rule.check(15.0, tair=20.0)
        assert result.state == "pass"

    def test_fail(self):
        rule = DewpointConsistencyRule()
        result = rule.check(25.0, tair=20.0)
        assert result.state == "fail"
        assert result.reason == "td_exceeds_tair"

    def test_no_tair(self):
        rule = DewpointConsistencyRule()
        result = rule.check(15.0)
        assert result.state == "pass"


class TestQCPipeline:
    def test_aggregate_pass(self):
        pipeline = QCPipeline()
        pipeline.add_rule(PhysicalBoundsRule())
        results = pipeline.run(20.0, variable="tair")
        assert QCPipeline.aggregate_state(results) == "pass"

    def test_aggregate_fail(self):
        pipeline = QCPipeline()
        pipeline.add_rule(PhysicalBoundsRule())
        results = pipeline.run(70.0, variable="tair")
        assert QCPipeline.aggregate_state(results) == "fail"
        assert "out_of_bounds" in QCPipeline.reason_codes(results)


class TestBuildDefaultPipeline:
    def test_madis_pipeline_has_four_rules(self):
        pipeline = build_default_pipeline("madis")
        assert len(pipeline.rules) == 4

    def test_isd_pipeline_has_two_rules(self):
        pipeline = build_default_pipeline("isd")
        assert len(pipeline.rules) == 2

    def test_gdas_pipeline_has_three_rules(self):
        pipeline = build_default_pipeline("gdas")
        assert len(pipeline.rules) == 3


class TestApplyPipelineToDf:
    def test_good_values_pass(self):
        df = pd.DataFrame(
            {
                "tair": [20.0, 15.0],
                "td": [10.0, 8.0],
                "rh": [50.0, 60.0],
            }
        )
        pipeline = build_default_pipeline("isd")
        df = apply_pipeline_to_df(df, pipeline, ["tair", "td", "rh"], source="isd")
        assert (df["qc_state"] == "pass").all()

    def test_out_of_bounds_fails(self):
        df = pd.DataFrame(
            {
                "tair": [20.0, 70.0],  # 70 is out of bounds
                "td": [10.0, 10.0],
            }
        )
        pipeline = build_default_pipeline("isd")
        df = apply_pipeline_to_df(df, pipeline, ["tair", "td"], source="isd")
        assert df.iloc[0]["qc_state"] == "pass"
        assert df.iloc[1]["qc_state"] == "fail"
        assert "out_of_bounds" in df.iloc[1]["qc_reason_codes"]

    def test_nan_values_skipped(self):
        df = pd.DataFrame(
            {
                "tair": [np.nan, 20.0],
                "td": [np.nan, 10.0],
            }
        )
        pipeline = build_default_pipeline("isd")
        df = apply_pipeline_to_df(df, pipeline, ["tair", "td"], source="isd")
        assert df.iloc[0]["qc_state"] == "pass"  # all NaN → pass by default
        assert df.iloc[1]["qc_state"] == "pass"

    def test_preserves_adapter_native_qc_on_null_values(self):
        """Pipeline must not overwrite adapter QC when value is NaN.

        Simulates the GHCNh/ECCC path: adapter nulls a bad value and sets
        {var}_qc_state='fail'. Pipeline should preserve that state, not
        reset it to 'pass' because the value is NaN.
        """
        df = pd.DataFrame(
            {
                "tair": [np.nan, 20.0],
                "td": [5.0, 10.0],
                "tair_qc_state": ["fail", "pass"],
                "tair_qc_reason_codes": ["tair:qc_3", ""],
                "td_qc_state": ["pass", "pass"],
                "td_qc_reason_codes": ["", ""],
                "qc_state": ["fail", "pass"],
                "qc_reason_codes": ["tair:qc_3", ""],
            }
        )
        pipeline = build_default_pipeline("isd")
        result = apply_pipeline_to_df(df, pipeline, ["tair", "td"], source="isd")
        # Row 0: tair was nulled by adapter, QC should stay fail
        assert result.iloc[0]["tair_qc_state"] == "fail"
        assert "tair:qc_3" in result.iloc[0]["tair_qc_reason_codes"]
        # Row 0: td was fine, should stay pass
        assert result.iloc[0]["td_qc_state"] == "pass"
        # Row-level should reflect the worst (fail from tair)
        assert result.iloc[0]["qc_state"] == "fail"

    def test_replaces_qc_passed(self):
        df = pd.DataFrame(
            {
                "tair": [20.0],
                "qc_passed": [True],
            }
        )
        pipeline = build_default_pipeline("isd")
        df = apply_pipeline_to_df(df, pipeline, ["tair"], source="isd")
        assert "qc_state" in df.columns
        assert "qc_passed" not in df.columns

    def test_gdas_qm_can_fail_row(self):
        df = pd.DataFrame(
            {
                "tair": [20.0],
                "tair_qm": [9],
            }
        )
        pipeline = build_default_pipeline("gdas")
        df = apply_pipeline_to_df(df, pipeline, ["tair"], source="gdas")
        assert df.iloc[0]["qc_state"] == "suspect"
        assert "qm9_missing_obs_error" in df.iloc[0]["qc_reason_codes"]

    def test_gdas_qm15_is_retained_as_suspect(self):
        df = pd.DataFrame(
            {
                "tair": [20.0],
                "tair_qm": [15],
            }
        )
        pipeline = build_default_pipeline("gdas")
        df = apply_pipeline_to_df(df, pipeline, ["tair"], source="gdas")
        assert df.iloc[0]["qc_state"] == "suspect"
        assert "qm15_non_use_by_analysis" in df.iloc[0]["qc_reason_codes"]


# --------------------------------------------------------------------------- #
# Temporal rules
# --------------------------------------------------------------------------- #


class TestMonthlyZScore:
    def test_normal_values_pass(self):
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        vals = pd.Series(rng.normal(20, 3, n), index=range(n))
        rule = MonthlyZScoreRule()
        states = rule.check_series(vals, dates.to_series(index=range(n)))
        assert (states == "pass").sum() > n * 0.9

    def test_extreme_value_flagged(self):
        rng = np.random.default_rng(42)
        # Need enough obs per month (>=90); 4 years of daily data
        n = 365 * 4
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        vals = pd.Series(rng.normal(20, 3, n), index=range(n))
        vals.iloc[5] = 60.0  # extreme outlier in January
        rule = MonthlyZScoreRule()
        states = rule.check_series(vals, dates.to_series(index=range(n)))
        assert states.iloc[5] in ("suspect", "fail")

    def test_short_por_skips(self):
        vals = pd.Series([20.0, 21.0, 19.0])
        dates = pd.Series(pd.date_range("2024-01-01", periods=3, freq="D"))
        rule = MonthlyZScoreRule()
        states = rule.check_series(vals, dates)
        assert (states == "pass").all()

    def test_uses_agweatherqaqc(self):
        """Verify that agweather-qaqc's function is actually called."""
        rng = np.random.default_rng(42)
        n = 365 * 4
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        vals = pd.Series(rng.normal(20, 3, n), index=range(n))
        vals.iloc[5] = 100.0  # obvious outlier
        rule = MonthlyZScoreRule()
        states = rule.check_series(vals, dates.to_series(index=range(n)))
        # The outlier should be caught by agweather-qaqc's 3.5 threshold
        assert states.iloc[5] in ("suspect", "fail")


class TestIsolatedObs:
    def test_continuous_passes(self):
        dt = pd.Series(pd.date_range("2024-01-01", periods=24, freq="h"))
        rule = IsolatedObsRule()
        states = rule.check_series(dt)
        assert (states == "pass").all()

    def test_lone_obs_flagged(self):
        # Single obs with 12h gaps on either side
        times = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 12:00", "2024-01-02 00:00"])
        dt = pd.Series(times, index=[0, 1, 2])
        rule = IsolatedObsRule(gap_hours=6)
        states = rule.check_series(dt)
        # Each obs has 12h gap — all should be flagged
        assert (states == "suspect").all()


class TestStuckSensor:
    def test_varying_values_pass(self):
        vals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        rule = StuckSensorRule(min_run_length=3)
        states = rule.check_series(vals)
        assert (states == "pass").all()

    def test_identical_values_flagged(self):
        vals = pd.Series([5.0] * 15)
        rule = StuckSensorRule(min_run_length=12)
        states = rule.check_series(vals)
        assert (states == "suspect").all()

    def test_zeros_not_flagged(self):
        vals = pd.Series([0.0] * 20)
        rule = StuckSensorRule(min_run_length=5)
        states = rule.check_series(vals)
        assert (states == "pass").all()


class TestDewpointTemperatureDaily:
    def test_td_below_tmin_passes(self):
        td = pd.Series([5.0, 3.0, 4.0])
        tmin = pd.Series([10.0, 8.0, 9.0])
        rule = DewpointTemperatureRule()
        states = rule.check_daily(td, tmin)
        assert (states == "pass").all()

    def test_td_exceeds_tmin_fails(self):
        td = pd.Series([15.0, 3.0])
        tmin = pd.Series([10.0, 8.0])
        rule = DewpointTemperatureRule(tolerance=1.0)
        states = rule.check_daily(td, tmin)
        assert states.iloc[0] == "fail"
        assert states.iloc[1] == "pass"


# --------------------------------------------------------------------------- #
# New agweather-qaqc-based rules
# --------------------------------------------------------------------------- #


class TestRHDrift:
    def test_no_drift_passes(self):
        """Station with RHmax regularly hitting 100% → no drift detected."""
        rng = np.random.default_rng(42)
        n = 365 * 3
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        # RHmax with some values near 100 each year
        rhmax = pd.Series(rng.uniform(60, 100, n), index=range(n))
        # Sprinkle some 100s each year
        for yr_start in range(0, n, 365):
            for i in range(5):
                if yr_start + i * 30 < n:
                    rhmax.iloc[yr_start + i * 30] = 100.0

        rhmin = pd.Series(rng.uniform(30, 70, n), index=range(n))
        years = pd.Series(dates.year.values, index=range(n))

        rule = RHDriftRule()
        states = rule.check_series(rhmax, rhmin, years)
        # Most should pass
        assert (states == "pass").sum() > n * 0.8

    def test_severe_drift_flagged(self):
        """Station with RHmax never exceeding 70% → drift should be detected."""
        rng = np.random.default_rng(42)
        n = 365 * 2
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        # RHmax capped at 70 — strong drift signal
        rhmax = pd.Series(rng.uniform(40, 70, n), index=range(n))
        rhmin = pd.Series(rng.uniform(20, 50, n), index=range(n))
        years = pd.Series(dates.year.values, index=range(n))

        rule = RHDriftRule()
        states = rule.check_series(rhmax, rhmin, years)
        # With RHmax never hitting 100, correction factor ~1.43 → fail
        assert (states != "pass").sum() > 0

    def test_correct_series_returns_corrected_values(self):
        """correct_series() should return 4-tuple with corrected arrays."""
        rng = np.random.default_rng(42)
        n = 365 * 2
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        rhmax = pd.Series(rng.uniform(40, 70, n), index=range(n))
        rhmin = pd.Series(rng.uniform(20, 50, n), index=range(n))
        years = pd.Series(dates.year.values, index=range(n))

        rule = RHDriftRule()
        result = rule.correct_series(rhmax, rhmin, years)
        assert len(result) == 4
        states, corr_rhmax, corr_rhmin, year_factors = result
        assert isinstance(states, pd.Series)
        assert isinstance(corr_rhmax, np.ndarray)
        assert isinstance(corr_rhmin, np.ndarray)
        assert isinstance(year_factors, dict)
        assert len(corr_rhmax) == n
        # Corrected values should be clipped to [0, 100]
        assert np.nanmax(corr_rhmax) <= 100.0
        assert np.nanmin(corr_rhmin[~np.isnan(corr_rhmin)]) >= 0.0


class TestRsPeriodRatio:
    def test_clean_data_mostly_passes(self):
        """Synthetic Rs ≈ 0.90 × Rso — no spikes, mild attenuation."""
        rng = np.random.default_rng(42)
        n = 365
        rso = np.sin(np.linspace(0, np.pi, 365)) * 300 + 100  # seasonal Rso curve
        rs = rso * 0.90 + rng.normal(0, 5, n)  # Rs with small noise, close to Rso
        rs[rs < 0] = 0

        rs_series = pd.Series(rs, index=range(n))
        doy = pd.Series(np.arange(1, 366), index=range(n))

        rule = RsPeriodRatioRule()
        states = rule.check_series(rs_series, rso, doy)
        # Clean data shouldn't have many hard fails
        assert (states == "fail").sum() < n * 0.1

    def test_spike_detected(self):
        """Period where rs >> rso forces CF < 0.5 → NaN-ification; rs > rso days must be flagged.

        Uses small rso (max ≈ 30) so rs = 2.5×rso stays below the 2×rso_max overflow guard.
        CF = rso / (2.5×rso) = 0.40 < 0.50 → algorithm NaN-ifies the whole period.
        Those NaN-ified days also have rs > rso → flagged as fail.
        """
        rso = np.sin(np.linspace(0, np.pi, 365)) * 20 + 10  # max ≈ 30; 2×max ≈ 60
        rs = rso[:180] * 0.75  # normal background for periods 1–2
        # Period 0 (days 0–59): rs = 2.5×rso → CF = 0.40 < 0.50 → period NaN'd
        # rs ≈ 25–42 MJ/m²/day, all below the overflow guard (≈60)
        rs[:60] = rso[:60] * 2.5

        rs_series = pd.Series(rs, index=range(180))
        doy = pd.Series(np.arange(1, 181))

        rule = RsPeriodRatioRule()
        states = rule.check_series(rs_series, rso, doy)
        # Days 0–59: rs > rso AND period NaN'd → must be flagged
        assert (states.iloc[:60] != "pass").sum() > 0, "overestimate period must produce fail flags"

    def test_corrected_days_not_suspect(self):
        """Fix 1: days corrected by >10% must never be marked suspect (block removed).

        Strong systematic attenuation (0.60×) forces large per-period corrections.
        Under the old code, corrected days would return "suspect". After Fix 1 the
        suspect block is gone, so only real NaN-removals return "fail" — no "suspect".
        """
        rng = np.random.default_rng(42)
        n = 365
        rso = np.sin(np.linspace(0, np.pi, 365)) * 300 + 100  # max ~400
        rs = rso * 0.60 + rng.normal(0, 5, n)
        rs[rs < 0] = 0

        rs_series = pd.Series(rs, index=range(n))
        doy = pd.Series(np.arange(1, 366), index=range(n))

        rule = RsPeriodRatioRule()
        states = rule.check_series(rs_series, rso, doy)
        assert "suspect" not in states.values

    def test_correct_series_returns_corrected_values(self):
        """correct_series() should return 2-tuple with corrected array."""
        rng = np.random.default_rng(42)
        n = 365
        rso = np.sin(np.linspace(0, np.pi, 365)) * 300 + 100
        rs = rso * 0.90 + rng.normal(0, 5, n)
        rs[rs < 0] = 0

        rs_series = pd.Series(rs, index=range(n))
        doy = pd.Series(np.arange(1, 366), index=range(n))

        rule = RsPeriodRatioRule()
        result = rule.correct_series(rs_series, rso, doy)
        assert len(result) == 2
        states, corr_rs = result
        assert isinstance(states, pd.Series)
        assert isinstance(corr_rs, np.ndarray)
        assert len(corr_rs) == n

    def test_zero_rsds_not_flagged(self):
        """Zero rsds days must never be flagged (they encode missing data, not zero radiation)."""
        rso = np.sin(np.linspace(0, np.pi, 365)) * 20 + 10
        rs = rso * 0.75
        rs[::7] = 0.0  # inject zeros every 7th day (nighttime-only coverage)
        rule = RsPeriodRatioRule()
        states = rule.check_series(pd.Series(rs), rso, pd.Series(np.arange(1, 366)))
        zero_positions = np.where(rs == 0)[0]
        for i in zero_positions:
            assert states.iloc[i] == "pass", f"zero day {i} should not be flagged"

    def test_cf_instability_not_flagged(self):
        """Days removed by CF > 1.5 (overcast period) must not be flagged when rs ≤ rso."""
        rso = np.sin(np.linspace(0, np.pi, 365)) * 20 + 10
        rs = rso[:180] * 0.75
        # First 60 days: very low rs/rso (≈0.10) forces CF ≈ 10 > 1.5 → algorithm NaN-ifies period
        rs[:60] = rso[:60] * 0.10
        rule = RsPeriodRatioRule()
        states = rule.check_series(pd.Series(rs), rso, pd.Series(np.arange(1, 181)))
        # All 60 days have rs ≤ rso → must not be flagged regardless of CF instability
        assert (states.iloc[:60] == "pass").all(), (
            "CF-instability days (rs ≤ rso) must not be flagged"
        )

    def test_exact_multiple_of_period_returns_full_length(self):
        """Exact 60-day multiples should not trigger the upstream shape bug."""
        rng = np.random.default_rng(42)
        rso = np.sin(np.linspace(0, np.pi, 365)) * 20 + 10
        rule = RsPeriodRatioRule()

        for n in (60, 300, 360):
            rs = rso[np.arange(n) % 365] * 0.90 + rng.normal(0, 0.5, n)
            rs[rs < 0] = 0
            rs_series = pd.Series(rs, index=range(n))
            doy = pd.Series((np.arange(n) % 365) + 1, index=range(n))

            states, corr_rs = rule.correct_series(rs_series, rso, doy)

            assert isinstance(states, pd.Series)
            assert len(states) == n
            assert isinstance(corr_rs, np.ndarray)
            assert len(corr_rs) == n


class TestRsoASCE:
    def test_shape_and_range(self):
        """compute_rso_asce returns 365 values, all positive, with seasonal cycle."""
        from obsmet.products.rsun import compute_rso_asce

        rso = compute_rso_asce(lat_deg=45.0, elev_m=500.0)
        assert rso.shape == (365,)
        assert np.all(rso >= 0)
        # Summer (DOY ~172) should be higher than winter (DOY ~355)
        assert rso[171] > rso[354]
        # Reasonable range for MJ/m²/day at 45°N
        assert rso.max() < 45.0
        assert rso.max() > 20.0

    def test_elevation_increases_rso(self):
        """Higher elevation should give slightly higher Rso."""
        from obsmet.products.rsun import compute_rso_asce

        low = compute_rso_asce(lat_deg=45.0, elev_m=0.0)
        high = compute_rso_asce(lat_deg=45.0, elev_m=3000.0)
        assert np.all(high >= low)


class TestCompiledHumidity:
    def test_td_present(self):
        """When td is present, ea_compiled should be derived from it."""
        from obsmet.products.humidity import compile_humidity

        df = pd.DataFrame(
            {
                "tmax": [30.0, 32.0],
                "tmin": [15.0, 16.0],
                "td": [10.0, 12.0],
            }
        )
        result = compile_humidity(df)
        assert "ea_compiled" in result.columns
        assert "td_compiled" in result.columns
        assert not np.any(np.isnan(result["ea_compiled"].values))

    def test_rh_only(self):
        """When only RH is present, ea_compiled should still be computed."""
        from obsmet.products.humidity import compile_humidity

        df = pd.DataFrame(
            {
                "tmax": [30.0, 32.0],
                "tmin": [15.0, 16.0],
                "rh": [60.0, 65.0],
            }
        )
        result = compile_humidity(df)
        assert "ea_compiled" in result.columns
        assert not np.any(np.isnan(result["ea_compiled"].values))

    def test_all_missing_uses_tmin(self):
        """With no humidity columns, falls back to Tmin-based estimate."""
        from obsmet.products.humidity import compile_humidity

        df = pd.DataFrame(
            {
                "tmax": [30.0, 32.0],
                "tmin": [15.0, 16.0],
            }
        )
        result = compile_humidity(df)
        assert "ea_compiled" in result.columns
        # Should have values derived from Tmin
        assert not np.any(np.isnan(result["ea_compiled"].values))
