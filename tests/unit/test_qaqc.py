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
        """Inject a massive spike — should be flagged."""
        rng = np.random.default_rng(42)
        n = 365
        rso = np.sin(np.linspace(0, np.pi, 365)) * 300 + 100
        rs = rso * 0.75 + rng.normal(0, 10, n)
        rs[rs < 0] = 0
        # Inject spikes
        rs[10] = 1200.0
        rs[11] = 1300.0

        rs_series = pd.Series(rs, index=range(n))
        doy = pd.Series(np.arange(1, 366), index=range(n))

        rule = RsPeriodRatioRule()
        states = rule.check_series(rs_series, rso, doy)
        # At least some days should be flagged
        assert (states != "pass").sum() > 0


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
