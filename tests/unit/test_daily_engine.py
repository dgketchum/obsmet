"""Tests for aggregate_daily_wide() in time_policy.py."""

import numpy as np
import pandas as pd
import pytest

from obsmet.core.provenance import RunProvenance
from obsmet.core.time_policy import (
    DAILY_AGG_MAP,
    DAILY_SOURCES,
    aggregate_daily_wide,
)


@pytest.fixture
def provenance():
    return RunProvenance(source="test", command="test", run_id="test123")


def _make_hourly(station_key, date_str, n_hours=24, **var_values):
    """Helper: build a wide-form hourly DataFrame for one station-day."""
    times = pd.date_range(date_str, periods=n_hours, freq="h", tz="UTC")
    df = pd.DataFrame({"station_key": station_key, "datetime_utc": times})
    for var, val in var_values.items():
        if isinstance(val, (list, np.ndarray)):
            df[var] = val[:n_hours]
        else:
            df[var] = val
    return df


class TestAggregateConstants:
    def test_daily_sources_list(self):
        assert "madis" in DAILY_SOURCES
        assert "ghcnh" in DAILY_SOURCES
        assert "gdas" in DAILY_SOURCES
        assert "ndbc" in DAILY_SOURCES

    def test_daily_agg_map_keys(self):
        assert "tair" in DAILY_AGG_MAP
        assert "prcp" in DAILY_AGG_MAP
        assert "wind_dir" in DAILY_AGG_MAP
        assert DAILY_AGG_MAP["prcp"] == "sum"
        assert DAILY_AGG_MAP["wind_dir"] == "circular_mean"


class TestAggregateDailyWide:
    def test_basic_mean(self, provenance):
        df = _make_hourly("test:001", "2024-01-15", tair=20.0)
        result = aggregate_daily_wide(df, provenance)
        assert len(result) == 1
        assert result.loc[0, "station_key"] == "test:001"
        assert result.loc[0, "tair"] == pytest.approx(20.0)
        assert result.loc[0, "day_basis"] == "utc"

    def test_tmax_tmin_tmean(self, provenance):
        tair_vals = list(np.linspace(5.0, 25.0, 24))
        df = _make_hourly("test:001", "2024-01-15", tair=tair_vals)
        result = aggregate_daily_wide(df, provenance)
        assert result.loc[0, "tmax"] == pytest.approx(25.0)
        assert result.loc[0, "tmin"] == pytest.approx(5.0)
        assert result.loc[0, "tmean"] == pytest.approx(15.0, abs=0.1)

    def test_prcp_sum(self, provenance):
        prcp_vals = [1.0] * 24
        df = _make_hourly("test:001", "2024-01-15", prcp=prcp_vals)
        result = aggregate_daily_wide(df, provenance)
        assert result.loc[0, "prcp"] == pytest.approx(24.0)

    def test_wind_dir_circular(self, provenance):
        wind_dirs = [350.0, 10.0, 355.0, 5.0] + [0.0] * 20
        df = _make_hourly("test:001", "2024-01-15", wind_dir=wind_dirs)
        result = aggregate_daily_wide(df, provenance)
        assert result.loc[0, "wind_dir"] % 360 == pytest.approx(0.0, abs=3.0)

    def test_rsds_hourly_conversion(self, provenance):
        """rsds_hourly W/m^2 → rsds MJ/m^2/day."""
        df = _make_hourly("test:001", "2024-01-15", rsds_hourly=200.0)
        result = aggregate_daily_wide(df, provenance)
        expected = 200.0 * 86400.0 / 1e6
        assert result.loc[0, "rsds"] == pytest.approx(expected)

    def test_multiple_stations(self, provenance):
        df1 = _make_hourly("test:001", "2024-01-15", tair=10.0)
        df2 = _make_hourly("test:002", "2024-01-15", tair=20.0)
        df = pd.concat([df1, df2], ignore_index=True)
        result = aggregate_daily_wide(df, provenance)
        assert len(result) == 2
        stns = set(result["station_key"])
        assert stns == {"test:001", "test:002"}

    def test_multiple_days(self, provenance):
        df1 = _make_hourly("test:001", "2024-01-15", tair=10.0)
        df2 = _make_hourly("test:001", "2024-01-16", tair=20.0)
        df = pd.concat([df1, df2], ignore_index=True)
        result = aggregate_daily_wide(df, provenance)
        assert len(result) == 2

    def test_empty_input(self, provenance):
        result = aggregate_daily_wide(pd.DataFrame(), provenance)
        assert result.empty

    def test_missing_datetime(self, provenance):
        df = pd.DataFrame({"station_key": ["a"], "tair": [10.0]})
        result = aggregate_daily_wide(df, provenance)
        assert result.empty

    def test_missing_station_key(self, provenance):
        df = pd.DataFrame(
            {"datetime_utc": pd.date_range("2024-01-15", periods=1, freq="h", tz="UTC")}
        )
        result = aggregate_daily_wide(df, provenance)
        assert result.empty

    def test_provenance_stamps(self, provenance):
        df = _make_hourly("test:001", "2024-01-15", tair=10.0)
        result = aggregate_daily_wide(df, provenance)
        assert result.loc[0, "qc_rules_version"] == provenance.qaqc_rules_version
        assert result.loc[0, "transform_version"] == provenance.transform_version
        assert result.loc[0, "ingest_run_id"] == "test123"

    def test_coverage_flags(self, provenance):
        df = _make_hourly("test:001", "2024-01-15", n_hours=24, tair=10.0)
        result = aggregate_daily_wide(df, provenance)
        flags = result.loc[0, "coverage_flags"]
        assert "n=24" in flags
        assert "thresh=Y" in flags

    def test_custom_agg_map(self, provenance):
        df = _make_hourly("test:001", "2024-01-15", tair=10.0)
        result = aggregate_daily_wide(df, provenance, agg_map={"tair": "sum"})
        assert result.loc[0, "tair"] == pytest.approx(240.0)

    def test_nan_values_ignored(self, provenance):
        tair_vals = [10.0, np.nan, 20.0] + [15.0] * 21
        df = _make_hourly("test:001", "2024-01-15", tair=tair_vals)
        result = aggregate_daily_wide(df, provenance)
        assert pd.notna(result.loc[0, "tair"])
        assert result.loc[0, "obs_count"] == 24
