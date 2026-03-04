"""Tests for daily aggregation and circular wind direction mean."""

import numpy as np
import pandas as pd
import pytest

from obsmet.core.time_policy import aggregate_daily, circular_mean_deg


class TestCircularMeanDeg:
    def test_simple_mean(self):
        """Non-wrapping angles should produce a simple mean."""
        angles = pd.Series([10.0, 20.0, 30.0])
        assert circular_mean_deg(angles) == pytest.approx(20.0, abs=0.1)

    def test_wraparound(self):
        """Angles near 360/0 should wrap correctly."""
        angles = pd.Series([350.0, 10.0])
        result = circular_mean_deg(angles)
        # 0° and 360° are equivalent for wind direction
        assert result % 360 == pytest.approx(0.0, abs=0.1)

    def test_north_wraparound(self):
        """More angles around north."""
        angles = pd.Series([355.0, 5.0, 10.0, 350.0])
        result = circular_mean_deg(angles)
        assert result % 360 == pytest.approx(0.0, abs=1.0)

    def test_opposite_directions(self):
        """180 and 0 should average to 90 or 270 depending on convention.
        With atan2: sin(0)+sin(pi)=0, cos(0)+cos(pi)=0 → atan2(0,0)=0."""
        angles = pd.Series([0.0, 180.0])
        # Result is technically undefined (zero vector), atan2(0,0)=0
        result = circular_mean_deg(angles)
        # Just verify it returns a valid angle
        assert 0 <= result < 360

    def test_south(self):
        """Angles around south (180)."""
        angles = pd.Series([170.0, 180.0, 190.0])
        assert circular_mean_deg(angles) == pytest.approx(180.0, abs=0.1)

    def test_empty_series(self):
        assert np.isnan(circular_mean_deg(pd.Series([], dtype=float)))

    def test_with_nans(self):
        """NaN values should be ignored."""
        angles = pd.Series([10.0, np.nan, 20.0, np.nan, 30.0])
        assert circular_mean_deg(angles) == pytest.approx(20.0, abs=0.1)


class TestAggregateDailyWindDir:
    def test_wind_dir_uses_circular_mean(self):
        """Wind direction aggregation should use circular mean."""
        times = pd.date_range("2024-01-15", periods=4, freq="6h", tz="UTC")
        values = pd.Series([350.0, 10.0, 355.0, 5.0])
        result = aggregate_daily(values, times, "wind_dir")
        # All in same UTC day, circular mean of ~0°
        assert len(result) == 1
        assert result.loc[0, "value"] % 360 == pytest.approx(0.0, abs=2.0)

    def test_rsds_converts_wm2_to_mj(self):
        """Solar radiation should convert mean W/m² to MJ/m²/day."""
        times = pd.date_range("2024-01-15", periods=4, freq="6h", tz="UTC")
        values = pd.Series([100.0, 200.0, 300.0, 100.0])
        result = aggregate_daily(values, times, "rsds")
        # mean = 175 W/m², daily = 175 * 86400 / 1e6 = 15.12 MJ/m²/day
        assert result.loc[0, "value"] == pytest.approx(175.0 * 86400 / 1e6)
