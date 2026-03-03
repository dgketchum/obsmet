"""Tests for obsmet.sources.madis extraction, QC, and normalization."""

import json

import numpy as np
import pandas as pd
import pytest

from obsmet.core.provenance import RunProvenance
from obsmet.qaqc.rules.madis import MadisDDRule, MadisQCRRule
from obsmet.sources.madis.adapter import (
    extract_station_metadata,
    normalize_to_canonical,
    normalize_to_canonical_wide,
)
from obsmet.sources.madis.extract import apply_qc


# --------------------------------------------------------------------------- #
# Fixture: synthetic MADIS-native DataFrame
# --------------------------------------------------------------------------- #


def _make_madis_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic MADIS extraction DataFrame."""
    rng = np.random.default_rng(seed)
    base_time = pd.Timestamp("2024-01-15 12:00:00")
    return pd.DataFrame(
        {
            "stationId": [f"STN{i:04d}" for i in range(n)],
            "latitude": rng.uniform(30, 48, n),
            "longitude": rng.uniform(-125, -70, n),
            "elevation": rng.uniform(0, 3000, n),
            "dataProvider": ["TestProvider"] * n,
            "observationTime": [base_time + pd.Timedelta(minutes=i) for i in range(n)],
            # MADIS native: temperatures in Kelvin
            "temperature": rng.uniform(260, 310, n),
            "dewpoint": rng.uniform(250, 290, n),
            "relHumidity": rng.uniform(20, 95, n),
            "windSpeed": rng.uniform(0, 15, n),
            "windDir": rng.uniform(0, 360, n),
            "precipAccum": rng.uniform(0, 5, n),
            "solarRadiation": rng.uniform(0, 800, n),
            # DD flags
            "temperatureDD": ["V"] * n,
            "dewpointDD": ["V"] * n,
            "relHumidityDD": ["V"] * n,
            "windSpeedDD": ["V"] * n,
            "windDirDD": ["V"] * n,
            # QCR bitmasks (all clear)
            "temperatureQCR": np.zeros(n),
            "dewpointQCR": np.zeros(n),
            "relHumidityQCR": np.zeros(n),
            "windSpeedQCR": np.zeros(n),
            "windDirQCR": np.zeros(n),
        }
    )


# --------------------------------------------------------------------------- #
# QC tests
# --------------------------------------------------------------------------- #


class TestApplyQC:
    def test_all_pass(self):
        df = _make_madis_df()
        result = apply_qc(df.copy())
        assert result["qc_passed"].all()

    def test_dd_rejection(self):
        df = _make_madis_df()
        df.loc[0, "temperatureDD"] = "X"  # rejected
        df.loc[1, "temperatureDD"] = "Z"  # no QC
        result = apply_qc(df.copy())
        assert pd.isna(result.loc[0, "temperature"])
        assert pd.isna(result.loc[1, "temperature"])
        # Other vars should still be fine for these rows
        assert result.loc[0, "qc_passed"]  # other vars survived

    def test_qcr_rejection(self):
        df = _make_madis_df()
        df.loc[0, "temperatureQCR"] = 2  # validity fail
        result = apply_qc(df.copy())
        assert pd.isna(result.loc[0, "temperature"])

    def test_bounds_rejection(self):
        df = _make_madis_df()
        df.loc[0, "temperature"] = 400.0  # way above 333.15 K
        df.loc[1, "windSpeed"] = 100.0  # way above 35 m/s
        result = apply_qc(df.copy())
        assert pd.isna(result.loc[0, "temperature"])
        assert pd.isna(result.loc[1, "windSpeed"])

    def test_all_vars_nan_fails_qc(self):
        df = _make_madis_df(n=1)
        for v in ["temperature", "dewpoint", "relHumidity", "windSpeed", "windDir"]:
            df.loc[0, f"{v}DD"] = "X"
        result = apply_qc(df.copy())
        assert not result.loc[0, "qc_passed"]


# --------------------------------------------------------------------------- #
# Normalization tests
# --------------------------------------------------------------------------- #


class TestNormalization:
    def test_wide_output_columns(self):
        df = _make_madis_df()
        df = apply_qc(df)
        prov = RunProvenance(source="madis")
        wide = normalize_to_canonical_wide(df, prov)
        assert "station_key" in wide.columns
        assert "tair" in wide.columns
        assert "td" in wide.columns
        assert "rh" in wide.columns
        assert "wind" in wide.columns
        assert "rsds_hourly" in wide.columns

    def test_wide_kelvin_to_celsius(self):
        df = _make_madis_df(n=1)
        df.loc[0, "temperature"] = 300.0  # 26.85 °C
        df = apply_qc(df)
        prov = RunProvenance(source="madis")
        wide = normalize_to_canonical_wide(df, prov)
        assert wide.loc[0, "tair"] == pytest.approx(26.85)

    def test_long_output_columns(self):
        df = _make_madis_df(n=2)
        df = apply_qc(df)
        prov = RunProvenance(source="madis")
        long = normalize_to_canonical(df, prov)
        assert "station_key" in long.columns
        assert "variable" in long.columns
        assert "value" in long.columns
        assert "unit" in long.columns
        assert "qc_flags_native" in long.columns
        # Should have multiple rows per station (one per variable)
        assert len(long) > 2

    def test_long_qc_flags_native(self):
        df = _make_madis_df(n=1)
        df = apply_qc(df)
        prov = RunProvenance(source="madis")
        long = normalize_to_canonical(df, prov)
        # temperature row should have DD and QCR in native flags
        temp_rows = long[long["variable"] == "tair"]
        assert len(temp_rows) == 1
        flags = json.loads(temp_rows.iloc[0]["qc_flags_native"])
        assert "dd" in flags
        assert flags["dd"] == "V"


# --------------------------------------------------------------------------- #
# Station metadata tests
# --------------------------------------------------------------------------- #


class TestStationMetadata:
    def test_extract_from_wide(self):
        df = _make_madis_df(n=5)
        df = apply_qc(df)
        prov = RunProvenance(source="madis")
        wide = normalize_to_canonical_wide(df, prov)
        meta = extract_station_metadata(wide)
        assert len(meta) == 5
        assert "station_key" in meta.columns
        assert "lat" in meta.columns
        assert meta["source"].iloc[0] == "madis"

    def test_extract_from_raw(self):
        df = _make_madis_df(n=3)
        meta = extract_station_metadata(df)
        assert len(meta) == 3


# --------------------------------------------------------------------------- #
# Tier 0 QC rule tests
# --------------------------------------------------------------------------- #


class TestMadisDDRule:
    def test_pass_verified(self):
        rule = MadisDDRule()
        result = rule.check(20.0, dd_flag="V")
        assert result.state == "pass"

    def test_pass_screened(self):
        rule = MadisDDRule()
        result = rule.check(20.0, dd_flag="S")
        assert result.state == "pass"

    def test_fail_excluded(self):
        rule = MadisDDRule()
        result = rule.check(20.0, dd_flag="X")
        assert result.state == "fail"

    def test_fail_no_qc(self):
        rule = MadisDDRule()
        result = rule.check(20.0, dd_flag="Z")
        assert result.state == "fail"

    def test_missing_flag(self):
        rule = MadisDDRule()
        result = rule.check(20.0)
        assert result.state == "suspect"


class TestMadisQCRRule:
    def test_pass_clear(self):
        rule = MadisQCRRule()
        result = rule.check(20.0, qcr_value=0)
        assert result.state == "pass"

    def test_fail_validity(self):
        rule = MadisQCRRule()
        result = rule.check(20.0, qcr_value=2)
        assert result.state == "fail"
        assert "validity" in result.detail

    def test_fail_temporal(self):
        rule = MadisQCRRule()
        result = rule.check(20.0, qcr_value=16)
        assert result.state == "fail"
        assert "temporal" in result.detail

    def test_pass_internal_consistency_only(self):
        """Bit 4 (=8, internal consistency) should not trigger rejection."""
        rule = MadisQCRRule()
        result = rule.check(20.0, qcr_value=8)
        assert result.state == "pass"

    def test_missing_qcr(self):
        rule = MadisQCRRule()
        result = rule.check(20.0)
        assert result.state == "pass"
