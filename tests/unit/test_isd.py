"""Tests for obsmet.sources.isd extraction, QC, and normalization."""

import numpy as np
import pandas as pd
import pytest

from obsmet.core.provenance import RunProvenance
from obsmet.qaqc.rules.isd import IsdQualityCodeRule
from obsmet.sources.isd.adapter import (
    extract_station_metadata,
    normalize_to_canonical_wide,
)
from obsmet.sources.isd.extract import apply_qc_mask, parse_line


# --------------------------------------------------------------------------- #
# Fixture: synthetic ISD line
# --------------------------------------------------------------------------- #

# Annotated example (from pyisd test data, Vance Brand airport):
# [0:4]   = "0165" (record length)
# [4:10]  = "720538" usaf_id
# [10:15] = "00164" ncei_id
# [15:19] = "2021" year
# [19:21] = "01" month
# [21:23] = "01" day
# [23:25] = "00" hour
# [25:27] = "15" minute
# [27]    = "4" data_source
# [28:34] = "+40167" latitude → 40.167
# [34:41] = "-105167" longitude → -105.167
# [41:46] = "FM-15" report_type
# [46:51] = "+1541" elevation → 1541.0
# [51:56] = "99999" call_letters
# [56:60] = "V020" qc_process
# [60:63] = "170" wind_direction → 170 deg
# [63]    = "1" wind_direction_qc
# [64]    = "N" wind_observation_type
# [65:69] = "0052" wind_speed → 5.2 m/s
# [69]    = "1" wind_speed_qc
# [70:75] = "03353" ceiling
# [75]    = "1" ceiling_qc
# [76:78] = "9N" ceiling_det + cavok
# [78:84] = "016093" visibility
# [84:87] = "199" vis_qc + vis_var + vis_var_qc
# [87:92] = "+0031" air_temperature → 3.1 °C
# [92]    = "1" air_temperature_qc
# [93:98] = "-0058" dew_point_temperature → -5.8 °C
# [98]    = "1" dew_point_temperature_qc
# [99:104]= "83150" sea_level_pressure → 8315.0/10 = 831.5 hPa
# [104]   = "1" sea_level_pressure_qc

_SAMPLE_LINE = (
    "0165"  # record length
    "720538"  # usaf
    "00164"  # wban
    "2021"  # year
    "01"  # month
    "01"  # day
    "00"  # hour
    "15"  # minute
    "4"  # data source
    "+40167"  # latitude
    "-105167"  # longitude
    "FM-15"  # report type
    "+1541"  # elevation
    "99999"  # call letters
    "V020"  # qc process
    "170"  # wind direction
    "1"  # wind_dir qc
    "N"  # wind obs type
    "0052"  # wind speed
    "1"  # wind_speed qc
    "03353"  # ceiling
    "1"  # ceiling qc
    "9"  # ceiling det
    "N"  # cavok
    "016093"  # visibility
    "1"  # vis qc
    "9"  # vis var code
    "9"  # vis var qc
    "+0031"  # air temp
    "1"  # air temp qc
    "-0058"  # dew point
    "1"  # dew point qc
    "83150"  # sea level pressure
    "1"  # slp qc
)


def _make_isd_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic ISD-parsed DataFrame."""
    rng = np.random.default_rng(seed)
    base_time = pd.Timestamp("2024-01-15 12:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "station_id": [f"72{i:04d}-0{i:04d}" for i in range(n)],
            "usaf_id": [f"72{i:04d}" for i in range(n)],
            "ncei_id": [f"0{i:04d}" for i in range(n)],
            "datetime_utc": [base_time + pd.Timedelta(hours=i) for i in range(n)],
            "latitude": rng.uniform(25, 50, n),
            "longitude": rng.uniform(-130, -65, n),
            "elevation": rng.uniform(0, 3000, n),
            "air_temperature": rng.uniform(-20, 40, n),
            "air_temperature_qc": ["1"] * n,
            "dew_point_temperature": rng.uniform(-30, 25, n),
            "dew_point_temperature_qc": ["1"] * n,
            "wind_speed": rng.uniform(0, 30, n),
            "wind_speed_qc": ["1"] * n,
            "wind_direction": rng.integers(0, 360, n).astype(float),
            "wind_direction_qc": ["1"] * n,
            "sea_level_pressure": rng.uniform(970, 1040, n),
            "sea_level_pressure_qc": ["1"] * n,
        }
    )


# --------------------------------------------------------------------------- #
# Parser tests
# --------------------------------------------------------------------------- #


class TestParseLine:
    def test_basic_parse(self):
        rec = parse_line(_SAMPLE_LINE)
        assert rec is not None
        assert rec["usaf_id"] == "720538"
        assert rec["ncei_id"] == "00164"
        assert rec["year"] == 2021
        assert rec["month"] == 1
        assert rec["day"] == 1
        assert rec["hour"] == 0
        assert rec["minute"] == 15

    def test_lat_lon(self):
        rec = parse_line(_SAMPLE_LINE)
        assert rec["latitude"] == pytest.approx(40.167)
        assert rec["longitude"] == pytest.approx(-105.167)

    def test_temperature(self):
        rec = parse_line(_SAMPLE_LINE)
        assert rec["air_temperature"] == pytest.approx(3.1)
        assert rec["air_temperature_qc"] == "1"

    def test_dew_point(self):
        rec = parse_line(_SAMPLE_LINE)
        assert rec["dew_point_temperature"] == pytest.approx(-5.8)
        assert rec["dew_point_temperature_qc"] == "1"

    def test_wind(self):
        rec = parse_line(_SAMPLE_LINE)
        assert rec["wind_direction"] == 170
        assert rec["wind_speed"] == pytest.approx(5.2)
        assert rec["wind_speed_qc"] == "1"

    def test_pressure(self):
        rec = parse_line(_SAMPLE_LINE)
        assert rec["sea_level_pressure"] == pytest.approx(8315.0)
        assert rec["sea_level_pressure_qc"] == "1"

    def test_sentinels(self):
        """Test that sentinel values produce None."""
        # Build a line with sentinel wind direction (999)
        line = list(_SAMPLE_LINE)
        # wind_direction is at positions 60:63
        line[60:63] = list("999")
        line_str = "".join(line)
        rec = parse_line(line_str)
        assert rec["wind_direction"] is None

    def test_short_line_returns_none(self):
        assert parse_line("too short") is None


# --------------------------------------------------------------------------- #
# QC mask tests
# --------------------------------------------------------------------------- #


class TestApplyQcMask:
    def test_all_good(self):
        df = _make_isd_df()
        result = apply_qc_mask(df)
        assert result["air_temperature"].notna().all()

    def test_bad_qc_code_masks_value(self):
        df = _make_isd_df()
        df.loc[0, "air_temperature_qc"] = "2"  # suspect
        df.loc[1, "air_temperature_qc"] = "9"  # missing
        result = apply_qc_mask(df)
        assert pd.isna(result.loc[0, "air_temperature"])
        assert pd.isna(result.loc[1, "air_temperature"])
        # Other rows unaffected
        assert result.loc[2, "air_temperature"] == df.loc[2, "air_temperature"]

    def test_wind_qc(self):
        df = _make_isd_df()
        df.loc[0, "wind_speed_qc"] = "9"
        result = apply_qc_mask(df)
        assert pd.isna(result.loc[0, "wind_speed"])


# --------------------------------------------------------------------------- #
# Normalization tests
# --------------------------------------------------------------------------- #


class TestNormalization:
    def test_wide_output_columns(self):
        df = _make_isd_df()
        prov = RunProvenance(source="isd")
        wide = normalize_to_canonical_wide(df, prov)
        assert "station_key" in wide.columns
        assert "tair" in wide.columns
        assert "td" in wide.columns
        assert "wind" in wide.columns
        assert "wind_dir" in wide.columns
        assert "slp" in wide.columns

    def test_station_key_prefix(self):
        df = _make_isd_df(n=1)
        prov = RunProvenance(source="isd")
        wide = normalize_to_canonical_wide(df, prov)
        assert wide.loc[0, "station_key"].startswith("isd:")

    def test_values_unchanged(self):
        """ISD temperature values are already in correct units."""
        df = _make_isd_df(n=1)
        df.loc[0, "air_temperature"] = 25.3
        prov = RunProvenance(source="isd")
        wide = normalize_to_canonical_wide(df, prov)
        assert wide.loc[0, "tair"] == pytest.approx(25.3)

    def test_pressure_hpa_to_pa(self):
        """ISD sea_level_pressure (hPa) should be converted to Pa."""
        df = _make_isd_df(n=1)
        df.loc[0, "sea_level_pressure"] = 1013.25
        prov = RunProvenance(source="isd")
        wide = normalize_to_canonical_wide(df, prov)
        assert wide.loc[0, "slp"] == pytest.approx(101325.0)


# --------------------------------------------------------------------------- #
# Station metadata tests
# --------------------------------------------------------------------------- #


class TestStationMetadata:
    def test_extract(self):
        df = _make_isd_df(n=5)
        prov = RunProvenance(source="isd")
        wide = normalize_to_canonical_wide(df, prov)
        meta = extract_station_metadata(wide)
        assert len(meta) == 5
        assert "station_key" in meta.columns
        assert meta["source"].iloc[0] == "isd"


# --------------------------------------------------------------------------- #
# Tier 0 QC rule tests
# --------------------------------------------------------------------------- #


class TestIsdQualityCodeRule:
    def test_pass_code_1(self):
        rule = IsdQualityCodeRule()
        result = rule.check(20.0, qc_code="1")
        assert result.state == "pass"

    def test_pass_code_5(self):
        rule = IsdQualityCodeRule()
        result = rule.check(20.0, qc_code="5")
        assert result.state == "pass"

    def test_fail_code_2(self):
        rule = IsdQualityCodeRule()
        result = rule.check(20.0, qc_code="2")
        assert result.state == "fail"
        assert "suspect" in result.reason

    def test_fail_code_9(self):
        rule = IsdQualityCodeRule()
        result = rule.check(20.0, qc_code="9")
        assert result.state == "fail"
        assert "missing" in result.reason

    def test_missing_code(self):
        rule = IsdQualityCodeRule()
        result = rule.check(20.0)
        assert result.state == "suspect"
