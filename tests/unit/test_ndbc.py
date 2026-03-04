"""Tests for obsmet.sources.ndbc extraction, parsing, and normalization."""

import textwrap

import numpy as np
import pandas as pd
import pytest

from obsmet.core.provenance import RunProvenance
from obsmet.qaqc.rules.ndbc import NdbcSentinelRule
from obsmet.sources.ndbc.adapter import (
    normalize_to_canonical_wide,
)
from obsmet.sources.ndbc.download import _parse_location
from obsmet.sources.ndbc.extract import read_stdmet_file


# --------------------------------------------------------------------------- #
# Fixture: synthetic NDBC data
# --------------------------------------------------------------------------- #

_SAMPLE_STDMET = textwrap.dedent("""\
    #YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE
    #yr  mo dy hr mn deg  m/s  m/s     m   sec   sec deg    hPa  degC  degC  degC   nmi    ft
    2024 01 15 00 00 180  5.2  7.1   1.2   8.0   5.5 170 1013.5  22.3  24.1  18.5  99.0  99.0
    2024 01 15 01 00 190  4.8  6.5   1.1   7.5   5.3 175 1013.8  22.1  24.0  18.2  99.0  99.0
    2024 01 15 02 00 200  5.5  7.8   1.3   8.5   5.8 180 1014.0  21.8  23.8  17.9  99.0  99.0
    2024 01 15 03 00 210  6.1  8.2   1.5   9.0   6.1 185 1014.2  21.5  23.5  17.5  99.0  99.0
    2024 01 15 04 00 220  5.8  7.5   1.4   8.2   5.9 190 1014.5  21.2  23.3  17.2  99.0  99.0
""")

# Missing values in various fields
_SAMPLE_MISSING = textwrap.dedent("""\
    #YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE
    #yr  mo dy hr mn deg  m/s  m/s     m   sec   sec deg    hPa  degC  degC  degC   nmi    ft
    2024 01 15 00 00 999  5.2  7.1   1.2   8.0   5.5 999 1013.5  22.3  24.1  18.5  99.0  99.0
    2024 01 15 01 00 190  999  6.5  99.0  99.0  99.0 175  999.0  999   999   999   99.0  99.0
""")

# 2-digit year format (older files)
_SAMPLE_2DIGIT = textwrap.dedent("""\
    YY MM DD hh WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE
    96 01 15 00  180  5.2  7.1   1.2   8.0   5.5 170 1013.5  22.3  24.1  18.5  99.0  99.0
    96 01 15 01  190  4.8  6.5   1.1   7.5   5.3 175 1013.8  22.1  24.0  18.2  99.0  99.0
""")


def _make_ndbc_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic NDBC-parsed DataFrame."""
    rng = np.random.default_rng(seed)
    base_time = pd.Timestamp("2024-01-15 00:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "datetime_utc": [base_time + pd.Timedelta(hours=i) for i in range(n)],
            "wind_dir": rng.uniform(0, 360, n),
            "wind_speed": rng.uniform(0, 15, n),
            "wind_gust": rng.uniform(5, 25, n),
            "wave_height": rng.uniform(0.5, 5, n),
            "dominant_wave_period": rng.uniform(5, 15, n),
            "average_wave_period": rng.uniform(3, 10, n),
            "mean_wave_dir": rng.uniform(0, 360, n),
            "pressure": rng.uniform(990, 1030, n),
            "air_temp": rng.uniform(10, 30, n),
            "water_temp": rng.uniform(15, 28, n),
            "dewpoint": rng.uniform(5, 25, n),
            "visibility": [np.nan] * n,
            "tide": [np.nan] * n,
        }
    )


# --------------------------------------------------------------------------- #
# Location parsing tests
# --------------------------------------------------------------------------- #


class TestParseLocation:
    def test_decimal_with_hemisphere(self):
        lat, lon = _parse_location("40.5 N 73.8 W")
        assert lat == pytest.approx(40.5)
        assert lon == pytest.approx(-73.8)

    def test_south_east(self):
        lat, lon = _parse_location("33.5 S 151.2 E")
        assert lat == pytest.approx(-33.5)
        assert lon == pytest.approx(151.2)

    def test_no_match(self):
        lat, lon = _parse_location("unknown location")
        assert lat is None
        assert lon is None


# --------------------------------------------------------------------------- #
# File parsing tests
# --------------------------------------------------------------------------- #


class TestReadStdmet:
    def _write_and_read(self, text, tmp_path, filename="test.txt"):
        path = tmp_path / filename
        path.write_text(text)
        return read_stdmet_file(path)

    def test_basic_parse(self, tmp_path):
        df = self._write_and_read(_SAMPLE_STDMET, tmp_path)
        assert len(df) == 5
        assert "datetime_utc" in df.columns
        assert "air_temp" in df.columns
        assert "wind_speed" in df.columns

    def test_values(self, tmp_path):
        df = self._write_and_read(_SAMPLE_STDMET, tmp_path)
        assert df.loc[0, "air_temp"] == pytest.approx(22.3)
        assert df.loc[0, "wind_speed"] == pytest.approx(5.2)
        assert df.loc[0, "pressure"] == pytest.approx(1013.5)
        assert df.loc[0, "wind_dir"] == pytest.approx(180.0)

    def test_missing_values(self, tmp_path):
        df = self._write_and_read(_SAMPLE_MISSING, tmp_path)
        assert len(df) == 2
        # wind_dir=999 should be NaN
        assert pd.isna(df.loc[0, "wind_dir"])
        # wind_speed=999 should be NaN
        assert pd.isna(df.loc[1, "wind_speed"])

    def test_2digit_year(self, tmp_path):
        df = self._write_and_read(_SAMPLE_2DIGIT, tmp_path)
        assert len(df) == 2
        # 96 → 1996
        assert df.loc[0, "datetime_utc"].year == 1996

    def test_empty_file(self, tmp_path):
        df = self._write_and_read("", tmp_path)
        assert df.empty

    def test_deduplication(self, tmp_path):
        # Double the data lines
        doubled = _SAMPLE_STDMET + "\n" + "\n".join(_SAMPLE_STDMET.split("\n")[2:])
        df = self._write_and_read(doubled, tmp_path)
        assert len(df) == 5  # Duplicates removed


# --------------------------------------------------------------------------- #
# Normalization tests
# --------------------------------------------------------------------------- #


class TestNdbcNormalization:
    def test_output_columns(self):
        df = _make_ndbc_df()
        prov = RunProvenance(source="ndbc")
        wide = normalize_to_canonical_wide(df, "41001", prov, latitude=34.7, longitude=-72.7)
        assert "station_key" in wide.columns
        assert "tair" in wide.columns
        assert "td" in wide.columns
        assert "wind" in wide.columns
        assert "slp" in wide.columns
        # Extension vars
        assert "wave_height" in wide.columns
        assert "water_temp" in wide.columns

    def test_station_key(self):
        df = _make_ndbc_df(n=1)
        prov = RunProvenance(source="ndbc")
        wide = normalize_to_canonical_wide(df, "41001", prov)
        assert wide.loc[0, "station_key"] == "ndbc:41001"

    def test_values_unchanged(self):
        """NDBC temperature values are already in standard units."""
        df = _make_ndbc_df(n=1)
        df.loc[0, "air_temp"] = 25.5
        prov = RunProvenance(source="ndbc")
        wide = normalize_to_canonical_wide(df, "41001", prov)
        assert wide.loc[0, "tair"] == pytest.approx(25.5)

    def test_pressure_hpa_to_pa(self):
        """NDBC pressure (hPa) should be converted to Pa."""
        df = _make_ndbc_df(n=1)
        df.loc[0, "pressure"] = 1013.25
        prov = RunProvenance(source="ndbc")
        wide = normalize_to_canonical_wide(df, "41001", prov)
        assert wide.loc[0, "slp"] == pytest.approx(101325.0)

    def test_coordinates(self):
        df = _make_ndbc_df(n=1)
        prov = RunProvenance(source="ndbc")
        wide = normalize_to_canonical_wide(df, "41001", prov, latitude=34.7, longitude=-72.7)
        assert wide.loc[0, "lat"] == pytest.approx(34.7)
        assert wide.loc[0, "lon"] == pytest.approx(-72.7)


# --------------------------------------------------------------------------- #
# Tier 0 sentinel QC rule tests
# --------------------------------------------------------------------------- #


class TestNdbcSentinelRule:
    def test_pass_normal_value(self):
        rule = NdbcSentinelRule()
        result = rule.check(22.5, variable="air_temp")
        assert result.state == "pass"

    def test_fail_99(self):
        rule = NdbcSentinelRule()
        result = rule.check(99.0, variable="wind_dir")
        assert result.state == "fail"
        assert "sentinel" in result.reason

    def test_fail_999(self):
        rule = NdbcSentinelRule()
        result = rule.check(999.0, variable="wind_speed")
        assert result.state == "fail"

    def test_fail_9999(self):
        rule = NdbcSentinelRule()
        result = rule.check(9999.0, variable="pressure")
        assert result.state == "fail"

    def test_fail_none(self):
        rule = NdbcSentinelRule()
        result = rule.check(None, variable="air_temp")
        assert result.state == "fail"
        assert "none" in result.reason.lower()

    def test_custom_sentinels(self):
        rule = NdbcSentinelRule(sentinels={-999.0, 0.0})
        assert rule.check(-999.0).state == "fail"
        assert rule.check(99.0).state == "pass"
