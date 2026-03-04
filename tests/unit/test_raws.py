"""Tests for obsmet.sources.raws_wrcc extraction, parsing, and normalization."""

import numpy as np
import pandas as pd
import pytest

from obsmet.core.provenance import RunProvenance
from obsmet.sources.raws_wrcc.adapter import (
    normalize_to_canonical_wide,
)
from obsmet.sources.raws_wrcc.download import _parse_dms
from obsmet.sources.raws_wrcc.extract import parse_response


# --------------------------------------------------------------------------- #
# Fixture: synthetic RAWS data
# --------------------------------------------------------------------------- #


def _make_raws_html(n: int = 5, start_date: str = "01/15/2024") -> str:
    """Create synthetic WRCC HTML response with data rows."""
    lines = ["<PRE>"]
    base = pd.Timestamp(start_date)
    for i in range(n):
        dt = base + pd.Timedelta(days=i)
        date_str = dt.strftime("%m/%d/%Y")
        # date year doy day_of_run srad wspd wdir gust tave tmax tmin rhave rhmax rhmin prcp
        fields = [
            date_str,
            str(dt.year),
            str(dt.day_of_year),
            str(i + 1),
            f"{5.0 + i * 0.1:.1f}",  # srad
            f"{3.5 + i * 0.2:.1f}",  # wspd
            f"{180 + i * 10}",  # wdir
            f"{8.0 + i * 0.3:.1f}",  # gust
            f"{10.0 + i * 0.5:.1f}",  # tave
            f"{15.0 + i * 0.5:.1f}",  # tmax
            f"{5.0 + i * 0.5:.1f}",  # tmin
            f"{45.0 + i * 1.0:.1f}",  # rhave
            f"{70.0 + i * 1.0:.1f}",  # rhmax
            f"{20.0 + i * 1.0:.1f}",  # rhmin
            f"{0.0 + i * 0.5:.1f}",  # prcp
        ]
        lines.append("  ".join(fields))
    lines.append("</PRE>")
    return "\n".join(lines)


def _make_raws_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic RAWS-parsed DataFrame."""
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2024-01-15")
    return pd.DataFrame(
        {
            "date": [base_date + pd.Timedelta(days=i) for i in range(n)],
            "year": [2024] * n,
            "doy": list(range(15, 15 + n)),
            "day_of_run": list(range(1, n + 1)),
            "srad_total_kwh_m2": rng.uniform(2, 8, n),
            "wspd_ave_ms": rng.uniform(0, 10, n),
            "wdir_vec_deg": rng.uniform(0, 360, n),
            "wspd_gust_ms": rng.uniform(5, 20, n),
            "tair_ave_c": rng.uniform(-5, 25, n),
            "tair_max_c": rng.uniform(5, 35, n),
            "tair_min_c": rng.uniform(-15, 15, n),
            "rh_ave_pct": rng.uniform(20, 80, n),
            "rh_max_pct": rng.uniform(50, 100, n),
            "rh_min_pct": rng.uniform(5, 40, n),
            "prcp_total_mm": rng.uniform(0, 10, n),
        }
    )


# --------------------------------------------------------------------------- #
# DMS parsing tests
# --------------------------------------------------------------------------- #


class TestParseDms:
    def test_standard_format(self):
        # 45°27'25"
        result = _parse_dms("45\u00b027'25\"")
        assert result == pytest.approx(45.0 + 27 / 60 + 25 / 3600, abs=0.001)

    def test_numeric_string(self):
        result = _parse_dms("45.456")
        assert result == pytest.approx(45.456)

    def test_invalid_returns_none(self):
        assert _parse_dms("not a number") is None


# --------------------------------------------------------------------------- #
# Response parsing tests
# --------------------------------------------------------------------------- #


class TestParseResponse:
    def test_basic_parse(self):
        html = _make_raws_html(n=5)
        df = parse_response(html)
        assert len(df) == 5
        assert "date" in df.columns
        assert "tair_ave_c" in df.columns

    def test_missing_value_replacement(self):
        html = _make_raws_html(n=1)
        # Inject a -9999 missing value
        html = html.replace("5.0  3.5", "-9999  3.5")
        df = parse_response(html)
        assert pd.isna(df.loc[0, "srad_total_kwh_m2"])
        # Other values should be fine
        assert df.loc[0, "wspd_ave_ms"] == pytest.approx(3.5)

    def test_empty_response(self):
        df = parse_response("")
        assert df.empty

    def test_error_response(self):
        df = parse_response("Improper program call")
        assert df.empty

    def test_short_row_padding(self):
        """Short rows should be padded with -9999 then replaced with NaN."""
        html = "<PRE>\n01/15/2024  2024  15  1  5.0  3.5  180\n</PRE>"
        df = parse_response(html)
        assert len(df) == 1
        # Fields after wdir should be NaN (padded from -9999)
        assert pd.isna(df.loc[0, "wspd_gust_ms"])

    def test_duplicate_removal(self):
        html = _make_raws_html(n=2)
        # Duplicate the first line
        lines = html.split("\n")
        lines.insert(2, lines[1])
        html = "\n".join(lines)
        df = parse_response(html)
        assert len(df) == 2


# --------------------------------------------------------------------------- #
# Normalization tests
# --------------------------------------------------------------------------- #


class TestRawsNormalization:
    def test_output_columns(self):
        df = _make_raws_df()
        prov = RunProvenance(source="raws_wrcc")
        wide = normalize_to_canonical_wide(df, "orTEST", prov, latitude=44.0, longitude=-121.0)
        assert "station_key" in wide.columns
        assert "tmean" in wide.columns
        assert "tmax" in wide.columns
        assert "tmin" in wide.columns
        assert "wind" in wide.columns
        assert "prcp" in wide.columns
        assert "rsds" in wide.columns

    def test_station_key(self):
        df = _make_raws_df(n=1)
        prov = RunProvenance(source="raws_wrcc")
        wide = normalize_to_canonical_wide(df, "orTEST", prov)
        assert wide.loc[0, "station_key"] == "raws:orTEST"

    def test_values_unchanged(self):
        """RAWS data is already metric — no conversion needed."""
        df = _make_raws_df(n=1)
        df.loc[0, "tair_ave_c"] = 22.5
        df.loc[0, "prcp_total_mm"] = 3.2
        prov = RunProvenance(source="raws_wrcc")
        wide = normalize_to_canonical_wide(df, "orTEST", prov)
        assert wide.loc[0, "tmean"] == pytest.approx(22.5)
        assert wide.loc[0, "prcp"] == pytest.approx(3.2)

    def test_coordinates(self):
        df = _make_raws_df(n=1)
        prov = RunProvenance(source="raws_wrcc")
        wide = normalize_to_canonical_wide(
            df, "orTEST", prov, latitude=44.0, longitude=-121.5, elevation_m=1200.0
        )
        assert wide.loc[0, "lat"] == pytest.approx(44.0)
        assert wide.loc[0, "lon"] == pytest.approx(-121.5)
        assert wide.loc[0, "elev_m"] == pytest.approx(1200.0)
