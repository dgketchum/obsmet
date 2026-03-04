"""Tests for obsmet.sources.gdas_prepbufr adapter and QC rules.

Note: BUFR extraction tests are integration-level (require py-ncepbufr).
These unit tests cover the adapter normalization and QM rule logic using
synthetic DataFrames that mimic extract output.
"""

import numpy as np
import pandas as pd
import pytest

from obsmet.core.provenance import RunProvenance
from obsmet.qaqc.rules.gdas import GdasQualityMarkerRule
from obsmet.sources.gdas_prepbufr.adapter import (
    normalize_to_canonical_wide,
)


# --------------------------------------------------------------------------- #
# Fixture: synthetic GDAS extraction DataFrame
# --------------------------------------------------------------------------- #


def _make_gdas_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic GDAS-extraction DataFrame."""
    rng = np.random.default_rng(seed)
    base_time = pd.Timestamp("2024-01-15 00:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "station_id": [f"STN{i:05d}" for i in range(n)],
            "latitude": rng.uniform(25, 60, n),
            "longitude": rng.uniform(-130, -65, n),
            "elevation": rng.uniform(0, 2000, n),
            "obs_type": rng.choice([181, 187, 180], n),
            "datetime_utc": [base_time + pd.Timedelta(hours=i) for i in range(n)],
            "cycle": [0] * n,
            "msg_type": rng.choice(["ADPSFC", "SFCSHP"], n),
            # Already converted: pressure in Pa, temperature in °C,
            # specific_humidity in kg/kg
            "pressure": rng.uniform(90000, 105000, n),
            "pressure_qm": rng.choice([0, 1, 2, 3, 9, 15], n),
            "temperature": rng.uniform(-20, 40, n),
            "temperature_qm": rng.choice([0, 1, 2], n),
            "specific_humidity": rng.uniform(0, 0.025, n),
            "humidity_qm": rng.choice([0, 1], n),
            "u_wind": rng.uniform(-15, 15, n),
            "v_wind": rng.uniform(-15, 15, n),
            "wind_qm": rng.choice([0, 1, 2], n),
            "height": rng.uniform(0, 2000, n),
            "height_qm": [1] * n,
            "sst": [np.nan] * n,
            "sst_qm": [None] * n,
        }
    )


# --------------------------------------------------------------------------- #
# Normalization tests
# --------------------------------------------------------------------------- #


class TestGdasNormalization:
    def test_output_columns(self):
        df = _make_gdas_df()
        prov = RunProvenance(source="gdas")
        wide = normalize_to_canonical_wide(df, prov)
        assert "station_key" in wide.columns
        assert "tair" in wide.columns
        assert "psfc" in wide.columns
        assert "u" in wide.columns
        assert "v" in wide.columns
        assert "q" in wide.columns

    def test_station_key_prefix(self):
        df = _make_gdas_df(n=1)
        prov = RunProvenance(source="gdas")
        wide = normalize_to_canonical_wide(df, prov)
        assert wide.loc[0, "station_key"].startswith("gdas:")

    def test_source_includes_msg_type(self):
        df = _make_gdas_df(n=1)
        df.loc[0, "msg_type"] = "ADPSFC"
        prov = RunProvenance(source="gdas")
        wide = normalize_to_canonical_wide(df, prov)
        assert wide.loc[0, "source"] == "gdas_adpsfc"

    def test_values_passthrough(self):
        """Values are already converted in the extract layer."""
        df = _make_gdas_df(n=1)
        df.loc[0, "temperature"] = 22.5
        df.loc[0, "pressure"] = 101325.0
        prov = RunProvenance(source="gdas")
        wide = normalize_to_canonical_wide(df, prov)
        assert wide.loc[0, "tair"] == pytest.approx(22.5)
        assert wide.loc[0, "psfc"] == pytest.approx(101325.0)

    def test_qm_columns_preserved(self):
        df = _make_gdas_df(n=1)
        df.loc[0, "temperature_qm"] = 1
        df.loc[0, "pressure_qm"] = 3
        prov = RunProvenance(source="gdas")
        wide = normalize_to_canonical_wide(df, prov)
        assert wide.loc[0, "tair_qm"] == 1
        assert wide.loc[0, "psfc_qm"] == 3


# --------------------------------------------------------------------------- #
# Tier 0 QM rule tests
# --------------------------------------------------------------------------- #


class TestGdasQualityMarkerRule:
    def test_pass_qm_0(self):
        rule = GdasQualityMarkerRule()
        result = rule.check(20.0, qm=0)
        assert result.state == "pass"

    def test_pass_qm_1(self):
        rule = GdasQualityMarkerRule()
        result = rule.check(20.0, qm=1)
        assert result.state == "pass"

    def test_pass_qm_2(self):
        rule = GdasQualityMarkerRule()
        result = rule.check(20.0, qm=2)
        assert result.state == "pass"

    def test_suspect_qm_3(self):
        rule = GdasQualityMarkerRule()
        result = rule.check(20.0, qm=3)
        assert result.state == "suspect"

    def test_fail_qm_4(self):
        rule = GdasQualityMarkerRule()
        result = rule.check(20.0, qm=4)
        assert result.state == "fail"

    def test_fail_qm_9(self):
        rule = GdasQualityMarkerRule()
        result = rule.check(20.0, qm=9)
        assert result.state == "fail"

    def test_fail_qm_15(self):
        rule = GdasQualityMarkerRule()
        result = rule.check(20.0, qm=15)
        assert result.state == "fail"
        assert "purged" in result.reason

    def test_missing_qm(self):
        rule = GdasQualityMarkerRule()
        result = rule.check(20.0)
        assert result.state == "pass"
