"""Integration tests: normalize + daily for all sources with real data.

Tests each source adapter end-to-end:
- ISD: reads from /nas/climate/isd/raw
- GDAS: reads from /nas/climate/gdas/prepbufr (requires ncepbufr)
- MADIS: reads from /tmp ingest output (requires MADIS credentials + wget)
- NDBC: reads from /tmp ingest output (requires network)
- RAWS: reads from /tmp ingest output (requires network)

Each source test: discover → normalize_key → schema check → physical bounds → daily aggregation.
"""

from pathlib import Path

import pandas as pd
import pytest

from obsmet.core.provenance import RunProvenance
from obsmet.core.time_policy import aggregate_daily_wide

# --------------------------------------------------------------------------- #
# Shared fixtures and helpers
# --------------------------------------------------------------------------- #

DAILY_CORE_COLS = {
    "station_key",
    "date",
    "day_basis",
    "obs_count",
    "coverage_flags",
    "qc_state",
    "ingest_run_id",
}


def _check_schema(df, source_prefix):
    """Common schema assertions."""
    assert "station_key" in df.columns
    assert "datetime_utc" in df.columns or "date" in df.columns
    assert "ingest_run_id" in df.columns
    assert df["station_key"].str.startswith(f"{source_prefix}:").all()


def _check_bounds(df, col, lo, hi):
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if not vals.empty:
            assert vals.min() > lo, f"{col} min {vals.min()} below {lo}"
            assert vals.max() < hi, f"{col} max {vals.max()} above {hi}"


# --------------------------------------------------------------------------- #
# ISD
# --------------------------------------------------------------------------- #

ISD_RAW = Path("/nas/climate/isd/raw")
ISD_KEYS = ["2024/720110-53983-2024.gz", "2024/720113-54829-2024.gz"]


class TestIsdEndToEnd:
    @pytest.fixture(autouse=True)
    def _check(self):
        for k in ISD_KEYS:
            if not (ISD_RAW / k).exists():
                pytest.skip("ISD raw data not found")

    @pytest.fixture(scope="class")
    def adapter(self):
        from obsmet.sources.isd.adapter import IsdAdapter

        return IsdAdapter(raw_dir=str(ISD_RAW))

    @pytest.fixture(scope="class")
    def prov(self):
        return RunProvenance(source="isd", command="inttest")

    def test_normalize_key(self, adapter, prov):
        df = adapter.normalize_key(ISD_KEYS[0], prov)
        assert df is not None and not df.empty
        _check_schema(df, "isd")

    def test_output_filename(self, adapter):
        fn = adapter.output_filename(ISD_KEYS[0])
        assert fn.endswith(".parquet")
        assert "/" not in fn

    def test_physical_bounds(self, adapter, prov):
        df = adapter.normalize_key(ISD_KEYS[0], prov)
        _check_bounds(df, "tair", -90, 65)
        _check_bounds(df, "wind", -1, 120)
        _check_bounds(df, "slp", 80000, 110000)

    def test_daily_aggregation(self, adapter, prov):
        df = adapter.normalize_key(ISD_KEYS[0], prov)
        daily = aggregate_daily_wide(df, prov)
        assert not daily.empty
        assert DAILY_CORE_COLS.issubset(daily.columns)
        if "tmax" in daily.columns and "tmin" in daily.columns:
            valid = daily.dropna(subset=["tmax", "tmin"])
            assert (valid["tmax"] >= valid["tmin"]).all()

    def test_multi_file(self, adapter, prov):
        total = 0
        for k in ISD_KEYS:
            df = adapter.normalize_key(k, prov)
            assert df is not None and not df.empty
            total += len(df)
        assert total > 100


# --------------------------------------------------------------------------- #
# GDAS
# --------------------------------------------------------------------------- #

GDAS_RAW = Path("/nas/climate/gdas/prepbufr")
GDAS_DAYS = ["20240101", "20240102"]


class TestGdasEndToEnd:
    @pytest.fixture(autouse=True)
    def _check(self):
        try:
            import ncepbufr  # noqa: F401
        except ImportError:
            pytest.skip("ncepbufr not installed")
        for d in GDAS_DAYS:
            tar = GDAS_RAW / d[:4] / f"prepbufr.{d}.nr.tar.gz"
            if not tar.exists():
                pytest.skip(f"GDAS raw not found: {tar}")

    @pytest.fixture(scope="class")
    def gdas_df(self):
        """Normalize one GDAS day (cached across all tests in this class)."""
        try:
            import ncepbufr  # noqa: F401
        except ImportError:
            pytest.skip("ncepbufr not installed")
        from obsmet.sources.gdas_prepbufr.adapter import GdasAdapter

        adapter = GdasAdapter(raw_dir=str(GDAS_RAW))
        prov = RunProvenance(source="gdas", command="inttest")
        return adapter.normalize_key(GDAS_DAYS[0], prov)

    def test_normalize_key(self, gdas_df):
        assert gdas_df is not None and not gdas_df.empty
        _check_schema(gdas_df, "gdas")

    def test_has_qm_columns(self, gdas_df):
        qm = [c for c in gdas_df.columns if c.endswith("_qm")]
        assert len(qm) >= 1

    def test_physical_bounds(self, gdas_df):
        # GDAS raw data includes unchecked obs; verify most are reasonable
        if "tair" in gdas_df.columns:
            tair = pd.to_numeric(gdas_df["tair"], errors="coerce").dropna()
            pct_extreme = ((tair < -90) | (tair > 65)).sum() / len(tair)
            assert pct_extreme < 0.001, f"{pct_extreme:.4%} of tair values outside [-90, 65]"

    def test_multiple_cycles(self, gdas_df):
        if "cycle" in gdas_df.columns:
            assert len(gdas_df["cycle"].unique()) >= 2

    def test_row_count(self, gdas_df):
        assert len(gdas_df) > 1000


# --------------------------------------------------------------------------- #
# MADIS
# --------------------------------------------------------------------------- #

MADIS_RAW = Path("/tmp/obsmet_inttest_madis_raw")
MADIS_DAYS = ["20240101", "20240102"]


class TestMadisEndToEnd:
    @pytest.fixture(autouse=True)
    def _check(self):
        for d in MADIS_DAYS:
            expected = MADIS_RAW / f"{d}_0000.gz"
            if not expected.exists():
                pytest.skip(f"MADIS raw data not found at {MADIS_RAW} (run MADIS ingest first)")

    @pytest.fixture(scope="class")
    def madis_df(self):
        """Normalize one MADIS day (cached across all tests in this class)."""
        from obsmet.sources.madis.adapter import MadisAdapter

        if not (MADIS_RAW / f"{MADIS_DAYS[0]}_0000.gz").exists():
            pytest.skip("MADIS raw data not found")
        adapter = MadisAdapter(raw_dir=str(MADIS_RAW))
        prov = RunProvenance(source="madis", command="inttest")
        return adapter.normalize_key(MADIS_DAYS[0], prov)

    def test_normalize_key(self, madis_df):
        assert madis_df is not None and not madis_df.empty
        _check_schema(madis_df, "madis")

    def test_physical_bounds(self, madis_df):
        _check_bounds(madis_df, "tair", -90, 65)
        _check_bounds(madis_df, "rh", -1, 110)

    def test_daily_aggregation(self, madis_df):
        prov = RunProvenance(source="madis", command="inttest")
        daily = aggregate_daily_wide(madis_df, prov)
        assert not daily.empty
        assert DAILY_CORE_COLS.issubset(daily.columns)

    def test_has_provider(self, madis_df):
        assert "provider" in madis_df.columns


# --------------------------------------------------------------------------- #
# NDBC
# --------------------------------------------------------------------------- #

NDBC_RAW = Path("/tmp/obsmet_inttest_ndbc_raw")


class TestNdbcEndToEnd:
    @pytest.fixture(autouse=True)
    def _check(self):
        if not list(NDBC_RAW.glob("*.txt.gz")):
            pytest.skip(f"NDBC raw data not found at {NDBC_RAW} (run NDBC ingest first)")

    @pytest.fixture(scope="class")
    def adapter(self):
        from obsmet.sources.ndbc.adapter import NdbcAdapter

        return NdbcAdapter(raw_dir=str(NDBC_RAW))

    @pytest.fixture(scope="class")
    def prov(self):
        return RunProvenance(source="ndbc", command="inttest")

    def test_discover_keys(self, adapter):
        keys = adapter.discover_keys(None, None)
        assert len(keys) > 0

    def test_normalize_key(self, adapter, prov):
        keys = adapter.discover_keys(None, None)
        df = adapter.normalize_key(keys[0], prov)
        assert df is not None and not df.empty
        _check_schema(df, "ndbc")

    def test_physical_bounds(self, adapter, prov):
        keys = adapter.discover_keys(None, None)
        df = adapter.normalize_key(keys[0], prov)
        _check_bounds(df, "tair", -90, 65)
        _check_bounds(df, "wind", -1, 120)

    def test_daily_aggregation(self, adapter, prov):
        keys = adapter.discover_keys(None, None)
        df = adapter.normalize_key(keys[0], prov)
        if "datetime_utc" in df.columns:
            daily = aggregate_daily_wide(df, prov)
            assert not daily.empty
            assert DAILY_CORE_COLS.issubset(daily.columns)


# --------------------------------------------------------------------------- #
# RAWS
# --------------------------------------------------------------------------- #

RAWS_RAW = Path("/tmp/obsmet_inttest_raws_raw")


class TestRawsEndToEnd:
    @pytest.fixture(autouse=True)
    def _check(self):
        if not list(RAWS_RAW.glob("*.parquet")):
            pytest.skip(f"RAWS raw data not found at {RAWS_RAW} (run RAWS ingest first)")

    @pytest.fixture(scope="class")
    def adapter(self):
        from obsmet.sources.raws_wrcc.adapter import RawsAdapter

        return RawsAdapter(raw_dir=str(RAWS_RAW))

    @pytest.fixture(scope="class")
    def prov(self):
        return RunProvenance(source="raws_wrcc", command="inttest")

    def test_discover_keys(self, adapter):
        keys = adapter.discover_keys(None, None)
        assert len(keys) > 0

    def test_normalize_key(self, adapter, prov):
        keys = adapter.discover_keys(None, None)
        df = adapter.normalize_key(keys[0], prov)
        assert df is not None and not df.empty
        _check_schema(df, "raws")

    def test_has_met_variables(self, adapter, prov):
        keys = adapter.discover_keys(None, None)
        df = adapter.normalize_key(keys[0], prov)
        met = {"tmean", "tmax", "tmin", "wind", "prcp"} & set(df.columns)
        assert len(met) >= 2

    def test_output_filename(self, adapter):
        keys = adapter.discover_keys(None, None)
        assert adapter.output_filename(keys[0]).endswith(".parquet")
