"""Tests for obsmet.products.station_por."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obsmet.core.provenance import RunProvenance
from obsmet.products.station_por import (
    _apply_tier2_qc,
    _passthrough_daily_file,
    _station_por_variable_columns,
    build_station_por,
)


class TestBuildStationPor:
    def _make_hourly_df(self, station_key, n_hours=72):
        """Build synthetic hourly data for a station."""
        rng = np.random.default_rng(42)
        dt = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "station_key": station_key,
                "source": "madis",
                "source_station_id": station_key.split(":")[-1],
                "datetime_utc": dt,
                "tair": rng.normal(5, 3, n_hours),
                "td": rng.normal(0, 2, n_hours),
                "rh": rng.uniform(40, 80, n_hours),
                "wind": rng.uniform(0, 10, n_hours),
                "wind_dir": rng.uniform(0, 360, n_hours),
                "prcp": np.zeros(n_hours),
                "rsds_hourly": rng.uniform(0, 500, n_hours),
                "qc_state": "pass",
                "qc_reason_codes": "",
                "ingest_run_id": "test123",
                "transform_version": "0.1.0",
            }
        )

    def test_builds_one_parquet_per_station(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            # Write two stations
            df1 = self._make_hourly_df("madis:STN_A")
            df2 = self._make_hourly_df("madis:STN_B")
            combined = pd.concat([df1, df2], ignore_index=True)
            combined.to_parquet(norm_dir / "day1.parquet", index=False)

            provenance = RunProvenance(source="madis", command="test")
            stats = build_station_por("madis", norm_dir, out_dir, provenance)

            assert len(stats) == 2
            assert (out_dir / "madis_STN_A.parquet").exists()
            assert (out_dir / "madis_STN_B.parquet").exists()

    def test_output_has_qc_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            df = self._make_hourly_df("madis:STN_A")
            df.to_parquet(norm_dir / "day1.parquet", index=False)

            provenance = RunProvenance(source="madis", command="test")
            build_station_por("madis", norm_dir, out_dir, provenance)

            result = pd.read_parquet(out_dir / "madis_STN_A.parquet")
            assert "qc_state" in result.columns
            assert "qc_reason_codes" in result.columns

    def test_failed_hourly_rows_are_excluded_from_daily_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            dt = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
            df = pd.DataFrame(
                {
                    "station_key": "madis:STN_A",
                    "datetime_utc": dt,
                    "tair": np.arange(24, dtype=float),
                    "qc_state": ["pass"] * 12 + ["fail"] * 12,
                    "qc_reason_codes": [""] * 12 + ["out_of_bounds"] * 12,
                    "ingest_run_id": "test123",
                    "transform_version": "0.1.0",
                }
            )
            df.to_parquet(norm_dir / "day1.parquet", index=False)

            provenance = RunProvenance(source="madis", command="test")
            build_station_por("madis", norm_dir, out_dir, provenance)

            result = pd.read_parquet(out_dir / "madis_STN_A.parquet")
            assert len(result) == 1
            assert result.loc[0, "obs_count"] == 12
            assert result.loc[0, "tair"] == pytest.approx(np.mean(np.arange(12, dtype=float)))

    def test_outlier_gets_tier2_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            # Need ~4 years for >=90 obs per calendar month (z-score threshold)
            df = self._make_hourly_df("madis:STN_A", n_hours=365 * 4 * 24)
            # Inject extreme outlier on one day (tair is skipped; use td)
            df.loc[df.index[:24], "td"] = 60.0
            df.to_parquet(norm_dir / "day1.parquet", index=False)

            provenance = RunProvenance(source="madis", command="test")
            build_station_por("madis", norm_dir, out_dir, provenance)

            result = pd.read_parquet(out_dir / "madis_STN_A.parquet")
            # The extreme day should have a non-empty reason code
            first_day = result.iloc[0]
            assert (
                first_day["qc_state"] in ("suspect", "fail") or first_day["qc_reason_codes"] != ""
            )

    def test_station_failure_report_written_and_build_continues(self, monkeypatch):
        """A failing station should be reported without aborting other station writes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por" / "madis"

            df_good = self._make_hourly_df("madis:GOOD")
            df_bad = self._make_hourly_df("madis:BAD")
            pd.concat([df_good, df_bad], ignore_index=True).to_parquet(
                norm_dir / "day1.parquet",
                index=False,
            )

            original_apply = __import__(
                "obsmet.products.station_por",
                fromlist=["_apply_tier2_qc"],
            )._apply_tier2_qc

            def fail_one_station(station_df, variable_columns, *, rso=None):
                station_key = str(station_df["station_key"].iloc[0])
                if station_key == "madis:BAD":
                    raise RuntimeError("synthetic tier2 failure")
                return original_apply(station_df, variable_columns, rso=rso)

            monkeypatch.setattr(
                "obsmet.products.station_por._apply_tier2_qc",
                fail_one_station,
            )

            provenance = RunProvenance(source="madis", command="test")
            stats = build_station_por("madis", norm_dir, out_dir, provenance)

            assert stats == {"madis:GOOD": 3}
            assert (out_dir / "madis_GOOD.parquet").exists()
            assert not (out_dir / "madis_BAD.parquet").exists()

            report_path = out_dir.parent / "station_por_failures_madis.json"
            assert report_path.exists()

            report = json.loads(report_path.read_text())
            assert report["source"] == "madis"
            assert report["run_id"] == provenance.run_id
            assert report["failure_count"] == 1
            assert len(report["failures"]) == 1
            assert report["failures"][0]["phase"] == "pass2_station"
            assert report["failures"][0]["station_key"] == "madis:BAD"
            assert report["failures"][0]["error_type"] == "RuntimeError"
            assert report["qc_skip_count"] == 1
            assert report["qc_skips"][0]["station_key"] == "madis:GOOD"

    def test_pass1_file_failure_written_and_build_continues(self):
        """Unreadable normalized parquet files should be reported without aborting valid files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por" / "madis"

            df = self._make_hourly_df("madis:GOOD")
            df.to_parquet(norm_dir / "day1.parquet", index=False)
            (norm_dir / "broken.parquet").write_text("not a parquet file", encoding="utf-8")

            provenance = RunProvenance(source="madis", command="test")
            stats = build_station_por("madis", norm_dir, out_dir, provenance)

            assert stats == {"madis:GOOD": 3}
            assert (out_dir / "madis_GOOD.parquet").exists()

            report_path = out_dir.parent / "station_por_failures_madis.json"
            report = json.loads(report_path.read_text())
            assert report["failure_count"] == 1
            assert report["failures"][0]["phase"] == "pass1_read_daily"
            assert report["failures"][0]["input_file"].endswith("broken.parquet")
            assert report["qc_skip_count"] == 1

    def test_rs_qc_skip_reason_written_when_rso_unavailable(self):
        """Stations with Rs data but no usable Rso should be reported as QC skips."""
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por" / "madis"

            df = self._make_hourly_df("madis:RS_SKIP")
            df.to_parquet(norm_dir / "day1.parquet", index=False)

            provenance = RunProvenance(source="madis", command="test")
            stats = build_station_por("madis", norm_dir, out_dir, provenance)

            assert stats == {"madis:RS_SKIP": 3}
            assert (out_dir / "madis_RS_SKIP.parquet").exists()

            report_path = out_dir.parent / "station_por_failures_madis.json"
            report = json.loads(report_path.read_text())
            assert report["failure_count"] == 0
            assert report["qc_skip_count"] == 1
            assert report["qc_skips"][0]["station_key"] == "madis:RS_SKIP"
            assert report["qc_skips"][0]["qc_name"] == "rs_period_ratio"
            assert report["qc_skips"][0]["reason"] == "missing_lat_or_elev"


class TestTier2QCImprovements:
    """Tests for Tier 2 QC alignment with CONUS-AgWeather methodology."""

    def _make_daily_df(self, n_days=365 * 4, obs_count=24):
        """Build synthetic daily data suitable for Tier 2 QC."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        return pd.DataFrame(
            {
                "station_key": "madis:TEST",
                "date": dates,
                "day_basis": "utc",
                "obs_count": obs_count,
                "coverage_flags": f"n={obs_count},thresh=Y,am=Y,pm=Y",
                "qc_state": "pass",
                "qc_reason_codes": "",
                "qc_rules_version": "0.1.0",
                "transform_version": "0.1.0",
                "ingest_run_id": "test",
                "tair": rng.normal(10, 5, n_days),
                "td": rng.normal(3, 3, n_days),
                "rh": rng.uniform(40, 95, n_days),
                "rhmax": rng.uniform(70, 100, n_days),
                "rhmin": rng.uniform(20, 60, n_days),
                "wind": rng.uniform(0, 8, n_days),
                "wind_dir": rng.uniform(0, 360, n_days),
                "prcp": np.where(rng.random(n_days) > 0.7, rng.exponential(5, n_days), 0.0),
                "rsds": rng.uniform(5, 30, n_days),
                "tmax": rng.normal(15, 5, n_days),
                "tmin": rng.normal(5, 5, n_days),
                "tmean": rng.normal(10, 5, n_days),
            }
        )

    def test_prcp_not_flagged_by_zscore(self):
        """Precip should not be z-scored; legitimate heavy events should pass."""
        df = self._make_daily_df()
        # Inject a legitimate heavy precip day (well under 610mm)
        df.loc[100, "prcp"] = 80.0
        var_cols = ["tair", "td", "rh", "wind", "prcp", "rsds"]
        result = _apply_tier2_qc(df, var_cols)
        row = result.iloc[100]
        assert "zscore_prcp" not in row["qc_reason_codes"]

    def test_wind_dir_not_flagged_by_zscore(self):
        """Wind direction is circular and should not be z-scored."""
        df = self._make_daily_df()
        df.loc[100, "wind_dir"] = 359.0
        df.loc[101, "wind_dir"] = 1.0
        result = _apply_tier2_qc(df, ["wind_dir"])
        assert "zscore_wind_dir" not in ",".join(result["qc_reason_codes"].tolist())

    def test_prcp_exceeds_daily_max(self):
        """Daily precip > 610mm should be flagged."""
        df = self._make_daily_df(n_days=100)
        df.loc[50, "prcp"] = 700.0
        result = _apply_tier2_qc(df, ["prcp"])
        row = result.iloc[50]
        assert "prcp_exceeds_daily_max" in row["qc_reason_codes"]
        assert row["qc_state"] == "fail"

    def test_rhmax_rhmin_in_output(self):
        """station_por should produce rhmax/rhmin columns from hourly data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            rng = np.random.default_rng(42)
            n = 72
            dt = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
            df = pd.DataFrame(
                {
                    "station_key": "madis:STN_A",
                    "datetime_utc": dt,
                    "rh": rng.uniform(30, 90, n),
                    "tair": rng.normal(10, 3, n),
                    "qc_state": "pass",
                    "qc_reason_codes": "",
                    "ingest_run_id": "test",
                    "transform_version": "0.1.0",
                }
            )
            df.to_parquet(norm_dir / "day1.parquet", index=False)

            provenance = RunProvenance(source="madis", command="test")
            build_station_por("madis", norm_dir, out_dir, provenance)

            result = pd.read_parquet(out_dir / "madis_STN_A.parquet")
            assert "rhmax" in result.columns
            assert "rhmin" in result.columns
            # rhmax should be >= rhmin for every row
            valid = result["rhmax"].notna() & result["rhmin"].notna()
            assert (result.loc[valid, "rhmax"] >= result.loc[valid, "rhmin"]).all()

    def test_corrected_columns_present(self):
        """RH corrected columns should be present after Tier 2 QC."""
        df = self._make_daily_df()
        result = _apply_tier2_qc(df, ["rh"])
        assert "rh_corrected" in result.columns
        assert "rhmax_corrected" in result.columns
        assert "rhmin_corrected" in result.columns

    def test_rso_asce_fallback_produces_rsds_corrected(self):
        """When lat/elev_m are in hourly data, ASCE Rso enables Rs correction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            rng = np.random.default_rng(42)
            n = 365 * 24  # 1 year hourly
            dt = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
            # Seasonal Rs pattern with noise
            doy = dt.dayofyear
            rs_base = np.sin(doy / 365 * np.pi) * 400 + 100
            df = pd.DataFrame(
                {
                    "station_key": "madis:RSO_TEST",
                    "datetime_utc": dt,
                    "tair": rng.normal(10, 5, n),
                    "rsds_hourly": rs_base + rng.normal(0, 20, n),
                    "lat": 45.0,
                    "lon": -117.0,
                    "elev_m": 800.0,
                    "qc_state": "pass",
                    "qc_reason_codes": "",
                    "ingest_run_id": "test",
                    "transform_version": "0.1.0",
                }
            )
            df.to_parquet(norm_dir / "day1.parquet", index=False)

            provenance = RunProvenance(source="madis", command="test")
            build_station_por("madis", norm_dir, out_dir, provenance)

            result = pd.read_parquet(out_dir / "madis_RSO_TEST.parquet")
            assert "rsds_corrected" in result.columns
            assert "lat" in result.columns
            assert "elev_m" in result.columns

    def test_min_por_filter(self):
        """Station with insufficient coverage should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            rng = np.random.default_rng(42)
            # Only 48 hours → 2 days, both with obs_count=24
            n = 48
            dt = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
            df = pd.DataFrame(
                {
                    "station_key": "madis:SHORT",
                    "datetime_utc": dt,
                    "tair": rng.normal(10, 3, n),
                    "qc_state": "pass",
                    "qc_reason_codes": "",
                    "ingest_run_id": "test",
                    "transform_version": "0.1.0",
                }
            )
            df.to_parquet(norm_dir / "day1.parquet", index=False)

            provenance = RunProvenance(source="madis", command="test")
            # Require 30 sufficient-coverage days — station only has 2
            stats = build_station_por("madis", norm_dir, out_dir, provenance, min_por_days=30)
            assert len(stats) == 0
            assert not (out_dir / "madis_SHORT.parquet").exists()
            report_path = out_dir.parent / "station_por_failures_madis.json"
            report = json.loads(report_path.read_text())
            assert report["qc_skip_count"] == 1
            assert report["qc_skips"][0]["qc_name"] == "min_por_days"
            assert report["qc_skips"][0]["reason"] == "insufficient_por_days"


class TestGdasStationPor:
    def test_gdas_station_por_targets_daily_tmean(self):
        assert _station_por_variable_columns(
            "gdas", ["tair", "td", "wind", "wind_dir", "psfc"]
        ) == [
            "tmean",
            "td",
            "psfc",
        ]

    def test_gdas_stitches_hourly_rows_across_file_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            common = {
                "station_key": "gdas:STN_A",
                "source": "gdas_adpsfc",
                "source_station_id": "STN_A",
                "lat": 45.0,
                "lon": -110.0,
                "elev_m": 1000.0,
                "msg_type": "ADPSFC",
                "q": 0.010,
                "u": -3.0,
                "v": -4.0,
                "psfc": 100000.0,
                "tair": 20.0,
                "qc_state": "pass",
                "qc_reason_codes": "",
                "ingest_run_id": "test",
                "transform_version": "0.1.0",
            }

            file1 = pd.DataFrame(
                [
                    {
                        **common,
                        "datetime_utc": pd.Timestamp("2024-01-02 00:00:00", tz="UTC"),
                        "cycle": 0,
                    },
                    {
                        **common,
                        "datetime_utc": pd.Timestamp("2024-01-02 06:00:00", tz="UTC"),
                        "cycle": 6,
                    },
                    {
                        **common,
                        "datetime_utc": pd.Timestamp("2024-01-02 12:00:00", tz="UTC"),
                        "cycle": 12,
                    },
                    {
                        **common,
                        "datetime_utc": pd.Timestamp("2024-01-02 18:00:00", tz="UTC"),
                        "cycle": 18,
                    },
                ]
            )

            file2 = pd.DataFrame(
                [
                    {
                        **common,
                        "datetime_utc": pd.Timestamp("2024-01-02 18:00:00", tz="UTC"),
                        "cycle": 0,
                    },
                    {
                        **common,
                        "datetime_utc": pd.Timestamp("2024-01-02 21:00:00", tz="UTC"),
                        "cycle": 0,
                    },
                    {
                        **common,
                        "datetime_utc": pd.Timestamp("2024-01-02 23:00:00", tz="UTC"),
                        "cycle": 0,
                    },
                    {
                        **common,
                        "datetime_utc": pd.Timestamp("2024-01-03 00:00:00", tz="UTC"),
                        "cycle": 0,
                    },
                ]
            )

            file1.to_parquet(norm_dir / "20240102.parquet", index=False)
            file2.to_parquet(norm_dir / "20240103.parquet", index=False)

            provenance = RunProvenance(source="gdas", command="test")
            stats = build_station_por("gdas", norm_dir, out_dir, provenance)

            assert stats == {"gdas:STN_A": 2}

            result = (
                pd.read_parquet(out_dir / "gdas_STN_A.parquet")
                .sort_values("date")
                .reset_index(drop=True)
            )
            assert list(result["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
            assert result.loc[0, "obs_count"] == 6
            assert "n=6" in result.loc[0, "coverage_flags"]
            assert result.loc[0, "wind"] == pytest.approx(5.0)
            assert pd.notna(result.loc[0, "td"])


class TestDailyNativeStationPor:
    """Tests for RAWS, SNOTEL, and NDBC station POR builds."""

    def _make_raws_daily_df(self, station_id="azTEST", n_days=100):
        """Synthetic RAWS daily data matching normalized schema."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "station_key": f"raws:{station_id}",
                "source": "raws_wrcc",
                "source_station_id": station_id,
                "lat": [None] * n_days,
                "lon": [None] * n_days,
                "elev_m": [None] * n_days,
                "tmean": rng.normal(10, 5, n_days),
                "tmax": rng.normal(15, 5, n_days),
                "tmin": rng.normal(5, 5, n_days),
                "wind": rng.uniform(0, 8, n_days),
                "wind_dir": rng.uniform(0, 360, n_days),
                "rh": rng.uniform(30, 80, n_days),
                "rh_max": rng.uniform(60, 100, n_days),
                "rh_min": rng.uniform(10, 50, n_days),
                "prcp": np.where(rng.random(n_days) > 0.8, rng.exponential(3, n_days), 0.0),
                "rsds": rng.uniform(0.5, 9.0, n_days),  # kWh/m²
                "qc_state": "pass",
                "qc_reason_codes": "",
                "ingest_run_id": "test",
                "transform_version": "0.1.0",
            }
        )

    def _make_snotel_daily_df(self, n_days=100):
        """Synthetic SNOTEL daily data matching normalized schema (NaN station_key, string date)."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        return pd.DataFrame(
            {
                "station_key": [np.nan] * n_days,
                "source": [np.nan] * n_days,
                "source_station_id": [np.nan] * n_days,
                "date": dates.strftime("%Y-%m-%d %H:%M:%S"),  # string, like real data
                "day_basis": "local",
                "swe": rng.uniform(0, 200, n_days),
                "tmin": rng.normal(-5, 5, n_days),
                "tmax": rng.normal(5, 5, n_days),
                "tmean": rng.normal(0, 5, n_days),
                "prcp": np.where(rng.random(n_days) > 0.7, rng.exponential(4, n_days), 0.0),
                "rh": rng.uniform(40, 95, n_days),
                "wind": rng.uniform(0, 10, n_days),
                "qc_state": "pass",
                "obs_count": 1,
                "ingest_run_id": "test",
                "transform_version": "0.1.0",
                "raw_source_uri": "/nas/climate/snotel/1002_Kraft_Creek_MT.csv",
            }
        )

    def test_raws_passthrough_renames_rh_and_converts_rsds(self):
        """RAWS passthrough should rename rh_max→rhmax and convert rsds kWh→MJ."""
        df = self._make_raws_daily_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = Path(tmpdir) / "azTEST.parquet"
            df.to_parquet(pf, index=False)

            result = _passthrough_daily_file((str(pf), "raws", None, None))
            daily = result["daily"]

            assert daily is not None
            assert "rhmax" in daily.columns
            assert "rhmin" in daily.columns
            assert "rh_max" not in daily.columns
            assert "rh_min" not in daily.columns
            # rsds should be converted: kWh * 3.6 = MJ
            orig_rsds = df["rsds"].iloc[0]
            assert daily["rsds"].iloc[0] == pytest.approx(orig_rsds * 3.6)

    def test_snotel_passthrough_derives_station_key_from_filename(self):
        """SNOTEL passthrough should derive station_key from filename when NaN."""
        df = self._make_snotel_daily_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = Path(tmpdir) / "1002_Kraft_Creek_MT.parquet"
            df.to_parquet(pf, index=False)

            result = _passthrough_daily_file((str(pf), "snotel", None, None))
            daily = result["daily"]

            assert daily is not None
            assert (daily["station_key"] == "snotel:1002").all()
            assert (daily["source"] == "snotel").all()
            assert (daily["source_station_id"] == "1002").all()
            # date should be coerced from string to datetime
            assert pd.api.types.is_datetime64_any_dtype(daily["date"])

    def test_raws_build_station_por_end_to_end(self):
        """Full build_station_por for RAWS produces correct output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            df = self._make_raws_daily_df("azTEST", n_days=200)
            df.to_parquet(norm_dir / "azTEST.parquet", index=False)

            provenance = RunProvenance(source="raws", command="test")
            stats = build_station_por("raws", norm_dir, out_dir, provenance)

            assert "raws:azTEST" in stats
            out_path = out_dir / "raws_azTEST.parquet"
            assert out_path.exists()

            result = pd.read_parquet(out_path)
            assert "qc_state" in result.columns
            assert "qc_reason_codes" in result.columns
            assert "rhmax" in result.columns
            assert "rhmin" in result.columns
            # rsds should be in MJ/m²/day range (original 0.5-9 kWh * 3.6 = 1.8-32.4)
            rsds = result["rsds"].dropna()
            assert rsds.min() > 1.0
            assert rsds.max() < 35.0
            # RH drift correction should produce corrected columns
            assert "rh_corrected" in result.columns

    def test_snotel_build_station_por_end_to_end(self):
        """Full build_station_por for SNOTEL with NaN station_key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            df = self._make_snotel_daily_df(n_days=200)
            df.to_parquet(norm_dir / "1002_Kraft_Creek_MT.parquet", index=False)

            provenance = RunProvenance(source="snotel", command="test")
            stats = build_station_por("snotel", norm_dir, out_dir, provenance)

            assert "snotel:1002" in stats
            out_path = out_dir / "snotel_1002.parquet"
            assert out_path.exists()

            result = pd.read_parquet(out_path)
            assert (result["station_key"] == "snotel:1002").all()
            assert "qc_state" in result.columns
            assert "swe" in result.columns
            assert "prcp" in result.columns

    def test_ndbc_build_station_por_end_to_end(self):
        """NDBC (hourly) should work through the standard aggregation path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            rng = np.random.default_rng(42)
            n = 72
            dt = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
            df = pd.DataFrame(
                {
                    "station_key": "ndbc:41001",
                    "source": "ndbc",
                    "source_station_id": "41001",
                    "datetime_utc": dt,
                    "tair": rng.normal(20, 3, n),
                    "td": rng.normal(15, 2, n),
                    "wind": rng.uniform(1, 15, n),
                    "wind_dir": rng.uniform(0, 360, n),
                    "slp": rng.normal(101325, 500, n),
                    "qc_state": "pass",
                    "qc_reason_codes": "",
                    "ingest_run_id": "test",
                    "transform_version": "0.1.0",
                }
            )
            df.to_parquet(norm_dir / "41001.parquet", index=False)

            provenance = RunProvenance(source="ndbc", command="test")
            stats = build_station_por("ndbc", norm_dir, out_dir, provenance)

            assert "ndbc:41001" in stats
            out_path = out_dir / "ndbc_41001.parquet"
            assert out_path.exists()

            result = pd.read_parquet(out_path)
            assert "tmax" in result.columns  # derived from tair
            assert "tmin" in result.columns
            assert "slp" in result.columns
            assert "qc_state" in result.columns

    def test_daily_passthrough_date_range_filter(self):
        """Date range filtering should work for daily-native sources."""
        from datetime import date

        df = self._make_raws_daily_df(n_days=365)
        with tempfile.TemporaryDirectory() as tmpdir:
            pf = Path(tmpdir) / "azTEST.parquet"
            df.to_parquet(pf, index=False)

            start = date(2020, 3, 1)
            end = date(2020, 6, 30)
            result = _passthrough_daily_file((str(pf), "raws", start, end))
            daily = result["daily"]

            assert daily is not None
            dates = pd.to_datetime(daily["date"])
            assert dates.min().date() >= start
            assert dates.max().date() <= end
