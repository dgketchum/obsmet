"""Tests for obsmet.products.station_por."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.products.station_por import _apply_tier2_qc, build_station_por


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

    def test_outlier_gets_tier2_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            # Need ~4 years for >=90 obs per calendar month (z-score threshold)
            df = self._make_hourly_df("madis:STN_A", n_hours=365 * 4 * 24)
            # Inject extreme outlier on one day
            df.loc[df.index[:24], "tair"] = 60.0
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
