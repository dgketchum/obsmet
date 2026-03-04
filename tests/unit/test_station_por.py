"""Tests for obsmet.products.station_por."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.products.station_por import build_station_por


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
            assert (out_dir / "madis:STN_A.parquet").exists()
            assert (out_dir / "madis:STN_B.parquet").exists()

    def test_output_has_qc_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            df = self._make_hourly_df("madis:STN_A")
            df.to_parquet(norm_dir / "day1.parquet", index=False)

            provenance = RunProvenance(source="madis", command="test")
            build_station_por("madis", norm_dir, out_dir, provenance)

            result = pd.read_parquet(out_dir / "madis:STN_A.parquet")
            assert "qc_state" in result.columns
            assert "qc_reason_codes" in result.columns

    def test_outlier_gets_tier2_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_dir = Path(tmpdir) / "normalized"
            norm_dir.mkdir()
            out_dir = Path(tmpdir) / "station_por"

            # 200 hours gives enough daily records for z-score
            df = self._make_hourly_df("madis:STN_A", n_hours=200 * 24)
            # Inject extreme outlier on one day
            df.loc[df.index[:24], "tair"] = 60.0
            df.to_parquet(norm_dir / "day1.parquet", index=False)

            provenance = RunProvenance(source="madis", command="test")
            build_station_por("madis", norm_dir, out_dir, provenance)

            result = pd.read_parquet(out_dir / "madis:STN_A.parquet")
            # The extreme day should have a non-empty reason code
            first_day = result.iloc[0]
            assert (
                first_day["qc_state"] in ("suspect", "fail") or first_day["qc_reason_codes"] != ""
            )
