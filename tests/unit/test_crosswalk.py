"""Tests for crosswalk index, builder, precedence, and fabric."""

import numpy as np
import pandas as pd

from obsmet.crosswalk.matchers import MatchConfidence, haversine_distance
from obsmet.crosswalk.precedence import DEFAULT_PRECEDENCE, load_precedence


class TestHaversineDistance:
    def test_same_point(self):
        assert haversine_distance(46.87, -114.0, 46.87, -114.0) == 0.0

    def test_known_distance(self):
        # Missoula to Helena ~160 km
        d = haversine_distance(46.87, -114.0, 46.60, -112.03)
        assert 150_000 < d < 170_000

    def test_close_points(self):
        # Two points ~100m apart
        d = haversine_distance(46.87, -114.0, 46.8709, -114.0)
        assert d < 200


class TestPrecedence:
    def test_default_has_expected_vars(self):
        assert "tair" in DEFAULT_PRECEDENCE.hourly
        assert "tmax" in DEFAULT_PRECEDENCE.daily
        assert "prcp" in DEFAULT_PRECEDENCE.daily
        assert "swe" in DEFAULT_PRECEDENCE.daily

    def test_default_daily_tmax_order(self):
        order = DEFAULT_PRECEDENCE.daily["tmax"]
        assert order[0] == "ghcnd"

    def test_default_swe_snotel_only(self):
        assert DEFAULT_PRECEDENCE.daily["swe"] == ["snotel"]

    def test_load_precedence_none_returns_default(self):
        p = load_precedence(None)
        assert p is DEFAULT_PRECEDENCE


class TestStationIndex:
    def test_index_per_station(self, tmp_path):
        """Index a directory of per-station parquet files."""
        from obsmet.crosswalk.station_index import _index_per_station_source

        # Create two fake station parquets
        norm_dir = tmp_path / "ghcnd"
        norm_dir.mkdir()

        for sid in ["USW00024153", "USC00244558"]:
            df = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=10),
                    "station_key": f"ghcnd:{sid}",
                    "source": "ghcnd",
                    "source_station_id": sid,
                    "lat": 46.87,
                    "lon": -114.0,
                    "elev_m": 972.0,
                    "tmax": np.random.uniform(20, 35, 10),
                }
            )
            df.to_parquet(norm_dir / f"{sid}.parquet", index=False)

        records = _index_per_station_source("ghcnd", norm_dir, "date")
        assert len(records) == 2
        assert records[0]["source"] == "ghcnd"
        assert records[0]["obs_count"] == 10

    def test_index_nan_station_key_fallback(self, tmp_path):
        """When station_key is NaN, derive from filename."""
        from obsmet.crosswalk.station_index import _index_per_station_source

        norm_dir = tmp_path / "ghcnd"
        norm_dir.mkdir()

        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "station_key": [np.nan] * 5,
                "source": [np.nan] * 5,
                "source_station_id": [np.nan] * 5,
                "lat": 46.87,
                "lon": -114.0,
                "tmax": [25.0] * 5,
            }
        )
        df.to_parquet(norm_dir / "USW00024153.parquet", index=False)

        records = _index_per_station_source("ghcnd", norm_dir, "date")
        assert len(records) == 1
        assert records[0]["canonical_id"] == "ghcnd:USW00024153"
        assert records[0]["source_station_id"] == "USW00024153"


class TestCrosswalkBuilder:
    def test_exact_ghcn_match(self, tmp_path):
        """GHCNh and GHCN-Daily stations with same ID get exact match."""
        from obsmet.crosswalk.builder import build_crosswalk

        idx = pd.DataFrame(
            [
                {
                    "canonical_id": "ghcnh:USW00024153",
                    "source": "ghcnh",
                    "source_station_id": "USW00024153",
                    "lat": 46.87,
                    "lon": -114.0,
                    "elev_m": 972.0,
                },
                {
                    "canonical_id": "ghcnd:USW00024153",
                    "source": "ghcnd",
                    "source_station_id": "USW00024153",
                    "lat": 46.87,
                    "lon": -114.0,
                    "elev_m": 972.0,
                },
            ]
        )
        idx_path = tmp_path / "station_index.parquet"
        idx.to_parquet(idx_path, index=False)

        result = build_crosswalk(idx_path)
        assert len(result) == 2
        assert result["canonical_station_id"].nunique() == 1
        assert result["canonical_station_id"].iloc[0] == "ghcn:USW00024153"
        assert all(result["confidence"] == MatchConfidence.EXACT.value)

    def test_geospatial_match(self, tmp_path):
        """Non-GHCN station near a GHCN station gets probable match."""
        from obsmet.crosswalk.builder import build_crosswalk

        idx = pd.DataFrame(
            [
                {
                    "canonical_id": "ghcnh:USW00024153",
                    "source": "ghcnh",
                    "source_station_id": "USW00024153",
                    "lat": 46.87,
                    "lon": -114.0,
                    "elev_m": 972.0,
                },
                {
                    "canonical_id": "madis:KMSO",
                    "source": "madis",
                    "source_station_id": "KMSO",
                    "lat": 46.8705,
                    "lon": -114.0005,
                    "elev_m": 972.0,
                },
            ]
        )
        idx_path = tmp_path / "station_index.parquet"
        idx.to_parquet(idx_path, index=False)

        result = build_crosswalk(idx_path)
        assert len(result) == 2
        assert result["canonical_station_id"].nunique() == 1

        madis_row = result[result["source"] == "madis"].iloc[0]
        assert madis_row["confidence"] == MatchConfidence.PROBABLE.value
        assert madis_row["distance_m"] < 1000

    def test_no_match_far_apart(self, tmp_path):
        """Stations >1km apart get no match."""
        from obsmet.crosswalk.builder import build_crosswalk

        idx = pd.DataFrame(
            [
                {
                    "canonical_id": "ghcnh:USW00024153",
                    "source": "ghcnh",
                    "source_station_id": "USW00024153",
                    "lat": 46.87,
                    "lon": -114.0,
                    "elev_m": 972.0,
                },
                {
                    "canonical_id": "madis:KHLN",
                    "source": "madis",
                    "source_station_id": "KHLN",
                    "lat": 46.60,
                    "lon": -112.03,
                    "elev_m": 1188.0,
                },
            ]
        )
        idx_path = tmp_path / "station_index.parquet"
        idx.to_parquet(idx_path, index=False)

        result = build_crosswalk(idx_path)
        assert result["canonical_station_id"].nunique() == 2

        madis_row = result[result["source"] == "madis"].iloc[0]
        assert madis_row["confidence"] == MatchConfidence.NONE.value


class TestFabric:
    def test_daily_precedence_single_source(self, tmp_path):
        """Fabric with one source passes data through."""
        from obsmet.products.fabric import _apply_precedence_daily

        src_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "tmax": [25.0, 26.0, 27.0, 28.0, 29.0],
                "qc_state": "pass",
            }
        )
        prec = {"tmax": ["ghcnd"]}
        result = _apply_precedence_daily({"ghcnd": src_df}, prec)
        assert len(result) == 5
        assert result["tmax"].tolist() == [25.0, 26.0, 27.0, 28.0, 29.0]
        assert all(result["tmax_source"] == "ghcnd")

    def test_daily_precedence_fallback(self, tmp_path):
        """Higher-priority source with NaN falls back to lower-priority."""
        from obsmet.products.fabric import _apply_precedence_daily

        ghcnd_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3),
                "tmax": [25.0, np.nan, 27.0],
                "qc_state": "pass",
            }
        )
        snotel_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3),
                "tmax": [24.0, 25.5, 26.0],
                "qc_state": "pass",
            }
        )
        prec = {"tmax": ["ghcnd", "snotel"]}
        result = _apply_precedence_daily({"ghcnd": ghcnd_df, "snotel": snotel_df}, prec)

        # Day 1: ghcnd wins (25.0), Day 2: snotel fills (25.5), Day 3: ghcnd wins (27.0)
        assert result["tmax"].tolist() == [25.0, 25.5, 27.0]
        assert result["tmax_source"].tolist() == ["ghcnd", "snotel", "ghcnd"]

    def test_daily_precedence_qc_fallback(self, tmp_path):
        """Source with QC=fail falls back to next source."""
        from obsmet.products.fabric import _apply_precedence_daily

        ghcnd_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=2),
                "tmax": [25.0, 99.0],
                "qc_state": ["pass", "fail"],
            }
        )
        snotel_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=2),
                "tmax": [24.0, 26.0],
                "qc_state": "pass",
            }
        )
        prec = {"tmax": ["ghcnd", "snotel"]}
        result = _apply_precedence_daily({"ghcnd": ghcnd_df, "snotel": snotel_df}, prec)

        # Day 1: ghcnd passes (25.0), Day 2: ghcnd fails → snotel (26.0)
        assert result["tmax"].tolist() == [25.0, 26.0]
        assert result["tmax_source"].tolist() == ["ghcnd", "snotel"]
