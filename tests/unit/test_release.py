"""Tests for obsmet.products.release."""

import json
import tempfile
from pathlib import Path

import pandas as pd

from obsmet.core.provenance import RunProvenance
from obsmet.products.release import build_release, promote_release, validate_release


def _make_station_por(tmp: Path, source: str, station_keys: list[str]):
    """Create fake station POR parquets."""
    por_dir = tmp / "station_por" / source
    por_dir.mkdir(parents=True)
    for sk in station_keys:
        df = pd.DataFrame(
            {
                "station_key": [sk],
                "date": ["2024-01-01"],
                "tair": [20.0],
                "qc_state": ["pass"],
            }
        )
        df.to_parquet(por_dir / f"{sk}.parquet", index=False)
    return tmp / "station_por"


class TestBuildRelease:
    def test_creates_metadata_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            por_root = _make_station_por(tmp, "madis", ["stn_a", "stn_b"])
            releases = tmp / "releases"
            channels = tmp / "channels"

            provenance = RunProvenance(source="madis", command="test")
            release_dir = build_release(
                "v0.0.1",
                "candidate",
                ["madis"],
                provenance,
                station_por_root=por_root,
                releases_root=releases,
                channels_root=channels,
            )

            assert (release_dir / "release_metadata.json").exists()
            assert (release_dir / "manifest.parquet").exists()

            with open(release_dir / "release_metadata.json") as f:
                meta = json.load(f)
            assert meta["version"] == "v0.0.1"
            assert meta["station_count"] == 2

            # Channel symlink
            assert (channels / "candidate").is_symlink()

    def test_hardlinks_parquets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            por_root = _make_station_por(tmp, "madis", ["stn_a"])
            releases = tmp / "releases"
            channels = tmp / "channels"

            provenance = RunProvenance(source="madis", command="test")
            release_dir = build_release(
                "v0.0.1",
                "candidate",
                ["madis"],
                provenance,
                station_por_root=por_root,
                releases_root=releases,
                channels_root=channels,
            )

            linked = release_dir / "station_por" / "madis" / "stn_a.parquet"
            assert linked.exists()


class TestValidateRelease:
    def test_valid_release_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            por_root = _make_station_por(tmp, "madis", ["stn_a"])
            releases = tmp / "releases"
            channels = tmp / "channels"

            provenance = RunProvenance(source="madis", command="test")
            build_release(
                "v0.0.1",
                "candidate",
                ["madis"],
                provenance,
                station_por_root=por_root,
                releases_root=releases,
                channels_root=channels,
            )

            ok, errors = validate_release("v0.0.1", releases_root=releases)
            assert ok
            assert errors == []

    def test_corrupted_file_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            por_root = _make_station_por(tmp, "madis", ["stn_a"])
            releases = tmp / "releases"
            channels = tmp / "channels"

            provenance = RunProvenance(source="madis", command="test")
            release_dir = build_release(
                "v0.0.1",
                "candidate",
                ["madis"],
                provenance,
                station_por_root=por_root,
                releases_root=releases,
                channels_root=channels,
            )

            # Corrupt the file
            corrupted = release_dir / "station_por" / "madis" / "stn_a.parquet"
            corrupted.write_bytes(b"corrupted data")

            ok, errors = validate_release("v0.0.1", releases_root=releases)
            assert not ok
            assert len(errors) == 1
            assert "checksum mismatch" in errors[0]


class TestPromoteRelease:
    def test_promote_creates_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            por_root = _make_station_por(tmp, "madis", ["stn_a"])
            releases = tmp / "releases"
            channels = tmp / "channels"

            provenance = RunProvenance(source="madis", command="test")
            build_release(
                "v0.0.1",
                "candidate",
                ["madis"],
                provenance,
                station_por_root=por_root,
                releases_root=releases,
                channels_root=channels,
            )

            promote_release("v0.0.1", "prod", releases_root=releases, channels_root=channels)
            assert (channels / "prod").is_symlink()
