"""Tests for source registry."""

import pytest

from obsmet.sources.registry import (
    SourceEntry,
    create_adapter,
    get_source,
    list_sources,
)


class TestListSources:
    def test_returns_all_five(self):
        sources = list_sources()
        assert set(sources) == {"madis", "isd", "ghcnh", "ghcnd", "gdas", "raws", "ndbc", "snotel"}

    def test_sorted(self):
        sources = list_sources()
        assert sources == sorted(sources)


class TestGetSource:
    def test_valid_source(self):
        entry = get_source("isd")
        assert isinstance(entry, SourceEntry)
        assert entry.name == "isd"
        assert entry.parallel is True

    def test_raws_not_parallel(self):
        entry = get_source("raws")
        assert entry.parallel is False

    def test_unknown_source_raises(self):
        with pytest.raises(KeyError, match="Unknown source"):
            get_source("nonexistent")


class TestCreateAdapter:
    def test_create_isd(self, tmp_path):
        adapter = create_adapter("isd", raw_dir=str(tmp_path))
        assert adapter.source_name == "isd"

    def test_create_madis(self, tmp_path):
        adapter = create_adapter("madis", raw_dir=str(tmp_path))
        assert adapter.source_name == "madis"

    def test_create_raws(self, tmp_path):
        adapter = create_adapter("raws", raw_dir=str(tmp_path))
        assert adapter.source_name == "raws_wrcc"

    def test_create_ndbc(self, tmp_path):
        adapter = create_adapter("ndbc", raw_dir=str(tmp_path))
        assert adapter.source_name == "ndbc"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            create_adapter("nonexistent")
