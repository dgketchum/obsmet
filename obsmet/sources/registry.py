"""Source registry — maps source names to adapter classes and config.

Simple dict-based registry with lazy adapter imports to avoid loading
heavy dependencies (eccodes, netCDF4) at CLI startup.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass
class SourceEntry:
    """Registry entry for a source connector."""

    name: str
    adapter_path: str
    parallel: bool


_SOURCES: dict[str, dict] = {
    "madis": {
        "adapter": "obsmet.sources.madis.adapter.MadisAdapter",
        "parallel": True,
    },
    "isd": {
        "adapter": "obsmet.sources.isd.adapter.IsdAdapter",
        "parallel": True,
    },
    "ghcnh": {
        "adapter": "obsmet.sources.ghcnh.adapter.GhcnhAdapter",
        "parallel": True,
    },
    "ghcnd": {
        "adapter": "obsmet.sources.ghcnd.adapter.GhcndAdapter",
        "parallel": True,
    },
    "gdas": {
        "adapter": "obsmet.sources.gdas_prepbufr.adapter.GdasAdapter",
        "parallel": True,
    },
    "raws": {
        "adapter": "obsmet.sources.raws_wrcc.adapter.RawsAdapter",
        "parallel": False,
    },
    "ndbc": {
        "adapter": "obsmet.sources.ndbc.adapter.NdbcAdapter",
        "parallel": True,
    },
    "snotel": {
        "adapter": "obsmet.sources.snotel.adapter.SnotelAdapter",
        "parallel": True,
    },
}


def list_sources() -> list[str]:
    """Return sorted list of registered source names."""
    return sorted(_SOURCES.keys())


def get_source(name: str) -> SourceEntry:
    """Look up a source by name. Raises KeyError if not found."""
    if name not in _SOURCES:
        raise KeyError(f"Unknown source {name!r}. Available: {list_sources()}")
    info = _SOURCES[name]
    return SourceEntry(
        name=name,
        adapter_path=info["adapter"],
        parallel=info["parallel"],
    )


def create_adapter(name: str, **kwargs):
    """Create an adapter instance by source name with lazy import.

    Keyword arguments are forwarded to the adapter constructor.
    """
    entry = get_source(name)
    module_path, class_name = entry.adapter_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)
