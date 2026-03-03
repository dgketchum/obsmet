"""Base interface for source adapters (plan section 12).

Each source connector implements this interface to handle raw data
acquisition and normalization to the canonical schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from obsmet.core.provenance import RunProvenance


class SourceAdapter(ABC):
    """Base class for all source connectors."""

    source_name: str  # e.g. "madis", "isd"

    @abstractmethod
    def discover_keys(self, start, end) -> list[str]:
        """List available raw data keys (files, days, etc.) in the date range."""
        ...

    @abstractmethod
    def fetch_raw(self, key: str, dest_dir: Path) -> Path:
        """Download/retrieve raw data for a key. Return path to raw file."""
        ...

    @abstractmethod
    def normalize(self, raw_path: Path, provenance: RunProvenance) -> pd.DataFrame:
        """Parse raw file into canonical hourly observation DataFrame.

        The returned DataFrame must conform to OBS_HOURLY_SCHEMA columns.
        """
        ...
