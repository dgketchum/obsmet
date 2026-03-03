"""Dedup and precedence auditing (plan section 14).

When multiple sources provide the same station-time-variable,
choose by precedence matrix and keep alternatives in an audit table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DedupRecord:
    """Record of a deduplication decision."""

    station_key: str
    datetime_utc: str
    variable: str
    kept_source: str
    dropped_source: str
    dedup_reason: str  # "lower_priority_source", "qc_failure", "time_conflict"
