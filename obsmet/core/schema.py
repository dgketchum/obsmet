"""Canonical schema definitions for obsmet data products.

Defines the column contracts for station metadata, hourly observations,
and daily aggregated observations (plan sections 8.1-8.3).
"""

import pyarrow as pa

# --------------------------------------------------------------------------- #
# 8.1  Station metadata
# --------------------------------------------------------------------------- #

STATION_SCHEMA = pa.schema(
    [
        pa.field("station_key", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_station_id", pa.string(), nullable=False),
        pa.field("lat", pa.float64()),
        pa.field("lon", pa.float64()),
        pa.field("elev_m", pa.float64()),
        pa.field("country", pa.string()),
        pa.field("state", pa.string()),
        pa.field("network", pa.string()),
        pa.field("provider", pa.string()),
        pa.field("por_start", pa.date32()),
        pa.field("por_end", pa.date32()),
        pa.field("active_flag", pa.bool_()),
        pa.field("metadata_version", pa.string()),
    ]
)

# --------------------------------------------------------------------------- #
# 8.2  Hourly observations
# --------------------------------------------------------------------------- #

OBS_HOURLY_SCHEMA = pa.schema(
    [
        pa.field("station_key", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_station_id", pa.string(), nullable=False),
        pa.field("datetime_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("variable", pa.string(), nullable=False),
        pa.field("value", pa.float64()),
        pa.field("unit", pa.string(), nullable=False),
        pa.field("qc_state", pa.string()),
        pa.field("qc_flags_native", pa.string()),
        pa.field("qc_rules_version", pa.string()),
        pa.field("transform_version", pa.string()),
        pa.field("raw_source_uri", pa.string()),
        pa.field("raw_file_hash", pa.string()),
        pa.field("ingest_run_id", pa.string()),
    ]
)

# --------------------------------------------------------------------------- #
# 8.3  Daily observations
# --------------------------------------------------------------------------- #

# Core columns present in every daily record; variable-specific metric
# columns (tmax, tmin, etc.) are appended by the daily aggregation engine.
OBS_DAILY_CORE_SCHEMA = pa.schema(
    [
        pa.field("station_key", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("day_basis", pa.string(), nullable=False),
        pa.field("obs_count", pa.int32()),
        pa.field("coverage_flags", pa.string()),
        pa.field("qc_state", pa.string()),
        pa.field("qc_rules_version", pa.string()),
        pa.field("transform_version", pa.string()),
        pa.field("ingest_run_id", pa.string()),
    ]
)

# Canonical daily metric variable names (appended to the core schema).
DAILY_METRIC_FIELDS = [
    pa.field("tmax", pa.float64()),
    pa.field("tmin", pa.float64()),
    pa.field("tmean", pa.float64()),
    pa.field("ea", pa.float64()),
    pa.field("vpd", pa.float64()),
    pa.field("u2", pa.float64()),
    pa.field("rsds", pa.float64()),
    pa.field("prcp", pa.float64()),
]

# --------------------------------------------------------------------------- #
# Manifest entry schema (shared across all sources)
# --------------------------------------------------------------------------- #

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string(), nullable=False),
        pa.field("key", pa.string(), nullable=False),
        pa.field("state", pa.string(), nullable=False),
        pa.field("updated_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("run_id", pa.string()),
        pa.field("message", pa.string()),
    ]
)

# Valid manifest states
MANIFEST_STATES = frozenset({"done", "missing", "suspect", "failed", "skipped"})

# Valid QC states
QC_STATES = frozenset({"pass", "fail", "suspect", "missing"})

# --------------------------------------------------------------------------- #
# Source identifiers (plan section 12)
# --------------------------------------------------------------------------- #

SOURCES = frozenset(
    {
        "madis",
        "isd",
        "gdas_adpsfc",
        "gdas_sfcshp",
        "raws_wrcc",
        "ndbc",
        "ghcn",
        "synoptic",
    }
)
