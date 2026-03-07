# NDBC Data Source

## What NDBC Is

The **National Data Buoy Center (NDBC)** operates and maintains a network of automated
marine observing stations (buoys, coastal stations, and ships) across US coastal waters,
the Great Lakes, and open ocean. NDBC provides hourly standard meteorological (stdmet)
observations including air temperature, wind, pressure, and wave measurements.

## Archive Coverage

| Property | Value |
|----------|-------|
| **Period of record** | 1970s through present (varies by station) |
| **Temporal resolution** | Hourly |
| **File format** | Per-station parquet (pre-parsed from dads-mvp) |
| **File naming** | `{station_id}.parquet` (e.g., `41001.parquet`) |
| **Organization** | One file per station, all dates |
| **Station count** | ~1,249 stations |

## Access

The NDBC data used by obsmet consists of pre-parsed hourly parquet files derived from
NDBC's stdmet text files. The original text files use space-delimited format with station
ID, year-specific filenames (e.g., `41001h2024.txt.gz`).

## Variables Extracted

### Core Meteorological Variables

| NDBC Column | Canonical Name | Unit | Conversion |
|-------------|----------------|------|------------|
| `air_temp` | `tair` | degC | None |
| `dewpoint` | `td` | degC | None |
| `wind_speed` | `wind` | m s-1 | None |
| `wind_dir` | `wind_dir` | deg | None |
| `pressure` | `slp` | Pa | hPa x 100 |
| `visibility` | `vis` | nmi | None |

### Extension Variables (Buoy-Specific)

These variables are preserved in output but are not part of the core canonical schema:

| Column | Unit | Description |
|--------|------|-------------|
| `wave_height` | m | Significant wave height |
| `dominant_wave_period` | s | Dominant wave period |
| `average_wave_period` | s | Average wave period |
| `mean_wave_dir` | deg | Mean wave direction |
| `water_temp` | degC | Sea surface temperature |
| `tide` | m | Tide level |
| `wind_gust` | m s-1 | Peak wind gust |

## QC Handling

NDBC does not provide per-observation QC flags. Missing data is encoded as sentinel
values (99, 999, 9999). The pre-parsed parquet files already have sentinels converted
to NaN by the upstream parser.

The `NdbcSentinelRule` exists to catch any sentinel values that leak through, but the
primary QC comes from:

1. **PhysicalBoundsRule** (Tier 1) — checks tair, td, wind, wind_dir, slp
2. **DewpointConsistencyRule** (Tier 1) — checks td <= tair

## DatetimeIndex Handling

The on-disk parquet files have a tz-naive DatetimeIndex (from the dads-mvp pipeline).
The adapter resets this index, renames the resulting column to `datetime_utc`, and
localizes to UTC before normalization.

## Workflow

```bash
# Normalize pre-existing parquet files to canonical schema
uv run obsmet normalize ndbc --start 1970-01-01 --end 2025-12-31 --workers 4
```

**Raw input:** `/nas/climate/ndbc/ndbc_records/{station_id}.parquet`

**Normalized output:** `/mnt/mco_nas1/shared/obsmet/normalized/ndbc/{STATION_ID}.parquet`
