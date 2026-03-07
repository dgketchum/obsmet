# ISD Data Source

## What ISD Is

The **Integrated Surface Database (ISD)** is maintained by NOAA's National Centers for
Environmental Information (NCEI). It contains hourly surface observations from over 30,000
stations worldwide, collected from METAR, SYNOP, ASOS, AWOS, and other reporting networks.
ISD is the largest single archive of surface weather observations, with records extending
back to 1901.

## Archive Coverage

| Property | Value |
|----------|-------|
| **Period of record** | 1901 through present |
| **Temporal resolution** | Hourly (sub-hourly for some stations) |
| **File format** | Gzip-compressed fixed-width text |
| **File naming** | `USAF-NCEI-YYYY.gz` (e.g., `720538-00164-2024.gz`) |
| **S3 bucket** | `noaa-isd-pds` (public, no credentials required) |
| **Organization** | One file per station-year |
| **Total volume** | ~13,000 station-year files per year (modern era) |

Each station-year file contains all hourly reports for that station in that year.

## Access

ISD is hosted on a public S3 bucket (`noaa-isd-pds`) with unsigned access — no AWS
credentials are needed. obsmet downloads files using `boto3` with threaded parallelism
(16 threads default). Files that already exist locally are skipped automatically.

## Variables Extracted

ISD uses a fixed-width format with positions encoding each variable. obsmet extracts these
from the mandatory data section (positions 0-104):

| ISD Field | Position | Raw Scale | Canonical Name | Canonical Unit |
|-----------|----------|-----------|----------------|----------------|
| `air_temperature` | 87-92 | /10 | `tair` | degC |
| `dew_point_temperature` | 93-98 | /10 | `td` | degC |
| `wind_speed` | 65-69 | /10 | `wind` | m s-1 |
| `wind_direction` | 60-63 | integer | `wind_dir` | deg |
| `sea_level_pressure` | 99-104 | /10 (hPa) | `slp` | Pa |

Pressure is the only variable requiring unit conversion during normalization (hPa to Pa,
multiply by 100). Temperature and wind speed are already in canonical units after the /10
scaling during parsing.

## QC Handling

Each ISD variable has a single-character quality code at a fixed position immediately
following the value. obsmet uses these during extraction:

| QC Code | Meaning | obsmet Action |
|---------|---------|---------------|
| `1` | Passed all QC checks | **Accept** |
| `5` | Estimated value | **Accept** |
| `0` | Not checked | **Reject** (NaN) |
| `2` | Suspect | **Reject** (NaN) |
| `3` | Erroneous | **Reject** (NaN) |
| `9` | Missing | **Reject** (NaN) |

Values with QC codes not in `{1, 5}` are set to NaN during extraction, before
normalization. This means the normalized output already has bad values removed. The
`IsdQualityCodeRule` exists as a formal rule class but is not run redundantly in the
pipeline since filtering happens at extract time.

After extraction filtering, the normalized output passes through:

1. **PhysicalBoundsRule** (Tier 1) — rejects values outside plausible physical limits
2. **DewpointConsistencyRule** (Tier 1) — rejects dewpoint exceeding air temperature

## Missing Value Sentinels

ISD encodes missing data as field-width-specific sentinel strings:

| Variable | Sentinel |
|----------|----------|
| Latitude | `+99999` |
| Longitude | `+999999` |
| Wind direction | `999` |
| Wind speed | `9999` |
| Temperature | `+9999` |
| Dewpoint | `+9999` |
| Pressure | `99999` |

These are detected during parsing and returned as `None`/NaN.

## Workflow

```bash
# Download 2024 station-year files from S3
uv run obsmet ingest isd --start 2024-01-01 --end 2024-12-31 --workers 8

# Normalize to canonical schema with QC
uv run obsmet normalize isd --start 2024-01-01 --end 2024-12-31 --workers 8
```

**Raw input:** `/nas/climate/isd/raw/YYYY/USAF-NCEI-YYYY.gz`

**Normalized output:** `/mnt/mco_nas1/shared/obsmet/normalized/isd/YYYY_USAF-NCEI-YYYY.parquet`
