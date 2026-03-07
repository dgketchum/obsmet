# GHCNh Data Source

## What GHCNh Is

The **Global Historical Climatology Network - Hourly (GHCNh)** is NCEI's next-generation
hourly surface observation dataset, replacing the Integrated Surface Database (ISD). GHCNh
subsumes all ISD stations plus additional sources, with approximately 24,800 stations
worldwide. Records extend from the 1890s through present.

GHCNh uses GHCN-aligned station identifiers, making crosswalks to GHCN-Daily trivial.
Station files contain the full period of record in a single file, eliminating the per-year
fragmentation of ISD.

## Archive Coverage

| Property | Value |
|----------|-------|
| **Period of record** | 1890s through present |
| **Temporal resolution** | Hourly (sub-hourly for some stations) |
| **File format** | PSV (pipe-separated values) |
| **File naming** | `{STATION_ID}.psv` (e.g., `USW00024153.psv`) |
| **Access** | HTTPS bulk download (no credentials) |
| **Organization** | One file per station (full POR) |
| **Station count** | ~24,800 |
| **Total volume** | ~200 GB |

## Access

GHCNh files are available via HTTPS from NCEI's v1beta distribution. No credentials are
required. obsmet downloads station files using threaded parallelism (8 workers default).
The manifest tracks download state for resume semantics.

## Variables Extracted

GHCNh values are in standard metric units — no scaling is needed (unlike GHCN-Daily tenths).

| GHCNh Column | Canonical Name | Unit |
|--------------|----------------|------|
| `temperature` | `tair` | degC |
| `dew_point_temperature` | `td` | degC |
| `wind_speed` | `wind` | m s-1 |
| `wind_direction` | `wind_dir` | deg |
| `wind_gust` | `wind_gust` | m s-1 |
| `sea_level_pressure` | `slp` | hPa |
| `station_level_pressure` | `psfc` | hPa |
| `precipitation` | `prcp` | mm |
| `relative_humidity` | `rh` | % |
| `wet_bulb_temperature` | `tw` | degC |

## QC Handling

Each variable has a per-observation `Quality_Code` column in the PSV. obsmet filters on
these codes during extraction:

| QC Code | Meaning | obsmet Action |
|---------|---------|---------------|
| `1` | Passed all QC | **Accept** |
| `2` | Not checked | **Accept** |
| `4` | Corrected, passed | **Accept** |
| `5` | Translated (no QC applied) | **Accept** |
| `A` | Adjusted | **Accept** |
| `3` | Failed one test | **Reject** (fail) |
| `6` | Failed multiple tests | **Reject** (fail) |
| `7` | Suspect / erroneous | **Reject** (fail) |

After QC code filtering, the normalized output passes through:

1. **PhysicalBoundsRule** (Tier 1) — rejects values outside plausible physical limits
2. **DewpointConsistencyRule** (Tier 1) — rejects dewpoint exceeding air temperature

## PSV Format

GHCNh files have ~190 columns per row. obsmet reads only the ~30 columns needed for
values and quality codes, skipping measurement type, source, and report type columns
for performance. The selective column read is handled via `usecols` during parsing.

## Workflow

```bash
# Download station PSV files from NCEI
uv run obsmet ingest ghcnh --workers 8

# Normalize to canonical schema with QC
uv run obsmet normalize ghcnh --start 1900-01-01 --end 2025-12-31 --workers 8
```

**Raw input:** `/nas/climate/ghcnh/*.psv`

**Normalized output:** `/mnt/mco_nas1/shared/obsmet/normalized/ghcnh/{STATION_ID}.parquet`
