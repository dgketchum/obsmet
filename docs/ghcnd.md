# GHCN-Daily Data Source

## What GHCN-Daily Is

The **Global Historical Climatology Network - Daily (GHCN-Daily)** is the definitive global
daily precipitation and temperature dataset, maintained by NCEI. It contains approximately
119,000 stations in 180+ countries, with records extending back to the 1840s.

Roughly 80,000 stations in GHCN-Daily are daily-only gauge sites not present in any hourly
network, making this the primary source for daily precipitation totals and temperature
extremes at locations without hourly coverage.

## Archive Coverage

| Property | Value |
|----------|-------|
| **Period of record** | 1840s through present |
| **Temporal resolution** | Daily |
| **File format** | CSV (daily-summaries distribution) |
| **File naming** | `{GHCN_STATION_ID}.csv` (e.g., `USW00024153.csv`) |
| **Access** | HTTPS bulk download (no credentials) |
| **Organization** | One file per station (full POR) |
| **Station count** | ~119,000 |
| **Total volume** | ~130 GB (2022 snapshot on disk) |

## Access

obsmet currently uses an existing 2022 snapshot at
`/nas/climate/ghcn/ghcn_daily_summaries_4FEB2022/`. The current NCEI distribution is
available via HTTPS or AWS S3 (`noaa-ghcn-pds`). No credentials are required.

## Variables Extracted

Most GHCN-Daily values are stored as integers in tenths of the canonical unit and require
scaling during normalization.

| GHCN Element | Canonical Name | Scale | Unit |
|--------------|----------------|-------|------|
| `TMAX` | `tmax` | x0.1 | degC |
| `TMIN` | `tmin` | x0.1 | degC |
| `TAVG` | `tmean` | x0.1 | degC |
| `PRCP` | `prcp` | x0.1 | mm |
| `SNOW` | `snow` | 1.0 | mm |
| `SNWD` | `snow_depth` | 1.0 | mm |
| `AWND` | `wind` | x0.1 | m s-1 |
| `WESD` | `swe` | x0.1 | mm |

## QC Handling

Each element has a paired `{ELEMENT}_ATTRIBUTES` column with format
`measurement_flag,quality_flag,source_flag[,obs_time]`. obsmet checks the quality flag
(second field):

| Quality Flag | Meaning | obsmet Action |
|--------------|---------|---------------|
| `D` | Duplicate | **Reject** |
| `I` | Internal consistency failure | **Reject** |
| `K` | Streak / frequent value | **Reject** |
| `L` | Multiday accumulation | **Reject** |
| `N` | Climatological outlier | **Reject** |
| `O` | Gap in record | **Reject** |
| `R` | Lagged range | **Reject** |
| `S` | Spatial consistency failure | **Reject** |
| `T` | Temporal consistency failure | **Reject** |
| `W` | Temperature too warm for snow | **Reject** |
| `X` | Failed bounds check | **Reject** |
| (blank) | Passed / not flagged | **Accept** |

After flag filtering, the normalized output passes through:

1. **PhysicalBoundsRule** (Tier 1) — rejects values outside plausible physical limits
2. **DewpointConsistencyRule** (Tier 1) — rejects dewpoint exceeding air temperature

## Day Basis

GHCN-Daily uses `day_basis=local` — each country's local observation conventions, not UTC
midnight. This is the authoritative source for daily extremes and precipitation totals, so
the local day definition is preserved rather than re-aggregated to UTC.

## Workflow

```bash
# Normalize 2022 snapshot to canonical schema
uv run obsmet normalize ghcnd --start 1840-01-01 --end 2025-12-31 --workers 8
```

**Raw input:** `/nas/climate/ghcn/ghcn_daily_summaries_4FEB2022/*.csv`

**Normalized output:** `/mnt/mco_nas1/shared/obsmet/normalized/ghcnd/{STATION_ID}.parquet`
