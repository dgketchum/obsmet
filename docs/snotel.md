# SNOTEL Data Source

## What SNOTEL Is

The **Snow Telemetry (SNOTEL)** network is operated by the USDA Natural Resources
Conservation Service (NRCS). It consists of approximately 855 automated high-elevation
stations in western US mountains (typically 1,000-3,500 m elevation), measuring snow water
equivalent (SWE), precipitation, temperature, and sometimes humidity and wind.

SNOTEL is the primary source for snowpack monitoring and water supply forecasting in the
western United States.

## Archive Coverage

| Property | Value |
|----------|-------|
| **Period of record** | 1980s through present |
| **Temporal resolution** | Daily |
| **File format** | CSV (pre-downloaded) |
| **File naming** | `{site_id}_{name}_{state}.csv` (e.g., `100_Bear_Mountain_ID.csv`) |
| **Access** | Pre-downloaded archive (NRCS AWDB API for updates) |
| **Organization** | One file per station (full POR) |
| **Station count** | ~855 |
| **Total volume** | ~670 MB |

## Access

obsmet uses a pre-downloaded archive at `/nas/climate/snotel/snotel_records/` (July 2024
snapshot). For incremental updates, the NRCS AWDB web service is available at
`https://wcc.sc.egov.usda.gov/awdbWebService/`.

## Variables Extracted

Units are already metric — converted during the original NRCS download. No scaling is
needed during normalization.

| CSV Column | Canonical Name | Unit | Notes |
|------------|----------------|------|-------|
| `swe` | `swe` | mm | Snow water equivalent |
| `tmin` | `tmin` | degC | Daily minimum temperature |
| `tmax` | `tmax` | degC | Daily maximum temperature |
| `tavg` | `tmean` | degC | Daily mean temperature |
| `prec` | `prcp` | mm | Cumulative — differenced to daily |
| `rh` | `rh` | % | Relative humidity (sparse) |
| `ws` | `wind` | m s-1 | Wind speed (sparse) |

## QC Handling

SNOTEL provides no source-native QC flags. Minimal NRCS validation is applied upstream.
The normalized output passes through:

1. **PhysicalBoundsRule** (Tier 1) — rejects values outside plausible physical limits
2. **DewpointConsistencyRule** (Tier 1) — rejects dewpoint exceeding air temperature

## Cumulative Precipitation

The `prec` column in SNOTEL CSV files is cumulative season-to-date precipitation. The
adapter differences consecutive days to compute daily precipitation totals. Negative
differences (gauge resets at start of water year) are set to NaN.

## Workflow

```bash
# Normalize pre-existing CSV files to canonical schema
uv run obsmet normalize snotel --start 1980-01-01 --end 2025-12-31 --workers 4
```

**Raw input:** `/nas/climate/snotel/snotel_records/*.csv`

**Normalized output:** `/mnt/mco_nas1/shared/obsmet/normalized/snotel/{station_key}.parquet`
