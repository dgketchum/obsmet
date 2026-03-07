# RAWS Data Source

## What RAWS Is

**Remote Automatic Weather Stations (RAWS)** are operated across the western United States
primarily for wildland fire management. The stations are maintained by federal and state
land management agencies (BLM, USFS, NPS) and report daily summaries through the **Western
Regional Climate Center (WRCC)**. RAWS stations are especially valuable in the rural and
mountainous western US where other networks are sparse.

## Archive Coverage

| Property | Value |
|----------|-------|
| **Period of record** | 1980s through present (varies by station) |
| **Temporal resolution** | Daily |
| **File format** | CSV (per-station files) |
| **File naming** | `{wrcc_id}.csv` (e.g., `idBOIS.csv`) |
| **Access method** | WRCC web scraper |
| **Organization** | One file per station, all dates |
| **Station count** | ~2,051 stations |

RAWS data arrives at daily resolution — unlike the other obsmet sources, there is no
hourly-to-daily aggregation step.

## Access

RAWS data is scraped from the WRCC web interface. The ingest process:

1. Scrapes the WRCC station inventory across all western US regions
2. Downloads daily observation HTML for each station
3. Parses the HTML response into tabular data
4. Writes per-station CSV files

## Variables Extracted

All RAWS data arrives in metric units — no unit conversion is needed:

| WRCC Column | Canonical Name | Unit | Description |
|-------------|----------------|------|-------------|
| `tair_ave_c` | `tmean` | degC | Mean air temperature |
| `tair_max_c` | `tmax` | degC | Maximum air temperature |
| `tair_min_c` | `tmin` | degC | Minimum air temperature |
| `wspd_ave_ms` | `wind` | m s-1 | Mean wind speed |
| `wdir_vec_deg` | `wind_dir` | deg | Vector wind direction |
| `wspd_gust_ms` | `wind_gust` | m s-1 | Peak wind gust |
| `rh_ave_pct` | `rh` | percent | Mean relative humidity |
| `rh_max_pct` | `rh_max` | percent | Maximum relative humidity |
| `rh_min_pct` | `rh_min` | percent | Minimum relative humidity |
| `prcp_total_mm` | `prcp` | mm | Total precipitation |
| `srad_total_kwh_m2` | `rsds` | kWh m-2 | Total solar radiation |

## QC Handling

WRCC does not provide per-observation QC flags. The normalized output passes through:

1. **PhysicalBoundsRule** (Tier 1) — checks tmean, tmax, tmin, wind, wind_dir, rh, prcp, rsds
2. **DewpointConsistencyRule** (Tier 1) — skipped (RAWS has no dewpoint variable)

RAWS runs serially (`parallel=False` in the registry) because the station count is small
and individual files are fast to process.

## Workflow

```bash
# Scrape and download RAWS station data from WRCC
uv run obsmet ingest raws --start 1990-01-01 --end 2025-12-31

# Normalize to canonical schema with QC
uv run obsmet normalize raws --start 1990-01-01 --end 2025-12-31 --workers 1
```

**Raw input:** `/nas/climate/raws/wrcc/station_data/{wrcc_id}.csv`

**Normalized output:** `/mnt/mco_nas1/shared/obsmet/normalized/raws_wrcc/{wrcc_id}.parquet`
