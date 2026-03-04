# MADIS Data Source

## What MADIS Is

The **Meteorological Assimilation Data Ingest System (MADIS)** is operated by NOAA's Earth
System Research Laboratory (ESRL). It collects, quality-controls, and distributes surface
observations from thousands of stations across the United States, drawn from federal networks
(ASOS, AWOS), state mesonets (Oklahoma, West Texas, MesoWest), cooperative observer networks,
and private networks.

obsmet uses the **LDAD (Local Data Acquisition and Dissemination) mesonet tier**, which provides
2–3x more stations than the public MADIS feed — particularly important in the western US where
public networks are sparse. LDAD access requires a MADIS Research account.

## Archive Coverage

| Property | Value |
|----------|-------|
| **Period of record** | 2001-07-01 through present |
| **Temporal resolution** | Hourly |
| **File format** | gzip-compressed netCDF |
| **File naming** | `YYYYMMDD_HHMM.gz` |
| **Archive URL** | `https://madis-data.ncep.noaa.gov/madisResearch/data/archive/{YYYY}/{MM}/{DD}/LDAD/mesonet/netCDF/` |
| **File sizes** | ~300 KB (2001) to ~35 MB (2025), growing with network expansion |
| **Total volume** | ~210,000+ hourly files as of early 2026 |

Each hourly file contains all reporting stations for that hour across CONUS and adjacent areas.

## Access

MADIS Research accounts are available to qualifying researchers. Authentication is via
username/password over HTTPS. obsmet parallelizes downloads with `aria2c` (16 connections per
file, 5 files concurrent) with `wget` as a fallback.

## Variables Extracted

obsmet extracts the following variables from MADIS netCDF files:

| MADIS Variable | Canonical Name | Native Unit | Canonical Unit | Description |
|----------------|----------------|-------------|----------------|-------------|
| `temperature` | `tair` | K | degC | Air temperature |
| `dewpoint` | `td` | K | degC | Dewpoint temperature |
| `relHumidity` | `rh` | % | percent | Relative humidity |
| `windSpeed` | `wind` | m/s | m s-1 | Wind speed |
| `windDir` | `wind_dir` | deg | deg | Wind direction |
| `precipAccum` | `prcp` | mm | mm | Precipitation accumulation |
| `solarRadiation` | `rsds_hourly` | W/m² | W m-2 | Downward shortwave radiation |

## QC Fields Used

Each MADIS observation carries two QC indicators that obsmet uses for Tier 0 quality control:

- **DD (Data Descriptor)** — a single-character flag summarizing upstream QC status
- **QCR (QC Results)** — a bitmask indicating which MADIS internal checks the observation failed

See [Quality Control](qaqc.md) for how these are applied.

## Spatial Coverage

MADIS mesonet observations cover CONUS and adjacent areas (southern Canada, northern Mexico,
Caribbean, Pacific islands). Station density is highest in the central and eastern US, with
sparser but growing coverage in the western interior. The LDAD tier significantly improves
western coverage through inclusion of state mesonets and cooperative networks.
