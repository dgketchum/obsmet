# GDAS Data Source

## What GDAS Is

The **Global Data Assimilation System (GDAS)** is operated by NCEP as part of the GFS
model cycle. Every 6 hours, GDAS assimilates surface and upper-air observations worldwide
into the model analysis. The raw observations ingested by GDAS are archived as **PrepBUFR**
files, which contain quality-marked surface (ADPSFC) and ship (SFCSHP) reports along with
the quality markers assigned by NCEP's multi-stage QC pipeline.

obsmet extracts surface observations from PrepBUFR archives, providing access to global
station data with NCEP's quality markers preserved.

## Archive Coverage

| Property | Value |
|----------|-------|
| **Period of record** | 1997 through present |
| **Temporal resolution** | 6-hourly (00, 06, 12, 18 UTC cycles) |
| **File format** | BUFR (tar.gz archives) + pre-extracted parquet |
| **File naming** | `prepbufr.YYYYMMDD.nr.tar.gz` |
| **Archive source** | NSF NCAR GDEX |
| **Organization** | One archive per day, containing 4 cycle files |
| **Total volume** | ~741 GB BUFR + pre-extracted parquet cache |

## Access

GDAS PrepBUFR archives are downloaded from the NSF NCAR GDEX archive. obsmet supports
two data paths:

1. **BUFR decode** — full extraction from raw PrepBUFR archives using `eccodes`
2. **Pre-extracted parquet fast path** — reads previously decoded parquet files at
   `raw_dir/parquet/YYYY/YYYYMMDD.parquet`, avoiding the expensive BUFR decode step

The fast path is used automatically when pre-extracted parquet exists. This is the primary
path for production runs where the BUFR data has already been decoded by prior systems
(e.g., GDASApp).

## Variables Extracted

| GDAS Variable | Canonical Name | Unit | Description |
|---------------|----------------|------|-------------|
| `temperature` | `tair` | degC | Air temperature |
| `pressure` | `psfc` | Pa | Surface pressure |
| `specific_humidity` | `q` | kg kg-1 | Specific humidity |
| `u_wind` | `u` | m s-1 | Zonal wind component |
| `v_wind` | `v` | m s-1 | Meridional wind component |
| `sst` | `sst` | K | Sea surface temperature |

The extraction layer handles unit conversions (hPa to Pa, mg/kg to kg/kg), so no
additional conversion is needed during normalization.

## Quality Markers

GDAS PrepBUFR files include per-variable quality markers (QM) from NCEP's QC pipeline.
These are preserved as columns in the normalized output:

| QM Value | Meaning | obsmet Interpretation |
|----------|---------|----------------------|
| 0 | Not checked | Pass |
| 1 | Good | Pass |
| 2 | Neutral | Pass |
| 3 | Suspect (inflated error) | Suspect |
| 4-14 | Rejected by various QC steps | Fail |
| 15 | Purged | Fail |

QM columns in output: `tair_qm`, `psfc_qm`, `q_qm`, `u_qm`, `v_qm`, `sst_qm`.

Additional metadata preserved: `cycle` (00/06/12/18), `obs_type`, `msg_type`
(ADPSFC/SFCSHP).

After QM column preservation, normalized output passes through:

1. **PhysicalBoundsRule** (Tier 1) — checks tair, td, wind, wind_dir, psfc
2. **DewpointConsistencyRule** (Tier 1) — checks td <= tair

## Nullable Dtype Handling

Pre-extracted parquet files use pandas nullable integer types (`Int8`, `Float64`) for QM
columns. The adapter converts these to standard numpy `float64` (with `np.nan` for missing)
before passing to the QC pipeline, which expects standard floats.

## Workflow

```bash
# Download PrepBUFR archives from GDEX
uv run obsmet ingest gdas --start 2024-01-01 --end 2024-12-31 --workers 4

# Normalize using fast path (pre-extracted parquet)
uv run obsmet normalize gdas --start 1997-01-01 --end 2025-12-31 --workers 4
```

**Raw input:** `/nas/climate/gdas/prepbufr/YYYY/prepbufr.YYYYMMDD.nr.tar.gz`

**Fast path input:** `/nas/climate/gdas/prepbufr/parquet/YYYY/YYYYMMDD.parquet`

**Normalized output:** `/mnt/mco_nas1/shared/obsmet/normalized/gdas/YYYYMMDD.parquet`
