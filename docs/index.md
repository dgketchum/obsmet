# obsmet

**Obs**ervation **Met**eorology — unified ingest, standardization, QAQC, and provenance
for surface meteorological observations.

## What obsmet does

- **Downloads** hourly and sub-hourly observations from five federal/cooperative networks
- **Normalizes** heterogeneous native formats into a single canonical schema with consistent units
- **Quality-controls** observations through a tiered rule system (source-native flags, physical bounds, temporal statistics)
- **Builds curated products** — station period-of-record parquets, daily aggregations, versioned releases with checksums
- **Tracks provenance** end-to-end: every output value links back to its raw source file, ingest run, and QC rule version

## Supported Sources

| Source | Network | Coverage | Status |
|--------|---------|----------|--------|
| **MADIS** | NOAA ESRL mesonet (LDAD tier) | 2001–present, hourly | Active |
| **ISD** | NOAA Integrated Surface Database | 1901–present, hourly | Active |
| **GDAS** | NCEP Global Data Assimilation System (ADPSFC/SFCSHP) | 2004–present, 6-hourly | Active |
| **RAWS** | Remote Automatic Weather Stations (WRCC) | 1980s–present, hourly | Active |
| **NDBC** | National Data Buoy Center | 1970s–present, hourly | Active |

## Quick Links

- [Architecture Overview](overview.md) — the 5-layer pipeline and design principles
- [MADIS Pipeline](madis/index.md) — end-to-end workflow for MADIS observations
- [Quality Control](madis/qaqc.md) — the tiered QC system explained
- [CLI Reference](cli_reference.md) — all commands, flags, and example invocations
