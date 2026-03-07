# CLI Reference

## Typical MADIS Workflow

```bash
uv run obsmet ingest madis --start 2018-01-01 --end 2024-12-31
uv run obsmet normalize madis --start 2018-01-01 --end 2024-12-31 --qc-profile strict
uv run obsmet build station-por --source madis --start 2018-01-01 --end 2024-12-31
uv run obsmet release build --version 1.0 --channel candidate --source madis
uv run obsmet release validate --version 1.0
uv run obsmet release promote --version 1.0 --channel prod
```

## Common Options

These flags are available on most commands:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--start` | datetime | required | Start date (YYYY-MM-DD) |
| `--end` | datetime | required | End date (YYYY-MM-DD) |
| `--resume` / `--no-resume` | flag | resume | Resume from manifest state, skipping completed files |
| `--workers` | int | 4 | Number of parallel workers |
| `--overwrite` | flag | off | Overwrite existing outputs |
| `--dry-run` | flag | off | Show what would be done without executing |

## Commands

### ingest

Download raw observation files from a source archive.

```bash
uv run obsmet ingest <source> --start DATE --end DATE [options]
```

**Sources:** `madis`, `isd`, `gdas`, `raws`, `ndbc`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--bounds` | string | none | Spatial filter: `west,south,east,north` |
| `--qcr-mask` | int | 115 | MADIS QCR reject bitmask |
| `--raw-dir` | path | source-specific (see below) | Override raw data directory |

**Example:**

```bash
uv run obsmet ingest madis --start 2024-01-01 --end 2024-01-31 --workers 4
```

### normalize

Parse raw files into canonical parquet format with Tier 0+1 QC.

```bash
uv run obsmet normalize <source> --start DATE --end DATE [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--qc-profile` | choice | strict | QC aggressiveness: `strict` or `permissive` |
| `--qcr-mask` | int | 115 | Override QCR reject bitmask directly |
| `--bounds` | string | none | Spatial filter: `west,south,east,north` |
| `--raw-dir` | path | auto | Override raw input directory |
| `--out-dir` | path | auto | Override output directory |

**Example:**

```bash
uv run obsmet normalize madis --start 2024-01-01 --end 2024-01-31 \
    --qc-profile strict --bounds "-125,42,-104,49"
```

### build

Build curated products from normalized observations.

```bash
uv run obsmet build <product> [options]
```

**Products:** `hourly`, `daily`, `station-por`, `fabric`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--source` | string | all | Filter by source |
| `--start` / `--end` | datetime | full range | Date range for product |

**Example:**

```bash
uv run obsmet build station-por --source madis --start 2018-01-01 --end 2024-12-31
```

### release

Manage versioned release snapshots.

#### release build

```bash
uv run obsmet release build --version VERSION --channel CHANNEL [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--version` | string | required | Release version (e.g., `1.0`, `2.1`) |
| `--channel` | string | required | Target channel: `candidate` or `prod` |
| `--source` | string | required | Source to include |
| `--qc-profile` | string | strict | QC profile used (recorded in metadata) |

#### release validate

```bash
uv run obsmet release validate --version VERSION
```

Verifies all file checksums in the release manifest match the actual files.

#### release promote

```bash
uv run obsmet release promote --version VERSION --channel CHANNEL
```

Updates a channel symlink to point to the specified release version.

**Example:**

```bash
uv run obsmet release build --version 2.1 --channel candidate --source madis
uv run obsmet release validate --version 2.1
uv run obsmet release promote --version 2.1 --channel prod
```

### qaqc

Apply QAQC rules to normalized observations.

```bash
uv run obsmet qaqc <source|all> --start DATE --end DATE
```

Runs the full QC pipeline (Tiers 0–2) and writes QC-annotated output.

### crosswalk

Build and manage station crosswalks across sources.

```bash
uv run obsmet crosswalk build
```

Matches stations across sources by location and metadata, producing a crosswalk table for
deduplication.

### corrections

Compute correction factors between station observations and gridded products.

```bash
uv run obsmet corrections compute --baseline PRODUCT
```

### diagnostics

Generate data quality and coverage reports.

#### diagnostics coverage

```bash
uv run obsmet diagnostics coverage [source]
```

Reports observation counts, station counts, and temporal coverage by variable.

#### diagnostics latency

```bash
uv run obsmet diagnostics latency [source]
```

Reports data latency (time between observation and availability in obsmet).

#### diagnostics qc

```bash
uv run obsmet diagnostics qc [source]
```

Reports QC pass/fail/suspect rates by variable, station, and time period.

## Data Paths

Raw data is stored at source-specific locations under `/nas/climate/`. Normalized outputs
and products are written to `/mnt/mco_nas1/shared/obsmet/`.

### Raw Data Defaults

| Source | Default raw path |
|--------|-----------------|
| MADIS | `/nas/climate/madis/LDAD/mesonet/netCDF` |
| ISD | `/nas/climate/isd/raw` |
| GDAS | `/nas/climate/gdas/prepbufr` |
| RAWS | `/nas/climate/raws/wrcc/station_data` |
| NDBC | `/nas/climate/ndbc/ndbc_records` |

### Normalized + Products

| Data | Path |
|------|------|
| Normalized | `/mnt/mco_nas1/shared/obsmet/normalized/<source>/` |
| Daily products | `/mnt/mco_nas1/shared/obsmet/products/daily/` |
| Station POR | `/mnt/mco_nas1/shared/obsmet/products/station_por/` |
| Releases | `/mnt/mco_nas1/shared/obsmet/releases/v<version>/` |
| Channels | `/mnt/mco_nas1/shared/obsmet/channels/{candidate,prod}` |
