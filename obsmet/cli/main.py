"""obsmet CLI entry point (plan section 17).

All commands support --start, --end, --resume, --workers, --overwrite, --dry-run.
"""

from __future__ import annotations

import click


def common_options(fn):
    """Shared options across all pipeline commands."""
    fn = click.option("--start", type=click.DateTime(), default=None, help="Start date")(fn)
    fn = click.option("--end", type=click.DateTime(), default=None, help="End date")(fn)
    fn = click.option("--resume/--no-resume", default=True, help="Resume from manifest state")(fn)
    fn = click.option("--workers", type=int, default=4, help="Parallel workers")(fn)
    fn = click.option("--overwrite", is_flag=True, help="Overwrite existing outputs")(fn)
    fn = click.option("--dry-run", is_flag=True, help="Show what would be done")(fn)
    return fn


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """obsmet — unified meteorological observation pipeline."""


# --------------------------------------------------------------------------- #
# Default paths per source
# --------------------------------------------------------------------------- #

_DEFAULT_RAW_DIRS = {
    "madis": "/nas/climate/madis/LDAD/mesonet/netCDF",
    "isd": "/nas/climate/isd/raw",
    "gdas": "/nas/climate/gdas/prepbufr",
    "raws": "/nas/climate/raws/wrcc/station_data",
    "ndbc": "/nas/climate/ndbc/ndbc_records",
}

_DEFAULT_NORM_DIRS = {
    "madis": "/mnt/mco_nas1/shared/obsmet/normalized/madis",
    "isd": "/mnt/mco_nas1/shared/obsmet/normalized/isd",
    "gdas": "/mnt/mco_nas1/shared/obsmet/normalized/gdas",
    "raws": "/mnt/mco_nas1/shared/obsmet/normalized/raws_wrcc",
    "ndbc": "/mnt/mco_nas1/shared/obsmet/normalized/ndbc",
}

# Manifest source names (some differ from CLI source name)
_MANIFEST_SOURCE = {
    "madis": "madis",
    "isd": "isd",
    "gdas": "gdas",
    "raws": "raws_wrcc",
    "ndbc": "ndbc",
}


# --------------------------------------------------------------------------- #
# Ingest command
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("source")
@click.option("--raw-dir", default=None, help="Override raw data directory")
@click.option(
    "--bounds",
    default=None,
    help="Spatial filter: west,south,east,north (e.g. -125,24,-66,53)",
)
@click.option(
    "--qcr-mask",
    type=int,
    default=None,
    help="MADIS QCR reject bitmask (default 115)",
)
@common_options
def ingest(source, raw_dir, bounds, qcr_mask, start, end, resume, workers, overwrite, dry_run):
    """Download/scrape raw data for SOURCE."""
    _INGEST_DISPATCH = {
        "madis": _ingest_madis,
        "isd": _ingest_isd,
        "gdas": _ingest_gdas,
        "raws": _ingest_raws,
        "ndbc": _ingest_ndbc,
    }
    fn = _INGEST_DISPATCH.get(source)
    if fn is None:
        click.echo(f"Source {source!r} not yet implemented.")
        return
    fn(
        start=start,
        end=end,
        raw_dir=raw_dir,
        bounds=bounds,
        qcr_mask=qcr_mask,
        resume=resume,
        workers=workers,
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# Normalize command — generic dispatch
# --------------------------------------------------------------------------- #


def _normalize_one(key, source_name, raw_dir, out_dir, run_id, overwrite, adapter_kwargs):
    """Module-level worker for normalize (picklable for ProcessPoolExecutor).

    Creates adapter inside the worker via registry (lazy import).
    """
    from pathlib import Path

    from obsmet.core.provenance import RunProvenance
    from obsmet.sources.registry import create_adapter

    adapter = create_adapter(source_name, raw_dir=raw_dir, **adapter_kwargs)
    provenance = RunProvenance(
        source=_MANIFEST_SOURCE[source_name], command="normalize", run_id=run_id
    )
    out_dir = Path(out_dir)

    out_path = out_dir / adapter.output_filename(key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return key, None, None

    df = adapter.normalize_key(key, provenance)
    if df is None or df.empty:
        return key, 0, "no data"

    from obsmet.qaqc.engines.pipeline import (
        _VARIABLE_COLUMNS,
        apply_pipeline_to_df,
        build_default_pipeline,
    )

    pipeline = build_default_pipeline(source_name)
    var_cols = _VARIABLE_COLUMNS.get(source_name, [])
    if var_cols:
        present = [c for c in var_cols if c in df.columns]
        if present:
            df = apply_pipeline_to_df(df, pipeline, present, source=source_name)

    df.to_parquet(out_path, index=False, compression="snappy")
    return key, len(df), None


def _run_normalize(
    source_name, start, end, raw_dir, out_dir, resume, workers, overwrite, dry_run, **adapter_kwargs
):
    """Generic normalize pipeline for any source."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial
    from pathlib import Path

    from obsmet.core.manifest import Manifest
    from obsmet.core.provenance import RunProvenance
    from obsmet.sources.registry import create_adapter, get_source

    entry = get_source(source_name)
    msrc = _MANIFEST_SOURCE[source_name]

    raw_dir = raw_dir or _DEFAULT_RAW_DIRS[source_name]
    out_dir = Path(out_dir) if out_dir else Path(_DEFAULT_NORM_DIRS[source_name])
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = create_adapter(source_name, raw_dir=raw_dir, **adapter_kwargs)
    provenance = RunProvenance(source=msrc, command="normalize")

    manifest_path = out_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source=msrc)

    if start is None or end is None:
        click.echo(f"Error: --start and --end are required for {source_name} normalize.")
        return

    keys = adapter.discover_keys(start, end)
    if resume:
        keys = manifest.pending_keys(keys)

    click.echo(f"{source_name.upper()} normalize: {len(keys)} keys, {workers} workers → {out_dir}")

    if dry_run:
        for k in keys[:10]:
            click.echo(f"  would normalize: {k}")
        if len(keys) > 10:
            click.echo(f"  ... and {len(keys) - 10} more")
        return

    worker = partial(
        _normalize_one,
        source_name=source_name,
        raw_dir=raw_dir,
        out_dir=str(out_dir),
        run_id=provenance.run_id,
        overwrite=overwrite,
        adapter_kwargs=adapter_kwargs,
    )

    done = 0
    errors = 0
    skipped = 0

    use_parallel = entry.parallel and workers > 1

    if use_parallel:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, k): k for k in keys}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    _, nrows, err = fut.result()
                except Exception as exc:
                    manifest.update(key, "failed", run_id=provenance.run_id, message=str(exc))
                    errors += 1
                    done += 1
                    continue
                if nrows is None:
                    skipped += 1
                elif err:
                    manifest.update(key, "missing", run_id=provenance.run_id, message=err)
                    errors += 1
                else:
                    manifest.update(key, "done", run_id=provenance.run_id)
                done += 1
                if done % 100 == 0:
                    click.echo(f"  progress: {done}/{len(keys)}, {skipped} skip, {errors} err")
                    manifest.flush()
    else:
        for key in keys:
            try:
                _, nrows, err = worker(key)
            except Exception as exc:
                manifest.update(key, "failed", run_id=provenance.run_id, message=str(exc))
                errors += 1
                done += 1
                continue
            if nrows is None:
                skipped += 1
            elif err:
                manifest.update(key, "missing", run_id=provenance.run_id, message=err)
                errors += 1
            else:
                manifest.update(key, "done", run_id=provenance.run_id)
            done += 1
            if done % 100 == 0:
                click.echo(f"  progress: {done}/{len(keys)}, {skipped} skip, {errors} err")
                manifest.flush()

    manifest.flush()
    click.echo(
        f"{source_name.upper()} normalize done: {done} processed, {skipped} skipped, {errors} errors"
    )


@cli.command()
@click.argument("source")
@click.option("--raw-dir", default=None, help="Override raw data directory")
@click.option("--out-dir", default=None, help="Override output directory")
@click.option(
    "--bounds",
    default=None,
    help="Spatial filter: west,south,east,north (e.g. -125,24,-66,53)",
)
@click.option(
    "--qcr-mask",
    type=int,
    default=None,
    help="MADIS QCR reject bitmask (default 115)",
)
@click.option(
    "--qc-profile",
    type=click.Choice(["strict", "permissive"]),
    default=None,
    help="QC profile (sets qcr_mask; overridden by explicit --qcr-mask)",
)
@common_options
def normalize(
    source,
    raw_dir,
    out_dir,
    bounds,
    qcr_mask,
    qc_profile,
    start,
    end,
    resume,
    workers,
    overwrite,
    dry_run,
):
    """Parse raw files into canonical observation rows for SOURCE."""
    from obsmet.sources.registry import list_sources

    if source not in list_sources():
        click.echo(f"Source {source!r} not yet implemented.")
        return

    # Build source-specific adapter kwargs
    adapter_kwargs = {}
    if source == "madis":
        if bounds:
            adapter_kwargs["bounds"] = tuple(float(x) for x in bounds.split(","))
        if qc_profile is not None:
            from obsmet.qaqc.engines.pipeline import QC_PROFILES

            adapter_kwargs["qcr_mask"] = QC_PROFILES[qc_profile]["qcr_mask"]
        if qcr_mask is not None:
            adapter_kwargs["qcr_mask"] = qcr_mask  # explicit override wins

    _run_normalize(
        source,
        start,
        end,
        raw_dir,
        out_dir,
        resume,
        workers,
        overwrite,
        dry_run,
        **adapter_kwargs,
    )


# --------------------------------------------------------------------------- #
# QAQC, Build, Crosswalk, Corrections, Diagnostics
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("source", default="all")
@common_options
def qaqc(source, start, end, resume, workers, overwrite, dry_run):
    """Apply QAQC rules to SOURCE (or all sources)."""
    click.echo(f"qaqc {source}: start={start} end={end}")


@cli.command()
@click.argument("product", type=click.Choice(["hourly", "daily", "station-por", "fabric"]))
@click.option(
    "--source",
    default="all",
    help="Source to build from (default: all)",
)
@common_options
def build(product, source, start, end, resume, workers, overwrite, dry_run):
    """Build curated PRODUCT (hourly, daily, station-por, or fabric)."""
    if product == "daily":
        _build_daily(source, start, end, resume, workers, overwrite, dry_run)
    elif product == "station-por":
        _build_station_por(source, start, end, resume, workers, overwrite, dry_run)
    else:
        click.echo(f"build {product}: start={start} end={end}")


@cli.group()
def crosswalk():
    """Station identity crosswalk commands."""


@crosswalk.command("build")
@common_options
def crosswalk_build(start, end, resume, workers, overwrite, dry_run):
    """Build or update the station crosswalk."""
    click.echo("crosswalk build")


@cli.group()
def corrections():
    """Correction-factor commands."""


@corrections.command("compute")
@click.option("--baseline", type=str, required=True, help="Baseline source (e.g. era5)")
@common_options
def corrections_compute(baseline, start, end, resume, workers, overwrite, dry_run):
    """Compute monthly correction factors against BASELINE."""
    click.echo(f"corrections compute baseline={baseline}")


@cli.group()
def diagnostics():
    """Diagnostics and monitoring commands."""


@diagnostics.command("coverage")
@click.argument("source", default="all")
@common_options
def diag_coverage(source, start, end, resume, workers, overwrite, dry_run):
    """Report observation coverage by source."""
    click.echo(f"diagnostics coverage {source}")


@diagnostics.command("latency")
@click.argument("source", default="all")
def diag_latency(source):
    """Report data latency by source."""
    click.echo(f"diagnostics latency {source}")


@diagnostics.command("qc")
@click.argument("source", default="all")
@common_options
def diag_qc(source, start, end, resume, workers, overwrite, dry_run):
    """Report QC pass/fail/suspect rates."""
    click.echo(f"diagnostics qc {source}")


# --------------------------------------------------------------------------- #
# Release commands
# --------------------------------------------------------------------------- #


@cli.group()
def release():
    """Release management commands."""


@release.command("build")
@click.option("--version", required=True, help="Release version (e.g. v0.1.0)")
@click.option("--channel", default="candidate", help="Channel name (default: candidate)")
@click.option("--source", multiple=True, default=["madis"], help="Sources to include")
@click.option(
    "--qc-profile",
    type=click.Choice(["strict", "permissive"]),
    default=None,
    help="QC profile used",
)
@click.option("--dry-run", is_flag=True, help="Show what would be done")
def release_build(version, channel, source, qc_profile, dry_run):
    """Build a versioned release from station POR products."""
    from obsmet.core.provenance import RunProvenance
    from obsmet.products.release import build_release

    sources = list(source)
    provenance = RunProvenance(source=",".join(sources), command="release_build")

    if dry_run:
        click.echo(f"Would build release {version} channel={channel} sources={sources}")
        return

    release_dir = build_release(
        version,
        channel,
        sources,
        provenance,
        qc_profile=qc_profile or "",
    )
    click.echo(f"Release built: {release_dir}")


@release.command("promote")
@click.option("--version", required=True, help="Release version to promote")
@click.option("--channel", default="prod", help="Target channel (default: prod)")
def release_promote(version, channel):
    """Promote a release to a channel after validation."""
    from obsmet.products.release import promote_release

    try:
        promote_release(version, channel)
        click.echo(f"Release {version} promoted to channel {channel}")
    except ValueError as exc:
        click.echo(f"Promotion failed: {exc}", err=True)


@release.command("validate")
@click.option("--version", required=True, help="Release version to validate")
def release_validate(version):
    """Validate a release's checksums against its manifest."""
    from obsmet.products.release import validate_release

    ok, errors = validate_release(version)
    if ok:
        click.echo(f"Release {version}: OK")
    else:
        click.echo(f"Release {version}: FAILED ({len(errors)} errors)")
        for e in errors:
            click.echo(f"  {e}")


# --------------------------------------------------------------------------- #
# Ingest implementations (source-specific, kept separate)
# --------------------------------------------------------------------------- #


def _ingest_madis(start, end, raw_dir, resume, dry_run, **_kw):
    """Download raw MADIS files for a date range."""
    from pathlib import Path

    from obsmet.core.manifest import Manifest
    from obsmet.core.provenance import generate_run_id
    from obsmet.sources.madis.download import DEFAULT_RAW_DIR, download_day, load_credentials

    raw_dir = Path(raw_dir) if raw_dir else Path(DEFAULT_RAW_DIR)
    manifest_path = raw_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source="madis")
    run_id = generate_run_id()

    if start is None or end is None:
        click.echo("Error: --start and --end are required for madis ingest.")
        return

    username, password = load_credentials()

    from datetime import timedelta

    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)

    day_strs = [d.strftime("%Y%m%d") for d in days]
    if resume:
        day_strs = manifest.pending_keys(day_strs)

    click.echo(f"MADIS ingest: {len(day_strs)} days to download → {raw_dir}")

    if dry_run:
        for ds in day_strs[:10]:
            click.echo(f"  would download: {ds}")
        if len(day_strs) > 10:
            click.echo(f"  ... and {len(day_strs) - 10} more")
        return

    from datetime import datetime

    ok = 0
    fail = 0
    for ds in day_strs:
        day = datetime.strptime(ds, "%Y%m%d")
        day_str, success, msg = download_day(day, raw_dir, username, password)
        if success:
            manifest.update(ds, "done", run_id=run_id)
            ok += 1
        else:
            manifest.update(ds, "failed", run_id=run_id, message=msg)
            fail += 1

        if (ok + fail) % 50 == 0:
            click.echo(f"  progress: {ok + fail}/{len(day_strs)}, {fail} failures")
            manifest.flush()

    manifest.flush()
    click.echo(f"MADIS ingest done: {ok} ok, {fail} failed")


def _ingest_isd(start, end, raw_dir, resume, workers, dry_run, **_kw):
    """Download raw ISD files from S3 for a year range."""
    from pathlib import Path

    from obsmet.core.manifest import Manifest
    from obsmet.core.provenance import generate_run_id
    from obsmet.sources.isd.download import DEFAULT_RAW_DIR, download_year

    raw_dir = Path(raw_dir) if raw_dir else Path(DEFAULT_RAW_DIR)
    manifest_path = raw_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source="isd")
    run_id = generate_run_id()

    if start is None or end is None:
        click.echo("Error: --start and --end are required for isd ingest.")
        return

    years = list(range(start.year, end.year + 1))
    click.echo(f"ISD ingest: years {years[0]}-{years[-1]}, workers={workers} → {raw_dir}")

    if dry_run:
        for y in years[:5]:
            click.echo(f"  would download year: {y}")
        if len(years) > 5:
            click.echo(f"  ... and {len(years) - 5} more years")
        return

    done_keys = set()
    if resume:
        done_keys = manifest.done_keys()

    for year in years:
        results = download_year(year, raw_dir, workers=workers, done_keys=done_keys)
        ok = sum(1 for _, s, _ in results if s)
        fail = sum(1 for _, s, _ in results if not s)
        click.echo(f"  {year}: {ok} ok, {fail} failed")
        for key, success, msg in results:
            state = "done" if success else "failed"
            manifest.update(key, state, run_id=run_id, message=msg)
        manifest.flush()


def _ingest_gdas(start, end, raw_dir, resume, workers, dry_run, **_kw):
    """Download GDAS PrepBUFR archives from GDEX."""
    from pathlib import Path

    from obsmet.core.manifest import Manifest
    from obsmet.core.provenance import generate_run_id
    from obsmet.sources.gdas_prepbufr.download import DEFAULT_RAW_DIR, download_range

    raw_dir = Path(raw_dir) if raw_dir else Path(DEFAULT_RAW_DIR)
    manifest_path = raw_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source="gdas")
    run_id = generate_run_id()

    if start is None or end is None:
        click.echo("Error: --start and --end are required for gdas ingest.")
        return

    done_dates = set()
    if resume:
        done_dates = manifest.done_keys()

    click.echo(f"GDAS ingest: {start.date()} to {end.date()}, workers={workers} → {raw_dir}")

    if dry_run:
        click.echo("  dry-run mode, no downloads")
        return

    results = download_range(
        start.date(), end.date(), raw_dir, workers=workers, done_dates=done_dates
    )
    ok = sum(1 for _, s, _, _ in results if s)
    fail = sum(1 for _, s, _, _ in results if not s)
    for date_str, success, msg, _ in results:
        state = "done" if success else "failed"
        manifest.update(date_str, state, run_id=run_id, message=msg)
    manifest.flush()
    click.echo(f"GDAS ingest done: {ok} ok, {fail} failed")


def _ingest_raws(start, end, raw_dir, bounds, resume, dry_run, **_kw):
    """Scrape RAWS WRCC inventory and download daily data."""
    from pathlib import Path

    from obsmet.core.manifest import Manifest
    from obsmet.core.provenance import generate_run_id
    from obsmet.sources.raws_wrcc.download import (
        DEFAULT_RAW_DIR,
        REGION_PAGES,
        download_station_data,
        scrape_region_stations,
    )
    from obsmet.sources.raws_wrcc.extract import parse_response

    raw_dir = Path(raw_dir) if raw_dir else Path(DEFAULT_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source="raws_wrcc")
    run_id = generate_run_id()

    if start is None or end is None:
        click.echo("Error: --start and --end are required for raws ingest.")
        return

    click.echo("RAWS: scraping station inventory...")
    all_stations = []
    for region in REGION_PAGES:
        stations = scrape_region_stations(region)
        all_stations.extend(stations)
    click.echo(f"  found {len(all_stations)} stations across {len(REGION_PAGES)} regions")

    if dry_run:
        click.echo(f"  would download data for {len(all_stations)} stations")
        return

    ok = 0
    fail = 0
    for stn in all_stations:
        wrcc_id = stn["wrcc_id"]
        if resume and not manifest.pending_keys([wrcc_id]):
            continue
        try:
            html = download_station_data(wrcc_id, start.date(), end.date())
            df = parse_response(html)
            if df.empty:
                manifest.update(wrcc_id, "missing", run_id=run_id, message="no data")
                fail += 1
            else:
                out_path = raw_dir / f"{wrcc_id}.parquet"
                df.to_parquet(out_path, index=False, compression="snappy")
                manifest.update(wrcc_id, "done", run_id=run_id)
                ok += 1
        except Exception as exc:
            manifest.update(wrcc_id, "failed", run_id=run_id, message=str(exc))
            fail += 1
        if (ok + fail) % 50 == 0:
            click.echo(f"  progress: {ok + fail}/{len(all_stations)}, {fail} failures")
            manifest.flush()

    manifest.flush()
    click.echo(f"RAWS ingest done: {ok} ok, {fail} failed")


def _ingest_ndbc(start, end, raw_dir, resume, dry_run, **_kw):
    """Download NDBC stdmet files."""
    from pathlib import Path

    from obsmet.core.manifest import Manifest
    from obsmet.core.provenance import generate_run_id
    from obsmet.sources.ndbc.download import (
        DEFAULT_RAW_DIR,
        download_station_year,
        get_ndbc_stations,
    )

    raw_dir = Path(raw_dir) if raw_dir else Path(DEFAULT_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source="ndbc")
    run_id = generate_run_id()

    if start is None or end is None:
        click.echo("Error: --start and --end are required for ndbc ingest.")
        return

    click.echo("NDBC: fetching station metadata...")
    stations = get_ndbc_stations()
    click.echo(f"  found {len(stations)} stations")

    if dry_run:
        click.echo(f"  would download {start.year}-{end.year} for {len(stations)} stations")
        return

    ok = 0
    fail = 0
    for _, row in stations.iterrows():
        sid = row["station_id"]
        for year in range(start.year, end.year + 1):
            key = f"{sid}_{year}"
            if resume and not manifest.pending_keys([key]):
                continue
            _, _, success, msg = download_station_year(sid, year, raw_dir)
            state = "done" if success else ("missing" if msg == "not_found" else "failed")
            manifest.update(key, state, run_id=run_id, message=msg)
            if success:
                ok += 1
            else:
                fail += 1

        if (ok + fail) % 200 == 0:
            click.echo(f"  progress: {ok + fail}, {fail} failed")
            manifest.flush()

    manifest.flush()
    click.echo(f"NDBC ingest done: {ok} ok, {fail} failed/missing")


# --------------------------------------------------------------------------- #
# Station POR product
# --------------------------------------------------------------------------- #


def _build_station_por(source, start, end, resume, workers, overwrite, dry_run):
    """Build station period-of-record parquets from normalized data."""
    from pathlib import Path

    from obsmet.core.provenance import RunProvenance
    from obsmet.products.station_por import build_station_por

    if source == "all":
        sources = ["madis", "isd", "gdas", "ndbc"]
    else:
        sources = [source]

    start_date = start.date() if start else None
    end_date = end.date() if end else None

    for src in sources:
        norm_dir = Path(_DEFAULT_NORM_DIRS.get(src, f"/nas/climate/obsmet/normalized/{src}"))
        out_dir = Path(f"/mnt/mco_nas1/shared/obsmet/products/station_por/{src}")

        if dry_run:
            click.echo(f"  {src}: would build station POR from {norm_dir} → {out_dir}")
            continue

        provenance = RunProvenance(source=src, command="build_station_por")
        click.echo(f"  {src}: building station POR → {out_dir}")

        stats = build_station_por(
            src, norm_dir, out_dir, provenance, start_date=start_date, end_date=end_date
        )
        click.echo(f"  {src}: {len(stats)} stations written")


# --------------------------------------------------------------------------- #
# Daily aggregation
# --------------------------------------------------------------------------- #


def _build_daily(source, start, end, resume, workers, overwrite, dry_run):
    """Aggregate normalized hourly observations to daily products."""
    from pathlib import Path

    import pandas as pd

    from obsmet.core.provenance import RunProvenance
    from obsmet.core.time_policy import DAILY_SOURCES, aggregate_daily_wide
    from obsmet.products.daily import write_daily

    sources = DAILY_SOURCES if source == "all" else [source]
    base_dir = Path("/mnt/mco_nas1/shared/obsmet/normalized")
    out_base = Path("/mnt/mco_nas1/shared/obsmet/products/daily")
    provenance = RunProvenance(source=source, command="build_daily")

    start_date = start.date() if start else None
    end_date = end.date() if end else None

    click.echo(f"Build daily: sources={sources}, {start} to {end}")

    if dry_run:
        for src in sources:
            src_dir = base_dir / src
            if src_dir.exists():
                n = len(list(src_dir.glob("*.parquet")))
                click.echo(f"  {src}: {n} parquet files to aggregate")
        return

    for src in sources:
        src_dir = base_dir / src
        out_dir = out_base / src
        out_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            click.echo(f"  {src}: normalized dir not found, skipping")
            continue

        parquet_files = sorted(src_dir.glob("*.parquet"))
        parquet_files = [p for p in parquet_files if p.name != "manifest.parquet"]
        click.echo(f"  {src}: aggregating {len(parquet_files)} files")

        for pf in parquet_files:
            out_path = out_dir / pf.name
            if out_path.exists() and not overwrite:
                continue

            try:
                df = pd.read_parquet(pf)
            except Exception:
                continue

            if df.empty or "datetime_utc" not in df.columns:
                continue

            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
            df["date"] = df["datetime_utc"].dt.date

            if start_date:
                df = df[df["date"] >= start_date]
            if end_date:
                df = df[df["date"] <= end_date]

            if df.empty:
                continue

            daily_df = aggregate_daily_wide(df, provenance)
            if not daily_df.empty:
                write_daily(daily_df, out_path)

    click.echo("Build daily done.")


if __name__ == "__main__":
    cli()
