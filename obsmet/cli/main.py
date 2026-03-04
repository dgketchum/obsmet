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
    if source == "madis":
        _ingest_madis(start, end, raw_dir, resume, dry_run)
    elif source == "isd":
        _ingest_isd(start, end, raw_dir, resume, workers, dry_run)
    elif source == "gdas":
        _ingest_gdas(start, end, raw_dir, resume, workers, dry_run)
    elif source == "raws":
        _ingest_raws(start, end, raw_dir, bounds, resume, dry_run)
    elif source == "ndbc":
        _ingest_ndbc(start, end, raw_dir, resume, dry_run)
    else:
        click.echo(f"Source {source!r} not yet implemented.")


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
@common_options
def normalize(
    source, raw_dir, out_dir, bounds, qcr_mask, start, end, resume, workers, overwrite, dry_run
):
    """Parse raw files into canonical observation rows for SOURCE."""
    if source == "madis":
        _normalize_madis(
            start, end, raw_dir, out_dir, bounds, qcr_mask, resume, workers, overwrite, dry_run
        )
    elif source == "isd":
        _normalize_isd(start, end, raw_dir, out_dir, resume, workers, overwrite, dry_run)
    elif source == "gdas":
        _normalize_gdas(start, end, raw_dir, out_dir, resume, workers, overwrite, dry_run)
    elif source == "raws":
        _normalize_raws(start, end, raw_dir, out_dir, resume, overwrite, dry_run)
    elif source == "ndbc":
        _normalize_ndbc(start, end, raw_dir, out_dir, resume, workers, overwrite, dry_run)
    else:
        click.echo(f"Source {source!r} not yet implemented.")


@cli.command()
@click.argument("source", default="all")
@common_options
def qaqc(source, start, end, resume, workers, overwrite, dry_run):
    """Apply QAQC rules to SOURCE (or all sources)."""
    click.echo(f"qaqc {source}: start={start} end={end}")


@cli.command()
@click.argument("product", type=click.Choice(["hourly", "daily", "fabric"]))
@click.option(
    "--source",
    default="all",
    help="Source to build from (default: all)",
)
@common_options
def build(product, source, start, end, resume, workers, overwrite, dry_run):
    """Build curated PRODUCT (hourly, daily, or fabric)."""
    if product == "daily":
        _build_daily(source, start, end, resume, workers, overwrite, dry_run)
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
# MADIS implementation
# --------------------------------------------------------------------------- #


def _ingest_madis(start, end, raw_dir, resume, dry_run):
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

    # Filter by manifest if resuming
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


def _madis_normalize_one(day_str, raw_dir, bounds, qcr_mask, out_dir, run_id, overwrite):
    """Module-level worker for MADIS normalize (picklable for ProcessPoolExecutor)."""
    from pathlib import Path

    from obsmet.core.provenance import RunProvenance
    from obsmet.sources.madis.adapter import MadisAdapter

    adapter = MadisAdapter(raw_dir=raw_dir, bounds=bounds, qcr_mask=qcr_mask)
    provenance = RunProvenance(source="madis", command="normalize", run_id=run_id)
    out_dir = Path(out_dir)

    out_path = out_dir / f"{day_str}.parquet"
    if out_path.exists() and not overwrite:
        return day_str, None, None

    df = adapter.extract_and_normalize_day(day_str, provenance, wide=True)
    if df is None or df.empty:
        return day_str, 0, "no data"

    df.to_parquet(out_path, index=False, compression="snappy")
    return day_str, len(df), None


def _normalize_madis(
    start, end, raw_dir, out_dir, bounds, qcr_mask, resume, workers, overwrite, dry_run
):
    """Extract, QC, and normalize MADIS data for a date range."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from datetime import timedelta
    from functools import partial
    from pathlib import Path

    from obsmet.core.manifest import Manifest
    from obsmet.core.provenance import RunProvenance
    from obsmet.sources.madis.extract import QCR_REJECT_BITS

    raw_dir = raw_dir or "/nas/climate/obsmet/raw/madis"
    out_dir = Path(out_dir) if out_dir else Path("/nas/climate/obsmet/normalized/madis")
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_bounds = None
    if bounds:
        parsed_bounds = tuple(float(x) for x in bounds.split(","))

    mask = qcr_mask if qcr_mask is not None else QCR_REJECT_BITS
    provenance = RunProvenance(source="madis", command="normalize")

    manifest_path = out_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source="madis")

    if start is None or end is None:
        click.echo("Error: --start and --end are required for madis normalize.")
        return

    # Build day list
    days = []
    current = start
    while current <= end:
        days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    if resume:
        days = manifest.pending_keys(days)

    click.echo(f"MADIS normalize: {len(days)} days, {workers} workers → {out_dir}")

    if dry_run:
        for ds in days[:10]:
            click.echo(f"  would normalize: {ds}")
        if len(days) > 10:
            click.echo(f"  ... and {len(days) - 10} more")
        return

    worker = partial(
        _madis_normalize_one,
        raw_dir=raw_dir,
        bounds=parsed_bounds,
        qcr_mask=mask,
        out_dir=str(out_dir),
        run_id=provenance.run_id,
        overwrite=overwrite,
    )

    done = 0
    errors = 0
    skipped = 0

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, d): d for d in days}
            for fut in as_completed(futures):
                day_str = futures[fut]
                try:
                    _, nrows, err = fut.result()
                except Exception as exc:
                    click.echo(f"  EXCEPTION {day_str}: {exc}")
                    manifest.update(day_str, "failed", run_id=provenance.run_id, message=str(exc))
                    errors += 1
                    done += 1
                    continue

                if nrows is None:
                    skipped += 1
                elif err:
                    manifest.update(day_str, "missing", run_id=provenance.run_id, message=err)
                    errors += 1
                else:
                    manifest.update(day_str, "done", run_id=provenance.run_id)
                done += 1

                if done % 50 == 0:
                    click.echo(f"  progress: {done}/{len(days)}, {skipped} skip, {errors} err")
                    manifest.flush()
    else:
        for day_str in days:
            try:
                _, nrows, err = worker(day_str)
            except Exception as exc:
                click.echo(f"  EXCEPTION {day_str}: {exc}")
                manifest.update(day_str, "failed", run_id=provenance.run_id, message=str(exc))
                errors += 1
                done += 1
                continue

            if nrows is None:
                skipped += 1
            elif err:
                manifest.update(day_str, "missing", run_id=provenance.run_id, message=err)
                errors += 1
            else:
                manifest.update(day_str, "done", run_id=provenance.run_id)
            done += 1

            if done % 50 == 0:
                click.echo(f"  progress: {done}/{len(days)}, {skipped} skip, {errors} err")
                manifest.flush()

    manifest.flush()
    click.echo(f"MADIS normalize done: {done} processed, {skipped} skipped, {errors} errors")


# --------------------------------------------------------------------------- #
# ISD implementation
# --------------------------------------------------------------------------- #


def _ingest_isd(start, end, raw_dir, resume, workers, dry_run):
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


def _isd_normalize_one(key, raw_dir, out_dir, run_id, overwrite):
    """Module-level worker for ISD normalize (picklable for ProcessPoolExecutor)."""
    from pathlib import Path

    from obsmet.core.provenance import RunProvenance
    from obsmet.sources.isd.adapter import IsdAdapter

    adapter = IsdAdapter(raw_dir=raw_dir)
    provenance = RunProvenance(source="isd", command="normalize", run_id=run_id)
    out_dir = Path(out_dir)

    raw_path = adapter.fetch_raw(key, out_dir)
    out_path = out_dir / key.replace("/", "_").replace(".gz", ".parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return key, None, None
    df = adapter.normalize(raw_path, provenance)
    if df is None or df.empty:
        return key, 0, "no data"
    df.to_parquet(out_path, index=False, compression="snappy")
    return key, len(df), None


def _normalize_isd(start, end, raw_dir, out_dir, resume, workers, overwrite, dry_run):
    """Parse ISD .gz files into canonical wide parquet."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial
    from pathlib import Path

    from obsmet.core.manifest import Manifest
    from obsmet.core.provenance import RunProvenance
    from obsmet.sources.isd.adapter import IsdAdapter

    raw_dir = raw_dir or "/nas/climate/isd/raw"
    out_dir = Path(out_dir) if out_dir else Path("/nas/climate/obsmet/normalized/isd")
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = IsdAdapter(raw_dir=raw_dir)
    provenance = RunProvenance(source="isd", command="normalize")

    manifest_path = out_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source="isd")

    if start is None or end is None:
        click.echo("Error: --start and --end are required for isd normalize.")
        return

    keys = adapter.discover_keys(start, end)
    if resume:
        keys = manifest.pending_keys(keys)

    click.echo(f"ISD normalize: {len(keys)} files, {workers} workers → {out_dir}")

    if dry_run:
        for k in keys[:10]:
            click.echo(f"  would normalize: {k}")
        if len(keys) > 10:
            click.echo(f"  ... and {len(keys) - 10} more")
        return

    worker = partial(
        _isd_normalize_one,
        raw_dir=raw_dir,
        out_dir=str(out_dir),
        run_id=provenance.run_id,
        overwrite=overwrite,
    )

    done = 0
    errors = 0
    skipped = 0

    if workers > 1:
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
    click.echo(f"ISD normalize done: {done} processed, {skipped} skipped, {errors} errors")


# --------------------------------------------------------------------------- #
# GDAS implementation
# --------------------------------------------------------------------------- #


def _ingest_gdas(start, end, raw_dir, resume, workers, dry_run):
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


def _gdas_normalize_one(day_str, raw_dir, out_dir, run_id, overwrite):
    """Module-level worker for GDAS normalize (picklable for ProcessPoolExecutor)."""
    from pathlib import Path

    from obsmet.core.provenance import RunProvenance
    from obsmet.sources.gdas_prepbufr.adapter import GdasAdapter

    adapter = GdasAdapter(raw_dir=raw_dir)
    provenance = RunProvenance(source="gdas", command="normalize", run_id=run_id)
    out_dir = Path(out_dir)

    raw_path = adapter.fetch_raw(day_str, out_dir)
    if not raw_path.exists():
        return day_str, 0, "raw file not found"
    out_path = out_dir / f"{day_str}.parquet"
    if out_path.exists() and not overwrite:
        return day_str, None, None
    df = adapter.normalize(raw_path, provenance)
    if df is None or df.empty:
        return day_str, 0, "no data"
    df.to_parquet(out_path, index=False, compression="snappy")
    return day_str, len(df), None


def _normalize_gdas(start, end, raw_dir, out_dir, resume, workers, overwrite, dry_run):
    """Extract and normalize GDAS PrepBUFR archives."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial
    from pathlib import Path

    from obsmet.core.manifest import Manifest
    from obsmet.core.provenance import RunProvenance
    from obsmet.sources.gdas_prepbufr.adapter import GdasAdapter

    raw_dir = raw_dir or "/nas/climate/gdas/prepbufr"
    out_dir = Path(out_dir) if out_dir else Path("/nas/climate/obsmet/normalized/gdas")
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = GdasAdapter(raw_dir=raw_dir)
    provenance = RunProvenance(source="gdas", command="normalize")

    manifest_path = out_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source="gdas")

    if start is None or end is None:
        click.echo("Error: --start and --end are required for gdas normalize.")
        return

    days = adapter.discover_keys(start, end)
    if resume:
        days = manifest.pending_keys(days)

    click.echo(f"GDAS normalize: {len(days)} days, {workers} workers → {out_dir}")

    if dry_run:
        for ds in days[:10]:
            click.echo(f"  would normalize: {ds}")
        if len(days) > 10:
            click.echo(f"  ... and {len(days) - 10} more")
        return

    worker = partial(
        _gdas_normalize_one,
        raw_dir=raw_dir,
        out_dir=str(out_dir),
        run_id=provenance.run_id,
        overwrite=overwrite,
    )

    done = 0
    errors = 0
    skipped = 0

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, d): d for d in days}
            for fut in as_completed(futures):
                day_str = futures[fut]
                try:
                    _, nrows, err = fut.result()
                except Exception as exc:
                    manifest.update(day_str, "failed", run_id=provenance.run_id, message=str(exc))
                    errors += 1
                    done += 1
                    continue
                if nrows is None:
                    skipped += 1
                elif err:
                    manifest.update(day_str, "missing", run_id=provenance.run_id, message=err)
                    errors += 1
                else:
                    manifest.update(day_str, "done", run_id=provenance.run_id)
                done += 1
                if done % 50 == 0:
                    click.echo(f"  progress: {done}/{len(days)}, {skipped} skip, {errors} err")
                    manifest.flush()
    else:
        for day_str in days:
            try:
                _, nrows, err = worker(day_str)
            except Exception as exc:
                manifest.update(day_str, "failed", run_id=provenance.run_id, message=str(exc))
                errors += 1
                done += 1
                continue
            if nrows is None:
                skipped += 1
            elif err:
                manifest.update(day_str, "missing", run_id=provenance.run_id, message=err)
                errors += 1
            else:
                manifest.update(day_str, "done", run_id=provenance.run_id)
            done += 1
            if done % 50 == 0:
                click.echo(f"  progress: {done}/{len(days)}, {skipped} skip, {errors} err")
                manifest.flush()

    manifest.flush()
    click.echo(f"GDAS normalize done: {done} processed, {skipped} skipped, {errors} errors")


# --------------------------------------------------------------------------- #
# RAWS implementation
# --------------------------------------------------------------------------- #


def _ingest_raws(start, end, raw_dir, bounds, resume, dry_run):
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

    # Build station inventory
    click.echo("RAWS: scraping station inventory...")
    all_stations = []
    for region in REGION_PAGES:
        stations = scrape_region_stations(region)
        all_stations.extend(stations)
    click.echo(f"  found {len(all_stations)} stations across {len(REGION_PAGES)} regions")

    if dry_run:
        click.echo(f"  would download data for {len(all_stations)} stations")
        return

    # Download data for each station
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


def _normalize_raws(start, end, raw_dir, out_dir, resume, overwrite, dry_run):
    """Normalize RAWS parquet files to canonical schema."""
    from pathlib import Path

    from obsmet.core.manifest import Manifest
    from obsmet.core.provenance import RunProvenance
    from obsmet.sources.raws_wrcc.adapter import RawsAdapter

    raw_dir = raw_dir or "/nas/climate/obsmet/raw/raws_wrcc"
    out_dir = Path(out_dir) if out_dir else Path("/nas/climate/obsmet/normalized/raws_wrcc")
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = RawsAdapter(raw_dir=raw_dir)
    provenance = RunProvenance(source="raws_wrcc", command="normalize")

    manifest_path = out_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source="raws_wrcc")

    keys = adapter.discover_keys(start, end)
    if resume:
        keys = manifest.pending_keys(keys)

    click.echo(f"RAWS normalize: {len(keys)} stations → {out_dir}")

    if dry_run:
        for k in keys[:10]:
            click.echo(f"  would normalize: {k}")
        return

    done = 0
    errors = 0
    for key in keys:
        try:
            raw_path = adapter.fetch_raw(key, out_dir)
            if not raw_path.exists():
                manifest.update(key, "missing", run_id=provenance.run_id)
                errors += 1
                continue
            out_path = out_dir / f"{key}.parquet"
            if out_path.exists() and not overwrite:
                done += 1
                continue
            df = adapter.normalize(raw_path, provenance)
            if df.empty:
                manifest.update(key, "missing", run_id=provenance.run_id)
                errors += 1
            else:
                df.to_parquet(out_path, index=False, compression="snappy")
                manifest.update(key, "done", run_id=provenance.run_id)
            done += 1
        except Exception as exc:
            manifest.update(key, "failed", run_id=provenance.run_id, message=str(exc))
            errors += 1
            done += 1

    manifest.flush()
    click.echo(f"RAWS normalize done: {done} processed, {errors} errors")


# --------------------------------------------------------------------------- #
# NDBC implementation
# --------------------------------------------------------------------------- #


def _ingest_ndbc(start, end, raw_dir, resume, dry_run):
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


def _ndbc_normalize_one(station_id, raw_dir, out_dir, run_id, overwrite):
    """Module-level worker for NDBC normalize (picklable for ProcessPoolExecutor)."""
    from pathlib import Path

    from obsmet.core.provenance import RunProvenance
    from obsmet.sources.ndbc.adapter import normalize_to_canonical_wide
    from obsmet.sources.ndbc.extract import read_station_files

    provenance = RunProvenance(source="ndbc", command="normalize", run_id=run_id)
    out_dir = Path(out_dir)

    out_path = out_dir / f"{station_id}.parquet"
    if out_path.exists() and not overwrite:
        return station_id, None, None
    df = read_station_files(Path(raw_dir), station_id)
    if df.empty:
        return station_id, 0, "no data"
    wide = normalize_to_canonical_wide(df, station_id, provenance)
    wide.to_parquet(out_path, index=False, compression="snappy")
    return station_id, len(wide), None


def _normalize_ndbc(start, end, raw_dir, out_dir, resume, workers, overwrite, dry_run):
    """Parse NDBC stdmet files into canonical wide parquet."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial
    from pathlib import Path

    from obsmet.core.manifest import Manifest
    from obsmet.core.provenance import RunProvenance
    from obsmet.sources.ndbc.adapter import NdbcAdapter

    raw_dir = raw_dir or "/nas/climate/obsmet/raw/ndbc"
    out_dir = Path(out_dir) if out_dir else Path("/nas/climate/obsmet/normalized/ndbc")
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = NdbcAdapter(raw_dir=raw_dir)
    provenance = RunProvenance(source="ndbc", command="normalize")

    manifest_path = out_dir / "manifest.parquet"
    manifest = Manifest(manifest_path, source="ndbc")

    keys = adapter.discover_keys(start, end)
    if resume:
        keys = manifest.pending_keys(keys)

    click.echo(f"NDBC normalize: {len(keys)} stations, {workers} workers → {out_dir}")

    if dry_run:
        for k in keys[:10]:
            click.echo(f"  would normalize: {k}")
        return

    worker = partial(
        _ndbc_normalize_one,
        raw_dir=raw_dir,
        out_dir=str(out_dir),
        run_id=provenance.run_id,
        overwrite=overwrite,
    )

    done = 0
    errors = 0
    skipped = 0

    if workers > 1:
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

    manifest.flush()
    click.echo(f"NDBC normalize done: {done} processed, {skipped} skipped, {errors} errors")


# --------------------------------------------------------------------------- #
# Daily aggregation
# --------------------------------------------------------------------------- #


_DAILY_SOURCES = ["madis", "isd", "gdas", "ndbc"]

# Variables and their aggregation type for daily rollup from hourly data
_DAILY_AGG_MAP = {
    "tair": "mean",
    "td": "mean",
    "rh": "mean",
    "wind": "mean",
    "wind_dir": "circular_mean",
    "slp": "mean",
    "prcp": "sum",
}


def _build_daily(source, start, end, resume, workers, overwrite, dry_run):
    """Aggregate normalized hourly observations to daily products."""
    from pathlib import Path

    import pandas as pd

    from obsmet.core.provenance import RunProvenance
    from obsmet.core.time_policy import circular_mean_deg, hourly_coverage
    from obsmet.products.daily import write_daily

    sources = _DAILY_SOURCES if source == "all" else [source]
    base_dir = Path("/nas/climate/obsmet/normalized")
    out_base = Path("/nas/climate/obsmet/products/daily")
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

            # Date filtering
            if start_date:
                df = df[df["date"] >= start_date]
            if end_date:
                df = df[df["date"] <= end_date]

            if df.empty or "station_key" not in df.columns:
                continue

            groups = df.groupby(["station_key", "date"])

            daily_records = []
            for (stn, day), grp in groups:
                cov = hourly_coverage(grp["datetime_utc"], day)
                cov_str = (
                    f"n={cov['obs_count']}"
                    f",thresh={'Y' if cov['meets_threshold'] else 'N'}"
                    f",am={'Y' if cov['morning_ok'] else 'N'}"
                    f",pm={'Y' if cov['afternoon_ok'] else 'N'}"
                )

                rec = {
                    "station_key": stn,
                    "date": day,
                    "day_basis": "utc",
                    "obs_count": len(grp),
                    "coverage_flags": cov_str,
                    "qc_state": "pass",
                    "qc_rules_version": provenance.qaqc_rules_version,
                    "transform_version": provenance.transform_version,
                    "ingest_run_id": provenance.run_id,
                }

                for var, agg_type in _DAILY_AGG_MAP.items():
                    if var not in grp.columns:
                        continue
                    vals = pd.to_numeric(grp[var], errors="coerce").dropna()
                    if vals.empty:
                        continue
                    if agg_type == "mean":
                        rec[var] = float(vals.mean())
                    elif agg_type == "sum":
                        rec[var] = float(vals.sum())
                    elif agg_type == "circular_mean":
                        rec[var] = circular_mean_deg(vals)

                # Tmax/tmin/tmean from tair
                if "tair" in grp.columns:
                    tair_vals = pd.to_numeric(grp["tair"], errors="coerce").dropna()
                    if not tair_vals.empty:
                        rec["tmax"] = float(tair_vals.max())
                        rec["tmin"] = float(tair_vals.min())
                        rec["tmean"] = float(tair_vals.mean())

                # Convert rsds_hourly W/m² to rsds MJ/m²/day
                if "rsds_hourly" in grp.columns:
                    rsds_vals = pd.to_numeric(grp["rsds_hourly"], errors="coerce").dropna()
                    if not rsds_vals.empty:
                        rec["rsds"] = float(rsds_vals.mean()) * 86400.0 / 1e6

                daily_records.append(rec)

            if daily_records:
                daily_df = pd.DataFrame(daily_records)
                daily_df["date"] = pd.to_datetime(daily_df["date"])
                write_daily(daily_df, out_path)

    click.echo("Build daily done.")


if __name__ == "__main__":
    cli()
