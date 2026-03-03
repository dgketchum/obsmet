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
@common_options
def build(product, start, end, resume, workers, overwrite, dry_run):
    """Build curated PRODUCT (hourly, daily, or fabric)."""
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


def _normalize_madis(
    start, end, raw_dir, out_dir, bounds, qcr_mask, resume, workers, overwrite, dry_run
):
    """Extract, QC, and normalize MADIS data for a date range."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from datetime import timedelta
    from pathlib import Path

    from obsmet.core.manifest import Manifest
    from obsmet.core.provenance import RunProvenance
    from obsmet.sources.madis.adapter import MadisAdapter
    from obsmet.sources.madis.extract import QCR_REJECT_BITS

    raw_dir = raw_dir or "/nas/climate/obsmet/raw/madis"
    out_dir = Path(out_dir) if out_dir else Path("/nas/climate/obsmet/normalized/madis")
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_bounds = None
    if bounds:
        parsed_bounds = tuple(float(x) for x in bounds.split(","))

    mask = qcr_mask if qcr_mask is not None else QCR_REJECT_BITS
    adapter = MadisAdapter(raw_dir=raw_dir, bounds=parsed_bounds, qcr_mask=mask)
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

    def _process_one(day_str):
        out_path = out_dir / f"{day_str}.parquet"
        if out_path.exists() and not overwrite:
            return day_str, None, None

        df = adapter.extract_and_normalize_day(day_str, provenance, wide=True)
        if df is None or df.empty:
            return day_str, 0, "no data"

        df.to_parquet(out_path, index=False, compression="snappy")
        return day_str, len(df), None

    done = 0
    errors = 0
    skipped = 0

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, d): d for d in days}
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
                _, nrows, err = _process_one(day_str)
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


if __name__ == "__main__":
    cli()
