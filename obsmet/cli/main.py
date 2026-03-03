"""obsmet CLI entry point (plan section 17).

All commands support --start, --end, --resume, --workers, --overwrite, --dry-run.
"""

from __future__ import annotations

import click


def common_options(fn):
    """Shared options across all pipeline commands."""
    fn = click.option(
        "--start", type=click.DateTime(), default=None, help="Start date"
    )(fn)
    fn = click.option("--end", type=click.DateTime(), default=None, help="End date")(fn)
    fn = click.option(
        "--resume/--no-resume", default=True, help="Resume from manifest state"
    )(fn)
    fn = click.option("--workers", type=int, default=4, help="Parallel workers")(fn)
    fn = click.option("--overwrite", is_flag=True, help="Overwrite existing outputs")(
        fn
    )
    fn = click.option("--dry-run", is_flag=True, help="Show what would be done")(fn)
    return fn


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """obsmet — unified meteorological observation pipeline."""


@cli.command()
@click.argument("source")
@common_options
def ingest(source, start, end, resume, workers, overwrite, dry_run):
    """Download/scrape raw data for SOURCE."""
    click.echo(
        f"ingest {source}: start={start} end={end} resume={resume} "
        f"workers={workers} overwrite={overwrite} dry_run={dry_run}"
    )


@cli.command()
@click.argument("source")
@common_options
def normalize(source, start, end, resume, workers, overwrite, dry_run):
    """Parse raw files into canonical observation rows for SOURCE."""
    click.echo(f"normalize {source}: start={start} end={end}")


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


if __name__ == "__main__":
    cli()
