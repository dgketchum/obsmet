"""Hourly canonical product builder (plan Layer D)."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from obsmet.core.schema import OBS_HOURLY_SCHEMA


def write_hourly(df, dest: Path | str) -> Path:
    """Write a canonical hourly observation DataFrame to parquet.

    Validates column presence against OBS_HOURLY_SCHEMA before writing.
    """
    import pyarrow as pa

    dest = Path(dest)
    expected = {f.name for f in OBS_HOURLY_SCHEMA}
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=OBS_HOURLY_SCHEMA)
    pq.write_table(table, dest)
    return dest
