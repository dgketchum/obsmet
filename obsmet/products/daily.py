"""Daily aggregated product builder (plan Layer D)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from obsmet.core.schema import DAILY_METRIC_FIELDS, OBS_DAILY_CORE_SCHEMA


def write_daily(df, dest: Path | str) -> Path:
    """Write a canonical daily observation DataFrame to parquet.

    Validates core columns against OBS_DAILY_CORE_SCHEMA before writing.
    Metric columns (tmax, tmin, etc.) are validated if present.
    """
    dest = Path(dest)
    core_cols = {f.name for f in OBS_DAILY_CORE_SCHEMA}
    actual = set(df.columns)
    missing = core_cols - actual
    if missing:
        raise ValueError(f"Missing required core columns: {missing}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Build schema: core + any metric fields present
    extra_fields = [f for f in DAILY_METRIC_FIELDS if f.name in actual]
    schema = pa.schema(list(OBS_DAILY_CORE_SCHEMA) + extra_fields)

    # Only keep columns that are in the schema
    keep = [f.name for f in schema]
    df_out = df[[c for c in keep if c in df.columns]]

    table = pa.Table.from_pandas(df_out, schema=schema)
    pq.write_table(table, dest)
    return dest
