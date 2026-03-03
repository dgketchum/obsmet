"""Manifest engine for tracking ingest/processing state (plan section 7A).

Each source maintains a manifest parquet with per-key states:
done, missing, suspect, failed, skipped.

Supports resume semantics: only unfinished keys are reprocessed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from obsmet.core.schema import MANIFEST_SCHEMA, MANIFEST_STATES


class Manifest:
    """Parquet-backed manifest for tracking per-key processing state."""

    def __init__(self, path: Path | str, source: str) -> None:
        self.path = Path(path)
        self.source = source
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        if self.path.exists():
            self._df = pq.read_table(self.path).to_pandas()
        else:
            self._df = pd.DataFrame(columns=[f.name for f in MANIFEST_SCHEMA])
        return self._df

    def get_state(self, key: str) -> str | None:
        """Return the current state for a key, or None if not tracked."""
        df = self._load()
        mask = (df["source"] == self.source) & (df["key"] == key)
        rows = df.loc[mask]
        if rows.empty:
            return None
        return rows.iloc[-1]["state"]

    def pending_keys(self, all_keys: list[str]) -> list[str]:
        """Return keys that are not yet 'done'."""
        df = self._load()
        done = set(df.loc[(df["source"] == self.source) & (df["state"] == "done"), "key"])
        return [k for k in all_keys if k not in done]

    def update(self, key: str, state: str, *, run_id: str = "", message: str = "") -> None:
        """Record or update a key's state."""
        if state not in MANIFEST_STATES:
            raise ValueError(f"Invalid state {state!r}; must be one of {MANIFEST_STATES}")
        df = self._load()
        now = pd.Timestamp(datetime.now(timezone.utc))
        new_row = pd.DataFrame(
            [
                {
                    "source": self.source,
                    "key": key,
                    "state": state,
                    "updated_utc": now,
                    "run_id": run_id,
                    "message": message,
                }
            ]
        )
        # Drop any existing row for this source+key, then append
        mask = (df["source"] == self.source) & (df["key"] == key)
        df = pd.concat([df[~mask], new_row], ignore_index=True)
        self._df = df

    def flush(self) -> None:
        """Write manifest to disk."""
        df = self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, schema=MANIFEST_SCHEMA)
        pq.write_table(table, self.path)

    def summary(self) -> dict[str, int]:
        """Return count of keys by state for this source."""
        df = self._load()
        src = df.loc[df["source"] == self.source]
        return dict(src["state"].value_counts())
